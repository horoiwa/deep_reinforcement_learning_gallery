from dataclasses import dataclass
from pathlib import Path
import shutil
import random
import datetime

import ray
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np
import gym

from buffer import SequenceReplayBuffer, EpisodeBuffer
from networks import PolicyNetwork, ValueNetwork, WorldModel
import util


@dataclass
class Config:

    batch_size: int = 20           # Batch size, B
    sequence_length: int = 20      # Sequence Lenght, L
    buffer_size: int = 100000       # Replay buffer size (FIFO)
    num_minibatchs: int = 10
    gamma: float = 0.997

    kl_scale: float = 0.1     # KL loss scale, β
    kl_alpha: float = 0.8          # KL balancing
    latent_dim: int = 24           # discrete latent dimensions
    n_atoms: int = 24              # discrete latent classes
    lr_world: float = 2e-4         # learning rate of world model

    imagination_horizon: int = 8   # Imagination horizon, H
    gamma_discount: float = 0.995  # discount factor γ
    lambda_gae: float = 0.95       # λ for Generalized advantage estimator
    ent_scale: float = 5e-3
    lr_actor: float = 4e-5
    lr_critic: float = 1e-4

    adam_epsilon: float = 1e-5
    adam_decay: float = 1e-6
    grad_clip: float = 100.


class DreamerV2Agent:

    def __init__(self, env_id: str, config: Config,
                 pid: int = None, epsilon: float = 0.,
                 summary_writer: tf.summary.SummaryWriter = None):

        self.env_id = env_id

        self.config = config

        self.pid = pid

        self.epsilon = epsilon

        self.summary_writer = summary_writer

        self.action_space = gym.make(self.env_id).action_space.n

        self.preprocess_func = util.get_preprocess_func(env_name=self.env_id)

        self.buffer = EpisodeBuffer(seqlen=self.config.sequence_length)

        self.world_model = WorldModel(config)
        self.wm_optimizer = tf.keras.optimizers.Adam(
            lr=self.config.lr_world, epsilon=1e-4)

        self.policy = PolicyNetwork(action_space=self.action_space)
        self.policy_optimizer = tf.keras.optimizers.Adam(
            lr=self.config.lr_actor, epsilon=1e-5)

        self.value = ValueNetwork(action_space=self.action_space)
        self.target_value = ValueNetwork(action_space=self.action_space)
        self.value_optimizer = tf.keras.optimizers.Adam(
            lr=self.config.lr_critic, epsilon=1e-5)

        self.setup()

    def setup(self):
        """ Build network weights """
        env = gym.make(self.env_id)
        obs = self.preprocess_func(env.reset())
        prev_z, prev_h = self.world_model.get_initial_state(batch_size=1)
        prev_a = tf.one_hot([0], self.action_space)
        _outputs = self.world_model(obs, prev_z, prev_h, prev_a)
        (h, z_prior, z_prior_prob, z_post, z_post_prob,
         feat, img_out, reward_pred, disc_logit) = _outputs
        self.policy(feat)
        self.value(feat)
        self.target_value(feat)
        self.target_value.set_weights(self.value.get_weights())

    def save(self, savedir=None):
        savedir = Path(savedir) if savedir is not None else Path("./checkpoints")
        self.world_model.save_weights(str(savedir / "worldmodel"))
        self.policy.save_weights(str(savedir / "policy"))
        self.value.save_weights(str(savedir / "critic"))

    def load(self, loaddir=None):
        loaddir = Path(loaddir) if loaddir is not None else Path("checkpoints")
        self.world_model.load_weights(str(loaddir / "worldmodel"))
        self.policy.load_weights(str(loaddir / "policy"))
        self.value.load_weights(str(loaddir / "critic"))
        self.target_value.load_weights(str(loaddir / "critic"))

    def set_weights(self, weights):

        wm_weights, policy_weights, value_weights = weights

        self.world_model.set_weights(wm_weights)
        self.policy.set_weights(policy_weights)
        self.value.set_weights(value_weights)
        self.target_value.set_weights(value_weights)

    def get_weights(self):

        weights = (
            self.world_model.get_weights(),
            self.policy.get_weights(),
            self.value.get_weights(),
            )

        return weights

    def rollout(self, weights=None):

        if weights:
            self.set_weights(weights)

        env = gym.make(self.env_id)

        obs = self.preprocess_func(env.reset())

        episode_steps, episode_rewards = 0, 0

        prev_z, prev_h = self.world_model.get_initial_state(batch_size=1)

        prev_a = tf.convert_to_tensor([[0]*self.action_space], dtype=tf.float32)

        done = False

        lives = int(env.ale.lives())

        while not done:

            h = self.world_model.step_h(prev_z, prev_h, prev_a)

            feat, z = self.world_model.get_feature(obs, h)

            action = self.policy.sample_action(feat, self.epsilon)

            action_onehot = tf.one_hot([action], self.action_space)

            next_frame, reward, done, info = env.step(action)

            next_obs = self.preprocess_func(next_frame)

            #: Note: DreamerV2 paper uses tanh clipping
            _reward = reward if reward <= 1.0 else 1.0

            #: Life loss as episode end
            if info["ale.lives"] != lives:
                _done = True
                lives = int(info["ale.lives"])
            else:
                _done = done

            #: (r_t-1, done_t-1, obs_t, action_t, done_t)
            self.buffer.add(
                obs, action_onehot, _reward, next_obs, _done,
                prev_z, prev_h, prev_a
                )

            #: Update states
            obs = next_obs

            prev_z, prev_h, prev_a = z, h, action_onehot

            episode_steps += 1

            episode_rewards += reward

            if episode_steps > 4000:
                _ = self.buffer.get_episode()
                return self.pid, [], 0, 0

        sequences = self.buffer.get_sequences()

        return self.pid, sequences, episode_steps, episode_rewards

    def update_networks(self, minibatchs):

        for minibatch in minibatchs:

            z_posts, hs, info = self.update_worldmodel(minibatch)

            trajectory_in_dream = self.rollout_in_dream(z_posts, hs)

            info_ac = self.update_actor_critic(trajectory_in_dream)

        info.update(info_ac)

        return self.get_weights(), info

    def update_worldmodel(self, minibatch):
        """
        Inputs:
            minibatch = {
                "obs":     (L, B, 64, 64, 1)
                "action":  (L, B, action_space)
                "reward":  (L, B)
                "done":    (L, B)
                "prev_z":  (1, B, latent_dim * n_atoms)
                "prev_h":  (1, B, 600)
                "prev_a":  (1, B, action_space)
            }

        Note:
            1. re-compute post and prior z by unrolling sequences
               from initial states, obs, prev_z, prev_h and prev_action
            2. Conmpute KL loss (post_z, prior_z)
            3. Reconstrunction loss, reward, discount loss
        """

        (observations, actions, rewards, next_observations, dones,
         prev_z, prev_h, prev_a) = minibatch.values()

        discounts = (1. - dones) * self.config.gamma_discount

        prev_z, prev_h, prev_a = prev_z[0], prev_h[0], prev_a[0]

        last_obs = next_observations[-1][None, ...]

        observations = tf.concat([observations, last_obs], axis=0)

        #: dummy action to avoid IndexError at last iteration
        last_action = tf.zeros((1,)+actions.shape[1:])

        actions = tf.concat([actions, last_action], axis=0)

        L = self.config.sequence_length

        with tf.GradientTape() as tape:

            hs, z_prior_probs, z_posts, z_post_probs = [], [], [], []

            img_outs, r_means, disc_logits = [], [], []

            for t in tf.range(L+1):

                _outputs = self.world_model(observations[t], prev_z, prev_h, prev_a)

                (h, z_prior, z_prior_prob, z_post, z_post_prob,
                 feat, img_out, reward_mu, disc_logit) = _outputs

                hs.append(h)

                z_prior_probs.append(z_prior_prob)

                z_posts.append(z_post)

                z_post_probs.append(z_post_prob)

                img_outs.append(img_out)

                r_means.append(reward_mu)

                disc_logits.append(disc_logit)

                prev_z, prev_h, prev_a = z_post, h, actions[t]

            #: Reshape outputs
            #: [(B, ...), (B, ...), ...] -> (L+1, B, ...) -> (L, B, ...)
            hs = tf.stack(hs, axis=0)[:-1]

            z_prior_probs = tf.stack(z_prior_probs, axis=0)[:-1]

            z_posts = tf.stack(z_posts, axis=0)[:-1]

            z_post_probs = tf.stack(z_post_probs, axis=0)[:-1]

            img_outs = tf.stack(img_outs, axis=0)[:-1]

            r_means = tf.stack(r_means, axis=0)[1:]

            disc_logits = tf.stack(disc_logits, axis=0)[1:]

            #: Compute loss terms
            kl_loss = self._compute_kl_loss(z_prior_probs, z_post_probs)

            img_log_loss = self._compute_img_log_loss(observations[:-1], img_outs)

            reward_log_loss = self._compute_log_loss(rewards, r_means, mode="reward")

            discount_log_loss = self._compute_log_loss(discounts, disc_logits, mode="discount")

            loss = - img_log_loss - reward_log_loss - discount_log_loss + self.config.kl_scale * kl_loss

            loss *= 1. / L

        grads = tape.gradient(loss, self.world_model.trainable_variables)
        grads, norm = tf.clip_by_global_norm(grads, 100.)
        self.wm_optimizer.apply_gradients(
            zip(grads, self.world_model.trainable_variables)
            )

        info = {"wm_loss": L * loss,
                "img_log_loss": -img_log_loss,
                "reward_log_loss": -reward_log_loss,
                "discount_log_loss": -discount_log_loss,
                "kl_loss": kl_loss}

        return z_posts, hs, info

    @tf.function
    def _compute_kl_loss(self, post_probs, prior_probs):
        """ Compute KL divergence between two OnehotCategorical Distributions

        Notes:
                KL[ Q(z_post) || P(z_prior) ]
                    Q(z_prior) := Q(z | h, o)
                    P(z_prior) := P(z | h)

        Scratch Impl.:
                qlogq = post_probs * tf.math.log(post_probs)
                qlogp = post_probs * tf.math.log(prior_probs)
                kl_div = tf.reduce_sum(qlogq - qlogp, [1, 2])

        Inputs:
            prior_probs (L, B, latent_dim, n_atoms)
            post_probs (L, B, latent_dim, n_atoms)
        """

        #: Add small value to prevent inf kl
        post_probs += 1e-5
        prior_probs += 1e-5

        #: KL Balancing: See 2.2 BEHAVIOR LEARNING Algorithm 2
        kl_div1 = tfd.kl_divergence(
            tfd.Independent(
                tfd.OneHotCategorical(probs=tf.stop_gradient(post_probs)),
                reinterpreted_batch_ndims=1),
            tfd.Independent(
                tfd.OneHotCategorical(probs=prior_probs),
                reinterpreted_batch_ndims=1)
                )

        kl_div2 = tfd.kl_divergence(
            tfd.Independent(
                tfd.OneHotCategorical(probs=post_probs),
                reinterpreted_batch_ndims=1),
            tfd.Independent(
                tfd.OneHotCategorical(probs=tf.stop_gradient(prior_probs)),
                reinterpreted_batch_ndims=1)
                )

        alpha = self.config.kl_alpha

        kl_loss = alpha * kl_div1 + (1. - alpha) * kl_div2

        #: Batch mean
        kl_loss = tf.reduce_mean(kl_loss)

        return kl_loss

    @tf.function
    def _compute_img_log_loss(self, img_in, img_out):
        """
        Inputs:
            img_in: (L, B, 64, 64, 1)
            img_out: (L, B, 64, 64, 1)
        """
        L, B, H, W, C = img_in.shape

        img_in = tf.reshape(img_in, (L * B, H * W * C))

        img_out = tf.reshape(img_out, (L * B, H * W * C))

        dist = tfd.Independent(tfd.Normal(loc=img_out, scale=1.))
        #dist = tfd.Independent(tfd.Bernoulli(logits=img_out))

        log_prob = dist.log_prob(img_in)

        loss = tf.reduce_mean(log_prob)

        return loss

    @tf.function
    def _compute_log_loss(self, y_true, y_pred, mode):
        """
        Inputs:
            y_true: (L, B, 1)
            y_pred: (L, B, 1)
            mode: "reward" or "discount"
        """
        if mode == "discount":
            dist = tfd.Independent(
                tfd.Bernoulli(logits=y_pred), reinterpreted_batch_ndims=1
                )
        elif mode == "reward":
            dist = tfd.Independent(
                tfd.Normal(loc=y_pred, scale=1.), reinterpreted_batch_ndims=1
                )

        log_prob = dist.log_prob(y_true)

        loss = tf.reduce_mean(log_prob)

        return loss

    def rollout_in_dream(self, z_init, h_init, video=False):
        """
        Inputs:
            h_init: (L, B, 1)
            z_init: (L, B, latent_dim * n_atoms)
            done_init: (L, B, 1)
        """
        L, B = h_init.shape[:2]

        horizon = self.config.imagination_horizon

        z, h = tf.reshape(z_init, [L*B, -1]), tf.reshape(h_init, [L*B, -1])
        feats = tf.concat([z, h], axis=-1)

        #: s_t, a_t, s_t+1
        trajectory = {"state": [], "action": [], 'next_state': []}

        for t in range(horizon):

            actions = tf.cast(self.policy.sample(feats), dtype=tf.float32)

            trajectory["state"].append(feats)
            trajectory["action"].append(actions)

            h = self.world_model.step_h(z, h, actions)
            z, _ = self.world_model.rssm.sample_z_prior(h)
            z = tf.reshape(z, [L*B, -1])

            feats = tf.concat([z, h], axis=-1)
            trajectory["next_state"].append(feats)

        trajectory = {k: tf.stack(v, axis=0) for k, v in trajectory.items()}

        #: reward_head(s_t+1) -> r_t
        #: Distribution.mode()は確立最大値を返すのでNormalの場合は
        #: trjactory["reward"] == rewards
        rewards = self.world_model.reward_head(trajectory['next_state'])
        trajectory["reward"] = rewards

        disc_logits = self.world_model.discount_head(trajectory['next_state'])
        trajectory["discount"] = tfd.Independent(
                tfd.Bernoulli(logits=disc_logits), reinterpreted_batch_ndims=1
                ).mean()

        return trajectory

    def update_actor_critic(self, trajectory, batch_size=512, strategy="PPO"):
        """ Actor-Critic update using PPO & Generalized Advantage Estimator
        """

        #: adv: (L*B, 1)
        targets, weights = self.compute_target(
            trajectory['state'], trajectory['reward'],
            trajectory['next_state'], trajectory['discount']
            )
        #: (H, L*B, ...)
        states = trajectory['state']
        selected_actions = trajectory['action']

        N = weights.shape[0] * weights.shape[1]
        states = tf.reshape(states, [N, -1])
        selected_actions = tf.reshape(selected_actions, [N, -1])
        targets = tf.reshape(targets, [N, -1])
        weights = tf.reshape(weights, [N, -1])
        _, old_action_probs = self.policy(states)
        old_logprobs = tf.math.log(old_action_probs + 1e-5)

        for _ in range(10):

            indices = np.random.choice(N, batch_size)

            _states = tf.gather(states, indices)
            _targets = tf.gather(targets, indices)
            _selected_actions = tf.gather(selected_actions, indices)
            _old_logprobs = tf.gather(old_logprobs, indices)
            _weights = tf.gather(weights, indices)

            #: Update value network
            with tf.GradientTape() as tape1:
                v_pred = self.value(_states)
                advantages = _targets - v_pred
                value_loss = 0.5 * tf.square(advantages)
                discount_value_loss = tf.reduce_mean(value_loss * _weights)

            grads = tape1.gradient(discount_value_loss, self.value.trainable_variables)
            self.value_optimizer.apply_gradients(
                        zip(grads, self.value.trainable_variables))

            #: Update policy network
            if strategy == "VanillaPG":

                with tf.GradientTape() as tape2:
                    _, action_probs = self.policy(_states)
                    action_probs += 1e-5

                    selected_action_logprobs = tf.reduce_sum(
                        _selected_actions * tf.math.log(action_probs),
                        axis=1, keepdims=True
                        )

                    objective = selected_action_logprobs * advantages

                    dist = tfd.Independent(
                        tfd.OneHotCategorical(probs=action_probs),
                        reinterpreted_batch_ndims=0)
                    ent = dist.entropy()

                    policy_loss = objective + self.config.ent_scale * ent[..., None]
                    policy_loss *= -1
                    discounted_policy_loss = tf.reduce_mean(policy_loss * _weights)

            elif strategy == "PPO":

                with tf.GradientTape() as tape2:
                    _, action_probs = self.policy(_states)
                    action_probs += 1e-5
                    new_logprobs = tf.math.log(action_probs)

                    ratio = tf.reduce_sum(
                        _selected_actions * tf.exp(new_logprobs - _old_logprobs),
                        axis=1, keepdims=True)
                    ratio_clipped = tf.clip_by_value(ratio, 0.9, 1.1)

                    obj_unclipped = ratio * advantages
                    obj_clipped = ratio_clipped * advantages

                    objective = tf.minimum(obj_unclipped, obj_clipped)

                    dist = tfd.Independent(
                        tfd.OneHotCategorical(probs=action_probs),
                        reinterpreted_batch_ndims=0)
                    ent = dist.entropy()

                    policy_loss = objective + self.config.ent_scale * ent[..., None]
                    policy_loss *= -1
                    discounted_policy_loss = tf.reduce_mean(policy_loss * _weights)

            grads = tape2.gradient(discounted_policy_loss, self.policy.trainable_variables)
            self.policy_optimizer.apply_gradients(
                        zip(grads, self.policy.trainable_variables))

        info = {
            "policy_loss": tf.reduce_mean(policy_loss),
            "objective": tf.reduce_mean(objective),
            "actor_entropy": tf.reduce_mean(ent),
            "value_loss": tf.reduce_mean(value_loss),
            "target_0": tf.reduce_mean(_targets),
            }

        return info

    def compute_target(self, states, rewards, next_states, discounts, strategy="mixed-multistep"):

        T, B, F = states.shape

        v_next = self.target_value(next_states)

        _weights = tf.concat(
            [tf.ones_like(discounts[:1]), discounts[:-1]], axis=0)
        weights = tf.math.cumprod(_weights, axis=0)

        if strategy == "gae":
            """ HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION
                https://arxiv.org/pdf/1506.02438.pdf
            """
            raise NotImplementedError()
            #lambda_ = self.config.lambda_gae
            #deltas = rewards + discounts * v_next - v
            #_weights = tf.concat(
            #    [tf.ones_like(discounts[:1]), discounts[:-1] * lambda_],
            #    axis=0)
            #weights = tf.math.cumprod(_weights, axis=0)
            #advantage = tf.reduce_sum(weights * deltas, axis=0)
            #v_target = advantage + v[0]

        elif strategy == "mixed-multistep":

            targets = np.zeros_like(v_next)  #: (H, L*B, 1)
            last_value = v_next[-1]

            for i in reversed(range(targets.shape[0])):
                last_value = rewards[i] + discounts[i] * last_value
                targets[i] = last_value

        else:
            raise NotImplementedError()

        return targets, weights

    def testplay(self, test_id, video_dir: Path = None, weights=None):

        if weights:
            self.set_weights(weights)

        images = []

        env = gym.make(self.env_id)

        obs = self.preprocess_func(env.reset())

        episode_steps, episode_rewards = 0, 0

        r_pred_total = 0.

        prev_z, prev_h = self.world_model.get_initial_state(batch_size=1)

        prev_a = tf.convert_to_tensor([[0]*self.action_space], dtype=tf.float32)

        done = False

        while not done:

            (h, z_prior, z_prior_probs, z_post,
             z_post_probs, feat, img_out,
             r_pred, discount_logit) = self.world_model(obs, prev_z, prev_h, prev_a)

            action = self.policy.sample_action(feat, 0)

            action_onehot = tf.one_hot([action], self.action_space)

            next_frame, reward, done, info = env.step(action)

            next_obs = self.preprocess_func(next_frame)

            #img_out = tfd.Independent(tfd.Bernoulli(logits=img_out), 3).mean()

            disc = tfd.Bernoulli(logits=discount_logit).mean()

            r_pred_total += float(r_pred)

            img = util.vizualize_vae(
                obs[0, :, :, 0], img_out.numpy()[0, :, :, 0],
                float(r_pred), float(disc), r_pred_total)

            images.append(img)

            #: Update states
            obs = next_obs

            prev_z, prev_h, prev_a = z_post, h, action_onehot

            episode_steps += 1
            episode_rewards += reward

            #: avoiding agent freeze
            if episode_steps > 300 and episode_rewards < 2:
                break
            elif episode_steps > 1000 and episode_rewards < 10:
                break
            elif episode_steps > 4000:
                break

        if video_dir is not None:
            images[0].save(
                f'{video_dir}/testplay_{test_id}.gif',
                save_all=True, append_images=images[1:],
                optimize=False, duration=120, loop=0)

        return episode_steps, episode_rewards

    def testplay_in_dream(self, test_id, outdir: Path, H, weights=None):

        if weights:
            self.set_weights(weights)

        img_outs = []

        prev_z, prev_h = self.world_model.get_initial_state(batch_size=1)

        prev_a = tf.convert_to_tensor([[0]*self.action_space], dtype=tf.float32)

        actions, rewards, discounts = [], [], []

        env = gym.make(self.env_id)

        obs = self.preprocess_func(env.reset())

        N = random.randint(2, 10)

        for i in range(N+H+1):

            if i < N:

                (h, z_prior, z_prior_probs, z_post,
                 z_post_probs, feat, img_out,
                 r_pred, disc_logit) = self.world_model(obs, prev_z, prev_h, prev_a)

                discount_pred = tfd.Bernoulli(logits=disc_logit).mean()

                img_out = obs[0, :, :, 0]

                action = 1 if i == 0 else self.policy.sample_action(feat, 0)

                next_frame, reward, done, info = env.step(action)

                obs = self.preprocess_func(next_frame)

                z = z_post

            else:
                h = self.world_model.step_h(prev_z, prev_h, prev_a)

                z, _ = self.world_model.rssm.sample_z_prior(h)

                z = tf.reshape(z, [1, -1])

                feat = tf.concat([z, h], axis=-1)

                img_out = self.world_model.decoder(feat)

                #img_out = tfd.Independent(tfd.Bernoulli(logits=img_out), 3).mean()

                img_out = img_out.numpy()[0, :, :, 0]

                r_pred = self.world_model.reward_head(feat)

                disc_logit = self.world_model.discount_head(feat)

                discount_pred = tfd.Bernoulli(logits=disc_logit).mean()

                action = self.policy.sample_action(feat, 0)

                actions.append(int(action))

                rewards.append(float(r_pred))

                discounts.append(float(discount_pred))

                img_outs.append(img_out)

            action_onehot = tf.one_hot([action], self.action_space)

            prev_z, prev_h, prev_a = z, h, action_onehot

        img_outs, actions, rewards, discounts = img_outs[:-1], actions[:-1], rewards[1:], discounts[1:]
        images = util.visualize_dream(img_outs, actions, rewards, discounts)
        images[0].save(
            f'{outdir}/test_in_dream_{test_id}.gif',
            save_all=True, append_images=images[1:],
            optimize=False, duration=1000, loop=0)


@ray.remote(num_cpus=1, num_gpus=1)
class Learner(DreamerV2Agent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@ray.remote(num_cpus=1, num_gpus=0)
class Actor(DreamerV2Agent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@ray.remote(num_cpus=1, num_gpus=0)
class Tester(DreamerV2Agent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def main(resume=None, num_actors=5, init_episodes=50,
         env_id="BreakoutDeterministic-v4", debug=True):

    """ Setup training log dirs
    """
    logdir, videodir = Path("./log"), Path("./video")

    if resume:
        #: Create backup
        if logdir.exists():
            try:
                shutil.copytree(
                    logdir, "./log_"+str(datetime.date.today())
                    )
            except FileExistsError:
                print("Skip crating bakup: log")

        if Path("./checkpoints").exists():
            try:
                shutil.copytree(
                    "./checkpoints", "./checkpoints_"+str(datetime.date.today())
                    )
            except FileExistsError:
                print("Skip crating bakup: checkpoints")
    else:
        if logdir.exists():
            shutil.rmtree(logdir)
        logdir.mkdir()

        if videodir.exists():
            shutil.rmtree(videodir)
        videodir.mkdir()

    summary_writer = tf.summary.create_file_writer(str(logdir))

    """ Setup Ape-X architecture
    """

    ray.init()

    config = Config()

    learner = Learner.remote(env_id=env_id, config=config)

    epsilons = [0.5 ** (1 + 7. * i / (num_actors - 1)) for i in range(num_actors)]
    print("Epsilons", epsilons)

    actors = [Actor.remote(
        pid=i, env_id=env_id, config=config, epsilon=max(0.05, epsilons[i]),
        ) for i in range(num_actors)]

    tester = Tester.remote(env_id=env_id, config=config)

    replay_buffer = SequenceReplayBuffer(
        buffer_size=config.buffer_size,
        seq_len=config.sequence_length,
        batch_size=config.batch_size,
        action_space=gym.make(env_id).action_space.n,
        )

    if resume:
        learner.load.remote("checkpoints")

    current_weights = ray.put(ray.get(learner.get_weights.remote()))

    """ Initial collcetion of experience
    """
    init_episodes = init_episodes if not debug else 5
    wip_actors = [actor.rollout.remote(weights=current_weights) for actor in actors]
    for _ in range(init_episodes):
        finished_actor, wip_actors = ray.wait(wip_actors, num_returns=1)
        pid, sequences, steps, score = ray.get(finished_actor[0])
        print(f"PID-{pid}: {score} : {steps}steps")
        replay_buffer.add_sequences(sequences)
        wip_actors.extend(
            [actors[pid].rollout.remote(current_weights)])

    """ Counters
    """
    global_steps = 0 if resume is None else int(resume["global_steps"])

    learner_update_count, test_count = 0, 0

    """ Start training
    """

    minibatchs = [replay_buffer.get_minibatch()
                  for _ in range(config.num_minibatchs)]

    wip_learner = learner.update_networks.remote(minibatchs)

    wip_tester = tester.testplay.remote(
        test_id=global_steps, video_dir=videodir, weights=current_weights)

    minibatchs = [replay_buffer.get_minibatch() for _ in range(10)]

    while True:

        finished_actor, wip_actors = ray.wait(wip_actors, num_returns=1, timeout=0)

        if finished_actor:
            pid, sequences, steps, score = ray.get(finished_actor[0])
            replay_buffer.add_sequences(sequences)
            global_steps += steps
            print(f"PID-{pid}: {score} : {steps}steps")
            if not debug:
                wip_actors.extend([actors[pid].rollout.remote(current_weights)])

        finished_learner, _ = ray.wait([wip_learner], timeout=0)

        if finished_learner:
            learner_update_count += 1

            print(f"== Learner Update: {learner_update_count} ==")

            weights, info = ray.get(finished_learner[0])
            current_weights = ray.put(weights)

            with summary_writer.as_default():
                for key, value in info.items():
                    tf.summary.scalar(key, value, step=global_steps)

            if learner_update_count % 50 == 0:
                print("== Save weights ==")
                ray.get(learner.save.remote("checkpoints"))

            if learner_update_count % 40 == 0:
                print("== Update Target-value ==")
                ray.get(learner.set_weights.remote(current_weights))

            wip_learner = learner.update_networks.remote(minibatchs)

            with util.Timer("Create Minibatch"):
                minibatchs = [
                    replay_buffer.get_minibatch() for _ in range(config.num_minibatchs)
                    ]

        finished_tester, _ = ray.wait([wip_tester], timeout=0)

        if finished_tester:

            test_count += 1

            test_steps, test_score = ray.get(finished_tester[0])

            if not debug:
                print(f"Test {test_count}: {test_score} : {test_steps}steps")

            _videodir = videodir if test_count % 50 == 0 else None
            wip_tester = tester.testplay.remote(
                test_id=global_steps, video_dir=_videodir, weights=current_weights)

            with summary_writer.as_default():
                tf.summary.scalar("test_steps", test_steps, step=global_steps)
                tf.summary.scalar("test_score", test_score, step=global_steps)


if __name__ == "__main__":
    resume = None
    #resume = {"global_steps": 129999999}
    main(resume, debug=False)
