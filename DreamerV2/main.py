from dataclasses import dataclass
import math
from pathlib import Path
import shutil

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import gym

from buffer import SequenceReplayBuffer
from networks import PolicyNetwork, ValueNetwork, WorldModel
import util


@dataclass
class Config:

    batch_size: int = 32           # Batch size, B
    sequence_length: int = 10      # Sequence Lenght, L
    buffer_size: int = int(1e6)    # Replay buffer size (FIFO)
    gamma: float = 0.997
    anneal_stpes: int = 500000
    update_period: int = 12
    target_update_period: int = 1200

    kl_scale: float = 0.1     # KL loss scale, β
    kl_alpha: float = 0.8          # KL balancing
    ent_scale: float = 1e-3
    latent_dim: int = 32           # discrete latent dimensions
    n_atoms: int = 32              # discrete latent classes
    lr_world: float = 2e-4         # learning rate of world model

    imagination_horizon: int = 5   # Imagination horizon, H
    gamma_discount: float = 0.995   # discount factor γ
    lambda_gae: float = 0.95       # λ for Generalized advantage estimator
    entropy_scale: float = 1e-3    # entropy loss scale
    lr_actor: float = 4e-5
    lr_critic: float = 1e-4

    adam_epsilon: float = 1e-5
    adam_decay: float = 1e-6
    grad_clip: float = 100.


class DreamerV2Agent:

    def __init__(self, env_id: str, config: Config,
                 summary_writer: tf.summary.SummaryWriter = None):

        self.env_id = env_id

        self.config = config

        self.summary_writer = summary_writer

        self.action_space = gym.make(self.env_id).action_space.n

        self.preprocess_func = util.get_preprocess_func(env_name=self.env_id)

        self.buffer = SequenceReplayBuffer(
            buffer_size=config.buffer_size,
            seq_len=config.sequence_length,
            batch_size=config.batch_size,
            action_space=self.action_space,
            )

        self.world_model = WorldModel(config)
        self.wm_optimizer = tf.keras.optimizers.Adam(lr=self.config.lr_world)

        self.actor = PolicyNetwork(action_space=self.action_space)
        self.actor_optimizer = tf.keras.optimizers.Adam(lr=self.config.lr_actor)

        self.critic = ValueNetwork(action_space=self.action_space)
        self.target_critic = ValueNetwork(action_space=self.action_space)
        self.critic_optimizer = tf.keras.optimizers.Adam(lr=self.config.lr_critic)

        self.global_steps = 0

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
        self.actor(feat)
        self.critic(feat)
        self.target_critic(feat)
        self.target_critic.set_weights(self.critic.get_weights())

    def get_weights(self):
        return (self.world_model.get_weights(),
                self.policy.get_weights(), self.critic.get_weights())

    def save(self, savedir=None):
        savedir = Path(savedir) if savedir is not None else Path("./checkpoints")
        self.world_model.save_weights(str(savedir / "worldmodel"))
        self.actor.save_weights(str(savedir / "actor"))
        self.critic.save_weights(str(savedir / "critic"))

    def load(self, loaddir=None):
        loaddir = Path(loaddir) if loaddir is not None else Path("checkpoints")
        self.world_model.load_weights(str(loaddir / "worldmodel"))
        self.actor.load_weights(str(loaddir / "actor"))
        self.critic.load_weights(str(loaddir / "critic"))
        self.target_critic.load_weights(str(loaddir / "critic"))

    @property
    def epsilon(self):
        eps = max(0.0, 0.6 * (self.config.anneal_stpes - self.global_steps) / self.config.anneal_stpes)
        return eps

    def rollout(self, training: bool):

        env = gym.make(self.env_id)

        obs = self.preprocess_func(env.reset())

        episode_steps, episode_rewards = 0, 0

        prev_z, prev_h = self.world_model.get_initial_state(batch_size=1)

        prev_a = tf.one_hot([0], self.action_space)

        done = False

        lives = int(env.ale.lives())

        while not done:

            h = self.world_model.step_h(prev_z, prev_h, prev_a)

            feat, z = self.world_model.get_feature(obs, h)

            action = self.actor.sample_action(feat, self.epsilon)

            action_onehot = tf.one_hot([action], self.action_space)

            next_frame, reward, done, info = env.step(action)

            next_obs = self.preprocess_func(next_frame)

            #: Reward clipping by tanh
            _reward = math.tanh(reward)

            #: Life loss as episode end
            if info["ale.lives"] != lives:
                lives = int(info["ale.lives"])
                _done = True
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

            #: Update world model and actor-critic
            if training and self.global_steps % self.config.update_period == 0:
                self.update_networks()

            #: Target update
            if training and self.global_steps % self.config.target_update_period == 0:
                print("== Target Update ==")
                self.target_critic.set_weights(self.critic.get_weights())

            #: Stats
            self.global_steps += 1

            episode_steps += 1

            episode_rewards += reward

        return episode_steps, episode_rewards, self.global_steps

    def update_networks(self):

        minibatch = self.buffer.get_minibatch()

        z_posts, hs = self.update_worldmodel(minibatch)

        trajectory_in_dream = self.rollout_in_dream(z_posts, hs)

        self.update_actor_critic(trajectory_in_dream)

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
                 feat, img_out, reward_pred, disc_logit) = _outputs

                hs.append(h)

                z_prior_probs.append(z_prior_prob)

                z_posts.append(z_post)

                z_post_probs.append(z_post_prob)

                img_outs.append(img_out)

                r_means.append(reward_pred)

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

            reward_log_loss = self._compute_log_loss(rewards, r_means, head="reward")

            discount_log_loss = self._compute_log_loss(discounts, disc_logits, head="discount")

            loss = - img_log_loss - reward_log_loss - discount_log_loss + self.config.kl_scale * kl_loss

            loss *= 1. / L

        grads = tape.gradient(loss, self.world_model.trainable_variables)
        grads, norm = tf.clip_by_global_norm(grads, 100.)
        self.wm_optimizer.apply_gradients(zip(grads, self.world_model.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar("wm_loss", L * loss, step=self.global_steps)
            tf.summary.scalar("img_log_loss", -img_log_loss, step=self.global_steps)
            tf.summary.scalar("reward_log_loss", -reward_log_loss, step=self.global_steps)
            tf.summary.scalar("discount_log_loss", -discount_log_loss, step=self.global_steps)
            tf.summary.scalar("kl_loss", kl_loss, step=self.global_steps)

        return z_posts, hs

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

        dist = tfd.Independent(tfd.Normal(loc=img_out, scale=2.))

        log_prob = dist.log_prob(img_in)

        loss = tf.reduce_mean(log_prob)

        return loss

    @tf.function
    def _compute_log_loss(self, y_true, y_pred, head):
        """
        Inputs:
            y_true: (L, B, 1)
            y_pred: (L, B, 1)
            head: "reward" or "discount"
        """

        if head == "reward":
            dist = tfd.Independent(
                tfd.Normal(loc=y_pred, scale=1.), reinterpreted_batch_ndims=1
                )
        elif head == "discount":
            dist = tfd.Independent(
                tfd.Bernoulli(logits=y_pred), reinterpreted_batch_ndims=1
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

            actions = tf.cast(self.actor.sample(feats), dtype=tf.float32)

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
        trajectory["reward"] = tfd.Independent(
                tfd.Normal(loc=rewards, scale=1.),
                reinterpreted_batch_ndims=1
                ).mode()

        disc_logits = self.world_model.discount_head(trajectory['next_state'])
        trajectory["discount"] = tfd.Independent(
                tfd.Bernoulli(logits=disc_logits), reinterpreted_batch_ndims=1
                ).mean()

        return trajectory

    def update_actor_critic(self, trajectory):
        """ Actor-Critic update using Generalized Advantage Estimator
        """

        #: adv: (L*B, 1)
        adv, v_target = self.compute_GAE(
            trajectory['state'], trajectory['reward'],
            trajectory['next_state'], trajectory['discount']
            )

        states = trajectory['state'][0]
        actions = trajectory['action'][0]

        with tf.GradientTape() as tape:

            _, action_probs = self.actor(states)

            objective = actions * tf.math.log(action_probs + 1e-5) * adv

            objective = tf.reduce_sum(objective, axis=-1)

            dist = tfd.Independent(
                tfd.OneHotCategorical(probs=action_probs),
                reinterpreted_batch_ndims=0)
            ent = dist.entropy()

            actor_loss = -1 * objective + self.config.ent_scale * ent
            actor_loss = tf.reduce_mean(actor_loss)

        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
                    zip(grads, self.actor.trainable_variables))

        with tf.GradientTape() as tape:
            v_pred = self.critic(states)
            value_loss = 0.5 * tf.square(v_target - v_pred)
            value_loss = tf.reduce_mean(value_loss)

        grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
                    zip(grads, self.critic.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar("actor_loss", actor_loss, step=self.global_steps)
            tf.summary.scalar("value_loss", value_loss, step=self.global_steps)

    def compute_GAE(self, states, rewards, next_states, discounts):
        """ HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION
            https://arxiv.org/pdf/1506.02438.pdf
        """
        T, B, F = states.shape
        lambda_ = self.config.lambda_gae

        v = self.target_critic(states)
        v_next = self.target_critic(next_states)
        deltas = rewards + discounts * v_next - v

        _weights = tf.concat(
            [tf.ones_like(discounts[:1]), discounts[:-1] * lambda_],
            axis=0)

        weights = tf.math.cumprod(_weights, axis=0)

        adv = tf.reduce_sum(weights * deltas, axis=0)

        v_target = adv + v[0]

        return adv, v_target

    def testplay(self, test_id, outdir: Path):

        images = []

        env = gym.make(self.env_id)

        obs = self.preprocess_func(env.reset())

        episode_steps, episode_rewards = 0, 0

        prev_z, prev_h = self.world_model.get_initial_state(batch_size=1)

        prev_a = tf.one_hot([0], self.action_space)

        done = False

        while not done:

            (h, z_prior, z_prior_probs, z_post,
             z_post_probs, feat, img_out,
             r_mean, discount_pred) = self.world_model(obs, prev_z, prev_h, prev_a)

            action = self.actor.sample_action(feat, 0)

            action_onehot = tf.one_hot([action], self.action_space)

            next_frame, reward, done, info = env.step(action)

            next_obs = self.preprocess_func(next_frame)

            img = util.vizualize_vae(obs[0, :, :, 0], img_out.numpy()[0, :, :, 0])
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

        images[0].save(
            f'{outdir}/testplay_{test_id}.gif',
            save_all=True, append_images=images[1:],
            optimize=False, duration=60, loop=0)

        return episode_steps, episode_rewards

    def testplay_in_dream(self, test_id, outdir: Path, initial_state=None, H=10):

        img_outs = []

        prev_z, prev_h = self.world_model.get_initial_state(batch_size=1)

        prev_a = tf.one_hot([0], self.action_space)

        actions, rewards, discounts = [], [], []

        for i in range(H+1):

            if i == 0:
                env = gym.make(self.env_id)

                obs = self.preprocess_func(env.reset())

                (h, z_prior, z_prior_probs, z_post,
                 z_post_probs, feat, img_out,
                 r_mean, disc_logit) = self.world_model(obs, prev_z, prev_h, prev_a)

                discount_pred = tfd.Bernoulli(logits=disc_logit).mean()

                img_out = obs[0, :, :, 0]

                z = z_post

                action = 1  #: 0: NOOP, 1:FIRE, 2: LEFT 3: RIGHT

            else:
                h = self.world_model.step_h(prev_z, prev_h, prev_a)

                z, _ = self.world_model.rssm.sample_z_prior(h)

                z = tf.reshape(z, [1, -1])

                feat = tf.concat([z, h], axis=-1)

                img_out = self.world_model.decoder(feat)

                img_out = img_out.numpy()[0, :, :, 0]

                r_mean = self.world_model.reward_head(feat)

                disc_logit = self.world_model.discount_head(feat)

                discount_pred = tfd.Bernoulli(logits=disc_logit).mean()

                action = self.actor.sample_action(feat, 0)

            action_onehot = tf.one_hot([action], self.action_space)

            actions.append(int(action))

            rewards.append(float(r_mean))

            discounts.append(float(discount_pred))

            img_outs.append(img_out)

            prev_z, prev_h, prev_a = z, h, action_onehot

        img_outs, actions, rewards, discounts = img_outs[:-1], actions[:-1], rewards[1:], discounts[1:]
        images = util.visualize_dream(img_outs, actions, rewards, discounts)
        images[0].save(
            f'{outdir}/test_in_dream_{test_id}.gif',
            save_all=True, append_images=images[1:],
            optimize=False, duration=1000, loop=0)


def main(resume=None):
    """ resume: Dict(n: int, global_steps: int)
    """

    logdir = Path("./log")
    summary_writer = tf.summary.create_file_writer(str(logdir))

    videodir = Path("./video")

    if resume is None:

        if logdir.exists():
            shutil.rmtree(logdir)
        logdir.mkdir()

        if videodir.exists():
            shutil.rmtree(videodir)
        videodir.mkdir()

    env_id = "BreakoutDeterministic-v4"

    config = Config()

    agent = DreamerV2Agent(
        env_id=env_id, config=config, summary_writer=summary_writer
        )

    init_episodes = 100

    test_interval = 100

    n = 0

    if resume:
        n = int(resume["n"])
        init_episodes += n
        agent.global_steps = int(resume["global_steps"])
        agent.load("checkpoints")
        print("== Load weights ==")

    while n < 10000:

        training = n > init_episodes

        steps, score, global_steps = agent.rollout(training)

        with summary_writer.as_default():
            tf.summary.scalar("train_score", score, step=global_steps)

        print(f"Episode {n}: {steps}steps {score}")
        print()

        if n % test_interval == 0:
            agent.testplay_in_dream(n, videodir)
            steps, score = agent.testplay(n, videodir)

            with summary_writer.as_default():
                tf.summary.scalar("test_score", score, step=global_steps)

            print(f"Test: {steps}steps {score}")
            print()

        if n % 50 == 0:
            agent.save()
            print("== Save weights ==")

        n += 1

if __name__ == "__main__":
    resume = None
    resume = {"n": 280, "global_steps":54000}
    main(resume)
