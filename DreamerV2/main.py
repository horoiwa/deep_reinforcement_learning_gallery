from dataclasses import dataclass
import math

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import gym

from buffer import SequenceReplayBuffer
from networks import PolicyNetwork, ValueNetwork, WorldModel
import util


@dataclass
class Config:

    num_episodes: int = 10         # Num of total rollouts
    batch_size: int = 48           # Batch size, B
    sequence_length: int = 15      # Sequence Lenght, L
    buffer_size: int = int(1e6)    # Replay buffer size (FIFO)
    gamma: float = 0.997
    anneal_stpes: int = 1000000
    update_interval: int = 8

    kl_loss_scale: float = 0.1     # KL loss scale, β
    kl_alpha: float = 0.8          # KL balancing
    ent_loss_scale: float = 1e-3
    latent_dim: int = 32           # discrete latent dimensions
    n_atoms: int = 32              # discrete latent classes
    lr_world: float = 2e-4         # learning rate of world model

    imagination_horizon: int = 10  # Imagination horizon, H
    discount_lambda: float = 0.995 # discount factor γ
    lambda_target: float = 0.95    # λ-target parameter
    entropy_scale: float = 1e-3    # entropy loss scale
    lr_actor: float = 4e-5
    lr_critic: float = 1e-4

    adam_epsilon: float = 1e-5
    adam_decay: float = 1e-6
    grad_clip: float = 100.


class DreamerV2Agent:

    def __init__(self, env_id: str, config: Config,
                 summary_writer: int = None):

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
        self.critic_optimizer = tf.keras.optimizers.Adam(lr=self.config.lr_critic)

        self.global_steps = 0

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

        prev_r, prev_done = 0, False

        done = False

        lives = int(env.ale.lives())

        while not done:

            h = self.world_model.step_h(prev_z, prev_h, prev_a)

            feat, z = self.world_model.get_feature(obs, h)

            action = self.actor.sample_action(feat, self.epsilon)

            action_onehot = tf.one_hot([action], self.action_space)

            next_frame, reward, done, info = env.step(action)

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
                prev_r, prev_done,
                obs, action_onehot, _reward, _done,
                prev_z, prev_h, prev_a
                )

            #: Update states
            obs = self.preprocess_func(next_frame)

            prev_z, prev_h, prev_a = z, h, action_onehot

            prev_r, prev_done = _reward, _done

            if training and self.global_steps % self.config.update_interval == 0:
                self.update_networks()

            #: Stats
            self.global_steps += 1

            episode_steps += 1

            episode_rewards += reward

        return episode_steps, episode_rewards

    def update_networks(self):

        minibatch = self.buffer.get_minibatch()

        z_posts, hs = self.update_worldmodel(minibatch)

        trajectory = self.rollout_in_dream(z_posts, hs, minibatch["done"])

        self.update_actor_critic(trajectory)

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
        #: r_t-1, done_t-1, obs_t, action_t
        (prev_rewards, prev_dones, observations, actions, rewards, dones,
         prev_z, prev_h, prev_a) = minibatch.values()

        prev_z, prev_h, prev_a = prev_z[0], prev_h[0], prev_a[0]

        L = observations.shape[0]

        with tf.GradientTape() as tape:

            hs, z_prior_probs, z_posts, z_post_probs = [], [], [], []

            img_outs, r_preds, done_preds = [], [], []

            for t in tf.range(L):

                _outputs = self.world_model(observations[t], prev_z, prev_h, prev_a)

                (h, z_prior, z_prior_prob, z_post, z_post_prob,
                 feat, img_out, reward_pred, done_pred) = _outputs

                hs.append(h)

                z_prior_probs.append(z_prior_prob)

                z_posts.append(z_post)

                z_post_probs.append(z_post_prob)

                img_outs.append(img_out)

                r_preds.append(reward_pred)

                done_preds.append(done_pred)

                prev_z, prev_h, prev_a = z_post, h, actions[t]

            #: Reshape outputs
            #: [(B, ...), (B, ...), ...] -> (L, B, ...)
            hs = tf.stack(hs, axis=0)

            z_prior_probs = tf.stack(z_prior_probs, axis=0)

            z_posts = tf.stack(z_posts, axis=0)

            z_post_probs = tf.stack(z_post_probs, axis=0)

            img_outs = tf.stack(img_outs, axis=0)

            r_preds = tf.stack(r_preds, axis=0)

            done_preds = tf.stack(done_preds, axis=0)

            #: Compute loss terms
            kl_loss = self._compute_kl_loss(z_prior_probs, z_post_probs)

            img_log_loss = self._compute_img_log_loss(observations, img_outs)

            reward_log_loss = self._compute_log_loss(prev_rewards, r_preds, head="reward")

            discount_log_loss = self._compute_log_loss(prev_dones, done_preds, head="discount")

            loss = - img_log_loss - reward_log_loss - discount_log_loss + kl_loss

            loss *= 1. / L

        grads = tape.gradient(loss, self.world_model.trainable_variables)
        grads, norm = tf.clip_by_global_norm(grads, 100.)
        self.wm_optimizer.apply_gradients(zip(grads, self.world_model.trainable_variables))

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
        else:
            raise NotImplementedError(head)

        log_prob = dist.log_prob(y_true)

        loss = tf.reduce_mean(log_prob)

        return loss

    def rollout_in_dream(self, z_init, h_init, done_init, video=False):
        """
        Inputs:
            h_init: (L, B, 1)
            z_init: (L, B, latent_dim * n_atoms)
            feat_init: (L, B, 1 + latent_dim * n_atoms)
            done_init: (L, B, 1)
        """
        L, B = h_init.shape[:2]

        horizon = self.config.imagination_horizon

        z, h = tf.reshape(z_init, [L*B, -1]), tf.reshape(h_init, [L*B, -1])
        feats = tf.concat([z, h], axis=-1)

        done_init = tf.reshape(done_init, [L*B, -1])

        #: s_t, a_t, s_t+1
        trajectory = {"feat": [], "action": [], 'next_feat': []}

        for t in range(horizon):

            actions = tf.cast(self.actor.sample(feats), dtype=tf.float32)

            trajectory["feat"].append(feats)
            trajectory["action"].append(actions)

            h = self.world_model.step_h(z, h, actions)
            z, _ = self.world_model.rssm.sample_z_prior(h)
            z = tf.reshape(z, [L*B, -1])

            feats = tf.concat([z, h], axis=-1)
            trajectory["next_feat"].append(feats)

        trajectory = {k: tf.stack(v, axis=0) for k, v in trajectory.items()}

        #: reward_head(s_t+1) -> r_t
        rewards = self.world_model.reward_head(trajectory['next_feat'])
        trajectory["r"] = tfd.Independent(
                tfd.Normal(loc=rewards, scale=1.),
                reinterpreted_batch_ndims=1
                ).mode()

        dones = self.world_model.discount_head(trajectory['next_feat'])
        dones = tfd.Independent(
                tfd.Bernoulli(logits=dones), reinterpreted_batch_ndims=1
                ).mode()

        #: first stepだけはis_doneを予測値でなく観測値に置き換える
        discount_weights = (1. - dones) * self.config.discount_lambda
        discount_weights[0] = (1. - done_init)
        raise NotImplementedError()

        trajactory["discount_weights"] = None
        return trajectory

    def update_actor_critic(self, minibatch):
        """
            1. Create imagined trajectories from each start states
            2. Compute target values from imagined trajectories (GAE)
            3. Update policy network to maximize target value
            4. Update value network to minimize (imagined_value - pred_value)^2
        """

        with tf.GradientTape() as tape:
            #: compute imagined trajectory
            #: computte target value
            target_value = None
            loss = -1 * target_value

        grads = tape.gradient(loss, self.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(
                    zip(grads, self.policy.trainable_variables))

        with tf.GradientTape() as tape:
            #: compute imagined trajectory
            #: computte target value
            loss = 0.5 * tf.square(target_value - pred_value)

        grads = tape.gradient(loss, self.value.trainable_variables)
        self.value_optimizer.apply_gradients(
                    zip(grads, self.value.trainable_variables))


def main():

    env_id = "BreakoutDeterministic-v4"

    config = Config()

    agent = DreamerV2Agent(env_id=env_id, config=config)

    n = 0

    while n < config.num_episodes:

        training = n > 2

        steps, score = agent.rollout(training)

        print(f"Episode {n}: {steps}steps {score}")
        print()

        n += 1

if __name__ == "__main__":
    main()
