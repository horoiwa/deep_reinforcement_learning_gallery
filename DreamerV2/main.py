from dataclasses import dataclass
import math

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import gym

from buffer import Experience, SequenceReplayBuffer
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
    discount: float = 0.995        # discount factor γ
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
        self.wm_optimizer = tf.keras.optimizers.Adam(lr=2e-4)

        self.actor = PolicyNetwork(action_space=self.action_space)
        self.actor_optimizer = tf.keras.optimizers.Adam()

        self.critic = ValueNetwork(action_space=self.action_space)
        self.critic_optimizer = tf.keras.optimizers.Adam()

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

            self.buffer.add(obs, action_onehot, _reward, _done, prev_z, prev_h)

            #: Update states
            obs = self.preprocess_func(next_frame)

            prev_z, prev_h, prev_a = z, h, action_onehot

            if training and self.global_steps % self.config.update_interval == 0:
                self.update_networks()

            #: Stats
            self.global_steps += 1

            episode_steps += 1

            episode_rewards += reward

        return episode_steps, episode_rewards

    def update_networks(self):

        minibatch = self.buffer.get_minibatch()

        minibatch = self.update_worldmodel(minibatch)

        self.update_actor_critic(minibatch)

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
        (observations, actions, rewards, dones, prev_z, prev_h, prev_a) = minibatch.values()

        prev_z, prev_h, prev_a = prev_z[0], prev_h[0], prev_a[0]

        L, B = observations.shape[0], observations.shape[1]

        with tf.GradeientTape as tape:

            loss = 0

            for idx in tf.range(L):

                _outputs = self.world_model(observations[idx], prev_z, prev_h, prev_a)

                (h, z_prior, z_prior_probs, z_post, z_post_probs,
                 feat, img_decoded, reward_pred, discount_pred) = _outputs

                kl_loss = self._compute_kl_loss(z_prior_probs, z_post_probs)

                img_log_loss = None

                reward_log_loss = None

                discount_log_loss = None

                prev_z, prev_h, prev_a = z_post, h, actions[idx]

            image_log_loss = self._compute_image_log_loss(features)

            reward_log_loss, discount_log_loss = self.compute_log_loss(features)

            loss = - image_log_loss + kl_loss

        grads = tape.gradient(loss, self.world_model.trainable_variables)
        grads, norm = tf.clip_by_global_norm(grads, 100.)
        self.wm_optimizer.apply_gradients(self.world_model.trainable_variables)

        return minibatch

    @tf.function
    def _unroll(self, observations, actions, z_init, h_init, a_init):

        prev_z, prev_h, prev_a = z_init, h_init, a_init

        for i in range():

        self.world_model

    def _compute_kl(self, left, right, dist_type=tfd.OneHotCategorical):
        pass

    def rollout_in_dream(self):
        return None

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
