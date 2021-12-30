from dataclasses import dataclass

import tensorflow as tf
import gym

from buffer import Experience, SequenceReplayBuffer
from networks import PolicyNetwork, ValueNetwork, WorldModel
import util


@dataclass
class Config:

    num_episodes: int = 10         # Num of total rollouts
    batch_size: int = 48           # Batch size, B
    sequence_length: int = 50      # Sequence Lenght, L
    buffer_size: int = int(1e6)    # Replay buffer size (FIFO)
    gamma: float = 0.997
    anneal_stpes: int = 1000000
    update_interval: int = 4

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

    def rollout(self, training=False):

        env = gym.make(self.env_id)

        obs = self.preprocess_func(env.reset())

        episode_steps, episode_rewards = 0, 0

        prev_z, prev_h = self.world_model.get_initial_state(batch_size=1)

        prev_a = tf.one_hot([0], self.action_space)

        done = False

        while not done:

            h = self.world_model.step_h(prev_z, prev_h, prev_a)

            feat, z = self.world_model.get_feature(obs, h)

            action = self.actor.sample_action(feat, self.epsilon)

            action_onehot = tf.one_hot([action], self.action_space)

            next_frame, reward, done, info = env.step(action)

            #: Send transition to replaybuffer
            is_first = True if episode_steps == 0 else False

            self.buffer.add(obs, action_onehot, reward, is_first, done)

            #: Update states
            obs = self.preprocess_func(next_frame)

            prev_z, prev_h, prev_a = z, h, action_onehot

            if self.global_steps % self.config.update_interval == 0:
                self.update_networks()

            #: Stats
            self.global_steps += 1

            episode_steps += 1

            episode_rewards += reward

        return episode_steps, episode_rewards

    def rollout_imagine(self):
        pass

    def update_networks(self, batch_size, sequence_length):

        minibatch = self.buffer.make_minibatch(batch_size, sequence_length)

        minibatch = self.update_worldmodel(minibatch)

        self.update_actor_critic(minibatch)

    def update_worldmodel(self, minibatch):
        """
            1. Conmpute latent states from raw observations
            2. Update representaion model and transition model
            3. Append latent states to minibatch
        """
        return minibatch

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

    for _ in range(5):
        agent.rollout()

    import sys
    sys.exit()

    n = 0

    while n < config.num_episodes:

        steps, score = agent.rollout()


if __name__ == "__main__":
    main()
