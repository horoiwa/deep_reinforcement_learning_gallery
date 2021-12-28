from dataclasses import dataclass

import tensorflow as tf
import gym

from buffer import Experience, SequenceReplayBuffer
from networks import PolicyNetwork, ValueNetwork
import util


@dataclass
class Config:

    num_episodes: int = 10         # Num of total rollouts
    batch_size: int = 50           # Batch size, B
    sequence_length: int = 50      # Sequence Lenght, L
    buffer_size: int = 2e6         # Replay buffer size (FIFO)

    kl_scale: float = 0.1           # KL loss scale, β
    kl_alpha: float = 0.8          # KL balancing
    latent_dim: int = 32           # discrete latent dimensions
    latent_classes: int = 32       # discrete latent classes
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


class DreamerV2:

    def __init__(self, env_id: str, config: Config,
                 summary_writer: int = None):

        self.env_id = env_id

        self.config = config

        self.summary_writer = summary_writer

        self.action_space = gym.make(self.env_id).action_space.n

        self.preprocess_func = util.get_preprocess_func(env_name=self.env_id)

        self.buffer = SequenceReplayBuffer(
            size=config.buffer_size,
            seq_len=config.sequence_length,
            batch_size=config.batch_size
            )

        self.policy = PolicyNetwork(action_space=self.action_space)

        self.policy_optimizer = tf.keras.optimizers.Adam()

        self.value = ValueNetwork()

        self.value_optimizer = tf.keras.optimizers.Adam()

        self.total_steps = 0

    @property
    def epsilon(self):
        return 0.4

    def rollout(self, training=False):

        env = gym.make(self.env_id)

        obs = self.preprocess_func(env.reset())

        total_rewards = 0

        done = False

        while not done:

            state = self.encoder(obs)

            action, latent_state = self.policy.sample(state, self.epsilon)

            next_frame, reward, done, info = env.step(action)

            exp = Experience(obs, action, reward, done)

            self.buffer.append(exp)

            obs = self.preprocess_func(next_frame)

            self.total_steps += 1

            total_rewards += reward

    def rollout_imagine(self):
        pass

    def update(self, batch_size, sequence_length):

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

    agent = DreamerV2(env_id=env_id, config=config)

    for _ in range(5):
        agent.rollout()

    import sys
    sys.exit()

    n = 0

    while n < config.num_episodes:

        for update_step in range(config.collect_interval):

            agent.update()

        agent.rollout()


if __name__ == "__main__":
    main()
