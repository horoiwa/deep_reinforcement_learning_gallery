from pathlib import Path
import shutil
from dataclasses import dataclass
from collections import namedtuple

import tensorflow as tf
import gym
from gym import wrappers
import numpy as np

from buffer import ReplayBuffer
from networks import PolicyNetwork, CriticNetwork


Transition = namedtuple('Transition', ["state", "action", "reward", "next_state", "done"])


class MPOAgent:

    def __init__(self, env_id: str, logdir: Path):

        self.env_id = env_id

        self.summary_writer = tf.summary.create_file_writer(str(logdir))

        self.action_space = gym.make(self.env_id).action_space.shape[0]

        self.replay_buffer = ReplayBuffer(maxlen=100000)

        self.policy = PolicyNetwork(action_space=self.action_space)
        self.target_policy = PolicyNetwork(action_space=self.action_space)

        self.critic = CriticNetwork()
        self.target_critic = CriticNetwork()

        self.log_eta = tf.Variable(0.)
        self.log_alpha = tf.Variable(0.)

        self.policy_optimizer = tf.keras.optimizers.Adam()
        self.critic_optimizer = tf.keras.optimizers.Adam()
        self.eta_optimizer = tf.keras.optimizers.Adam()
        self.eps_mu_optimizer = tf.keras.optimizers.Adam()
        self.eps_std_optimizer = tf.keras.optimizers.Adam()

        self.batch_size = 256

        self.update_period = 4

        self.global_steps = 0

        self.setup()

    def setup(self):
        """ Initialize network weights """

        env = gym.make(self.env_id)

        dummy_state = env.reset()
        dummy_state = (dummy_state[np.newaxis, ...]).astype(np.float32)

        dummy_action = np.random.normal(0, 0.1, size=self.action_space)
        dummy_action = (dummy_action[np.newaxis, ...]).astype(np.float32)

        self.policy(dummy_state)
        self.target_policy(dummy_state)

        self.critic(dummy_state, dummy_action)
        self.target_critic(dummy_state, dummy_action)

        self.target_policy.set_weights(self.policy.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    def save(self, save_dir):
        save_dir = Path(save_dir)

        self.policy.save_weights(str(save_dir / "policy"))
        self.critic.save_weights(str(save_dir / "critic"))

    def load(self, load_dir=None):
        load_dir = Path(load_dir)

        self.policy.load_weights(str(load_dir / "policy"))
        self.target_policy.load_weights(str(load_dir / "policy"))

        self.critic.load_weights(str(load_dir / "critic"))
        self.target_critic.load_weights(str(load_dir / "critic"))

    def rollout(self):

        episode_rewards, episode_steps = 0, 0

        done = False

        env = gym.make(self.env_id)

        state = env.reset()

        while not done:

            action = self.policy.sample_action(np.atleast_2d(state))

            action = action.numpy()[0]

            next_state, reward, done, _ = env.step(action)

            transition = Transition(state, action, reward, next_state, done)

            self.replay_buffer.add(transition)

            state = next_state

            episode_rewards += reward

            episode_steps += 1

            self.global_steps += 1

            if (len(self.replay_buffer) >= 5000 and self.global_steps % self.update_period == 0):
                self.update_networks()

        with self.summary_writer.as_default():
            tf.summary.scalar("episode_reward", episode_rewards, step=self.global_steps)
            tf.summary.scalar("episode_steps", episode_steps, step=self.global_steps)

        return episode_rewards, episode_steps

    def update_networks(self):

        (states, actions, rewards,
         next_states, dones) = self.buffer.get_minibatch(batch_size=self.batch_size)

        #: Update Q-network

        #: E-step

        #: M-step
        #with self.summary_writer.as_default():
        #    tf.summary.scalar("eta", self.eta, step=self.global_steps)
        #    tf.summary.scalar("eps_kl", self.eps, step=self.global_steps)

    def testplay(self, n_repeat, monitor_dir):

        total_rewards = []

        for n in range(n_repeat):

            env = wrappers.RecordVideo(
                gym.make(self.env_id),
                video_folder=monitor_dir,
                step_trigger=lambda i: True,
                name_prefix=f"test{n}"
            )

            state = env.reset()

            done = False

            total_reward = 0

            while not done:

                action = self.policy.sample_action(np.atleast_2d(state))

                action = action.numpy()[0]

                next_state, reward, done, _ = env.step(action)

                total_reward += reward

                state = next_state

            total_rewards.append(total_reward)

            print(total_reward)


def main(env_id="BipedalWalker-v3", n_episodes=1000, n_testplay=5):
    """
    Note:
        if you failed to "pip install gym[box2d]", try "pip install box2d"
    """

    LOGDIR = Path(__file__).parent / "log"
    if LOGDIR.exists():
        shutil.rmtree(LOGDIR)

    agent = MPOAgent(env_id=env_id, logdir=LOGDIR)

    for n in range(n_episodes):

        rewards, steps = agent.rollout()

        if n % 10 == 0:
            print(f"Episode {n}: {rewards}, {steps} steps")
        break

    agent.save("checkpoints/")

    if n_testplay:
        MONITOR_DIR = Path(__file__).parent / "mp4"
        if MONITOR_DIR.exists():
            shutil.rmtree(MONITOR_DIR)

        agent = MPOAgent(env_id=env_id, logdir=LOGDIR)
        agent.load("checkpoints/")
        agent.testplay(n_repeat=n_testplay, monitor_dir=MONITOR_DIR)


if __name__ == '__main__':
    main()
