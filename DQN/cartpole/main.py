from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from dataclasses import dataclass
import collections

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl

import gym
from gym import wrappers

from models import QNetwork


@dataclass
class Experience:

    state: np.ndarray

    action: int

    reward: float

    next_state: np.ndarray

    done: bool


class DQNAgent:

    MAX_EXPERIENCES = 10000

    MIN_EXPERIENCES = 96

    BATCH_SIZE = 32

    def __init__(self, env, gamma=0.95, epsilon=1.0):
        """
            gammma: 割引率
            epsilon: 探索と活用の割合
        """

        self.env = env

        self.gamma = gamma

        self.epsion = epsilon

        self.q_network = QNetwork(self.env.action_space.n)

        self.target_network = QNetwork(self.env.action_space.n)

        self.experiences = collections.deque(maxlen=self.MAX_EXPERIENCES)

    def play(self, episodes):

        total_rewards = []

        for n in range(episodes):
            #: 探索率の更新
            self.epsilon = 1.0 / np.sqrt(n+1)

            total_reward = self.play_episode()

            total_rewards.append(total_reward)

            #: target networkの更新
            self.update_target_network()

        return total_rewards

    def play_episode(self):

        total_reward = 0

        steps = 0
        done = False
        state = self.env.reset()

        while not done and steps < 20000:

            action = self.q_network.sample_action(state, self.epsilon)

            next_state, reward, done, info = self.env.step(action)

            exp = Experience(state, action, reward, next_state, done)

            self.experiences.append(exp)

            total_reward += reward

            state = next_state

            steps += 1

            self.train()

        return total_reward

    def train(self):
        """Experienceが規定数に達していなかったら何もしない
        """
        if len(self.experiences) < self.MIN_EXPERIENCES:
            return

        (states, actions, rewards,
         next_states, dones) = self.get_minibatch(self.BATCH_SIZE)

        return

    def get_minibatch(self, batch_size):
        """Experience Replay mechanism
        """
        indices = np.random.choice(len(self.experiences),
                                   size=batch_size, replace=False)

        selected_experiences = [self.experiences[i] for i in indices]

        states = [exp.state for exp in selected_experiences]
        actions = [exp.action for exp in selected_experiences]
        rewards = [exp.rewad for exp in selected_experiences]
        next_states = [exp.next_state for exp in selected_experiences]
        dones = [exp.done for exp in selected_experiences]

        return (states, actions, rewards, next_states, dones)

    def update_target_network(self):
        """Update target Q-network
        """

        targetnet_variables = self.target_network.trainable_variables
        qnet_variables = self.q_network.trainable_variables

        for var1, var2 in zip(targetnet_variables, qnet_variables):
            var1.assign(var2.numpy())


def main():

    ENV_NAME = "CartPole-v0"
    env = gym.make(ENV_NAME)
    monitor_dir = Path(__file__).parent / "video"
    env = wrappers.Monitor(env, monitor_dir, force=True)

    agent = DQNAgent(env=env)
    agent.play(episodes=10)


if __name__ == "__main__":
    main()
