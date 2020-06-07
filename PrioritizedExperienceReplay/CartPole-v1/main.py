from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from dataclasses import dataclass
import collections
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as kl
import matplotlib.pyplot as plt
import gym
from gym import wrappers

from models import QNetwork
from buffer import PrioritizedReplayBuffer


@dataclass
class Experience:

    state: np.ndarray

    action: int

    reward: float

    next_state: np.ndarray

    done: bool


class DQNAgent:

    MAX_EXPERIENCES = 20000

    MIN_EXPERIENCES = 512

    BATCH_SIZE = 16

    BETA_INIT = 0.5

    def __init__(self, env, gamma=0.95, epsilon=1.0,
                 copy_period=1000, lr=0.01, update_period=2):
        """
            gammma: 割引率
            epsilon: 探索と活用の割合
        """

        self.env = env

        self.gamma = gamma

        self.epsion = epsilon

        self.copy_period = copy_period

        self.update_period = update_period

        self.lr = lr

        self.global_steps = 0

        self.q_network = QNetwork(self.env.action_space.n, lr=lr)

        self.q_network.build(input_shape=(None, 4))

        self.target_network = QNetwork(self.env.action_space.n)

        self.target_network.build(input_shape=(None, 4))

        self.replay_buffer = PrioritizedReplayBuffer(
            max_experiences=self.MAX_EXPERIENCES)

        self.beta = self.BETA_INIT

    def play(self, n_episodes):

        total_rewards = []

        for n in range(n_episodes):

            self.beta = self.BETA_INIT + (1-self.BETA_INIT) * (n / n_episodes)

            self.epsilon = 1.0 - min(0.95, self.global_steps * 0.95 / 500)

            total_reward = self.play_episode()

            total_rewards.append(total_reward)

            print(f"Episode {n}: {total_reward}")
            print(f"Current buffer size {len(self.replay_buffer)}")
            print(f"Current beta {self.beta}")
            print(f"Current epsilon {self.epsilon}")
            print()

        return total_rewards

    def play_episode(self):

        total_reward = 0

        steps = 0

        done = False

        state = self.env.reset()

        while not done:

            action = self.sample_action(state)

            next_state, reward, done, info = self.env.step(action)

            total_reward += reward

            exp = Experience(state, action, reward, next_state, done)

            self.replay_buffer.add_experience(exp)

            state = next_state

            steps += 1

            self.global_steps += 1

            if self.global_steps % self.update_period == 0:
                self.update_qnetwork()

            if self.global_steps % self.copy_period == 0:
                self.target_network.set_weights(self.q_network.get_weights())

        return total_reward

    def sample_action(self, state):
        """探索と活用"""

        if np.random.random() < self.epsilon:
            random_action = np.random.choice(self.env.action_space.n)
            return random_action
        else:
            selected_action = np.argmax(self.q_network.predict(state))
            return selected_action

    def update_qnetwork(self):
        """ Q-Networkの訓練
            ただしExperiencesが規定数に達していないうちは何もしない
        """
        if len(self.replay_buffer) < self.MIN_EXPERIENCES:
            return

        indices, weights, experiences = self.replay_buffer.get_minibatch(self.BATCH_SIZE, self.beta)

        states = [exp.state for exp in experiences]

        actions = [exp.action for exp in experiences]

        rewards = [exp.reward for exp in experiences]

        next_states = [exp.next_state for exp in experiences]

        dones = [exp.done for exp in experiences]

        next_Qs = np.max(self.target_network.predict(next_states), axis=1)

        target_values = [reward + self.gamma * next_q if not done else reward
                         for reward, next_q, done
                         in zip(rewards, next_Qs, dones)]

        td_errors = self.q_network.update(
            np.vstack(states), np.array(actions),
            np.vstack(target_values), np.vstack(weights))

        self.replay_buffer.update_priority(indices, td_errors)


def main(copy_period, lr):

    monitor_dir = Path(__file__).parent / f"CP{copy_period}_LR{lr}"

    ENV_NAME = "CartPole-v1"
    env = gym.make(ENV_NAME)
    env = wrappers.Monitor(env, monitor_dir, force=True,
                           video_callable=(lambda ep: ep % 100 == 0))

    agent = DQNAgent(env=env, copy_period=copy_period, lr=lr)
    history = agent.play(n_episodes=501)
    print(history)

    plt.plot(range(len(history)), history)
    plt.plot([0, 400], [195, 195], "--", color="darkred")
    plt.xlabel("episodes")
    plt.ylabel("Total Reward")
    plt.savefig(monitor_dir / "dqn_cartpole-v1.png")

    df = pd.DataFrame()
    df["Total Reward"] = history
    df.to_csv(monitor_dir / "dqn_cartpole-v1.csv", index=None)


if __name__ == "__main__":
    copy_period = 250
    lr = 0.0005
    main(copy_period, lr)
