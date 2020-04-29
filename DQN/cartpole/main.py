from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from dataclasses import dataclass
import collections

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as kl
import matplotlib.pyplot as plt
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

    MAX_EXPERIENCES = 50000

    MIN_EXPERIENCES = 256

    BATCH_SIZE = 32

    def __init__(self, env, gamma=0.95, epsilon=1.0):
        """
            gammma: 割引率
            epsilon: 探索と活用の割合
        """

        self.env = env

        self.gamma = gamma

        self.epsion = epsilon

        self.copy_period = 10

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

            recent_score = (sum(total_rewards[-5:]) / 5)

            self.copy_period = max(int(recent_score*0.8), 10)

            print(f"Episode {n}: {total_reward}")
            print(f"Current copy period {self.copy_period}")
            print(f"Current experiences {len(self.experiences)}")
            print()

        return total_rewards

    def play_episode(self):

        total_reward = 0

        steps = 0
        done = False
        state = self.env.reset()

        while not done and steps < 20000:

            action = self.sample_action(state)

            next_state, reward, done, info = self.env.step(action)

            reward = reward if not done else -100

            exp = Experience(state, action, reward, next_state, done)

            self.experiences.append(exp)

            total_reward += reward

            state = next_state

            steps += 1

            self.update_qnetwork()

            if (steps != 0) and (steps % self.copy_period == 0):
                self.update_target_network()

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
        if len(self.experiences) < self.MIN_EXPERIENCES:
            return

        (states, actions, rewards,
         next_states, dones) = self.get_minibatch(self.BATCH_SIZE)

        next_Qs = np.max(self.target_network.predict(next_states), axis=1)

        target_values = [reward + self.gamma * next_q if not done else reward
                         for reward, next_q, done
                         in zip(rewards, next_Qs, dones)]

        self.q_network.update(np.array(states), np.array(actions),
                              np.array(target_values))

    def update_target_network(self):
        """Update target Q-network
        """

        targetnet_variables = self.target_network.trainable_variables
        qnet_variables = self.q_network.trainable_variables

        for var1, var2 in zip(targetnet_variables, qnet_variables):
            var1.assign(var2.numpy())

    def get_minibatch(self, batch_size):
        """Experience Replay mechanism
        """
        indices = np.random.choice(len(self.experiences),
                                   size=batch_size, replace=False)

        selected_experiences = [self.experiences[i] for i in indices]

        states = [exp.state for exp in selected_experiences]
        actions = [exp.action for exp in selected_experiences]
        rewards = [exp.reward for exp in selected_experiences]
        next_states = [exp.next_state for exp in selected_experiences]
        dones = [exp.done for exp in selected_experiences]

        return (states, actions, rewards, next_states, dones)


def main():

    monitor_dir = Path(__file__).parent / "history"

    ENV_NAME = "CartPole-v0"
    env = gym.make(ENV_NAME)
    env = wrappers.Monitor(env, monitor_dir, force=True)

    agent = DQNAgent(env=env)
    history = agent.play(episodes=250)

    plt.plot(range(len(history)), history)
    plt.xlabel("episodes")
    plt.ylabel("Total Reward")
    plt.savefig(monitor_dir / "dqn_cartpole.png")

    df = pd.DataFrame()
    df["Total Reward"] = history
    df.to_csv(monitor_dir / "dqn_cartpole.csv", index=None)


if __name__ == "__main__":
    main()
