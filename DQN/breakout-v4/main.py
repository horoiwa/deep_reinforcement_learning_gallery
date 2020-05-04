from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from dataclasses import dataclass
import random
import collections
from datetime import datetime

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


@tf.function
def frame_preprocessor(frame):
    """
    Args:
        frame (np.ndarray): shape=(84, 84, 3)

    Returns:
        tf.Tensor: shape=(84, 84, 1)
    """
    image = tf.cast(tf.convert_to_tensor(frame), tf.float32)
    image_gray = tf.image.rgb_to_grayscale(image)
    image_crop = tf.image.crop_to_bounding_box(image_gray, 34, 0, 160, 160)
    image_resize = tf.image.resize(image_crop, [84, 84])
    image_scaled = tf.divide(image_resize, 255)

    return image_scaled


class DQNAgent:

    MAX_EXPERIENCES = 300000

    MIN_EXPERIENCES = 30000

    BATCH_SIZE = 32

    def __init__(self, env, gamma=0.98, epsilon=1.0):
        """
            gammma: 割引率
            epsilon: 探索と活用の割合
        """

        self.env = env

        self.gamma = gamma

        self.epsion = epsilon

        self.copy_period = 250

        self.global_steps = 0

        self.q_network = QNetwork(self.env.action_space.n)

        self.target_network = QNetwork(self.env.action_space.n)

        self.experiences = collections.deque(maxlen=self.MAX_EXPERIENCES)

    def play(self, episodes):

        total_rewards = []

        recent_localsteps = collections.deque(maxlen=5)

        for n in range(episodes):

            self.epsilon = 1.0 - min(0.9, self.global_steps * 0.9 / 200000)

            total_reward, localsteps = self.play_episode()

            total_rewards.append(total_reward)

            recent_localsteps.append(localsteps)

            recent_average = sum(recent_localsteps) / len(recent_localsteps)

            if recent_average > self.copy_period:
                self.copy_period *= 2

            print(f"Episode {n}: {total_reward}")
            print(f"Reward {total_reward}")
            print(f"Local steps {localsteps}")
            print(f"Experiences {len(self.experiences)}")
            print(f"Current epsilon {self.epsilon}")
            print(f"Current copy period {self.copy_period}")
            print(f"Global step {self.global_steps}")
            print()

        return total_rewards

    def play_episode(self):

        total_reward = 0

        lives = 5

        steps = 0

        done = False

        frame = self.env.reset()

        state = np.stack(
            [np.squeeze(frame_preprocessor(frame))]*4,
            axis=2).astype(np.float32)

        while not done:

            action = self.sample_action(state)

            frame, reward, done, info = self.env.step(action)

            next_state = np.append(frame_preprocessor(frame),
                                   state[..., :3], axis=2)

            if info["ale.lives"] != lives:
                lives = info["ale.lives"]
                exp = Experience(state, action, -100, next_state, done)
            else:
                exp = Experience(state, action, reward, next_state, done)

            exp = Experience(state, action, reward, next_state, done)

            self.experiences.append(exp)

            state = next_state

            self.update_qnetwork()

            total_reward += reward

            steps += 1

            self.global_steps += 1

            if self.global_steps % self.copy_period == 0:
                print("==Update target newwork==")
                self.update_target_network()


        return total_reward, steps

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

        next_Qs = np.max(
            self.target_network.predict(np.array(next_states)), axis=1)

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

    start = datetime.now()

    monitor_dir = Path(__file__).parent / "history"
    ENV_NAME = "Breakout-v4"
    env = gym.make(ENV_NAME)
    env = wrappers.Monitor(env, monitor_dir, force=True,
                           video_callable=(lambda ep: ep % 100 == 0))

    agent = DQNAgent(env=env)
    history = agent.play(episodes=5001)


    plt.plot(range(len(history)), history)
    plt.xlabel("episodes")
    plt.ylabel("Total Reward")
    plt.savefig(monitor_dir / "dqn_breakout-v4.png")

    df = pd.DataFrame()
    df["Total Reward"] = history
    df.to_csv(monitor_dir / "dqn_breakout-v4.csv", index=None)

    end = datetime.now() - start
    print(end)


if __name__ == "__main__":
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    print()

    main()
