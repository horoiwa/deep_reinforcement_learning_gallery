from dataclasses import dataclass
import collections
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import numpy as np
import tensorflow as tf
import gym
from gym import wrappers
import matplotlib.pyplot as plt

from buffer import ReplayBuffer
from models import PolicyNetwork, ValueNetwork


class TRPOAgent:

    MAX_EXPERIENCES = 1000

    BATCH_SIZE = 1024

    ENV_ID = "Pendulum-v0"

    ACTION_SPACE = 1

    def __init__(self):

        self.policy = PolicyNetwork(action_space=self.ACTION_SPACE)

        self.value_network = ValueNetwork()

        self.env = gym.make(self.ENV_ID)

        self.global_steps = 0

        self.history = []

        self.hiscore = 0

    def play(self, n_iters):

        self.epi_reward = 0

        self.state = self.env.reset()

        for n in range(n_iters):

            trajectory = self.generate_trajectory()

            print(trajectory)

    def generate_trajectory(self):
        """generate trajectory on current policy
        """

        trajectory = {"s": np.zeros((self.BATCH_SIZE, self.ACTION_SPACE), dtype=np.float32),
                      "a": np.zeros((self.BATCH_SIZE, 1), dtype=np.float32),
                      "r": np.zeros((self.BATCH_SIZE, 1), dtype=np.float32),
                      "done": np.zeros((self.BATCH_SIZE, 1), dtype=np.float32),
                      "s2": np.zeros((self.BATCH_SIZE, self.ACTION_SPACE), dtype=np.float32)}

        state = self.state

        for i in range(self.BATCH_SIZE):

            action = self.policy.sample_action(state)

            next_state, reward, done, _ = self.env.step(action)

            trajectory["s"][i] = state

            trajectory["a"][i] = action

            trajectory["reward"][i] = reward

            trajectory["s2"][i] = next_state

            trajectory["done"][i] = done

            self.epi_reward += reward

            self.global_steps += 1

            if done:
                state = self.env.reset()

                self.history.append(self.epi_reward)

                recent_score = self.history[-10:] / 10

                print("===="*5)
                print("Episode:", len(self.history))
                print("Episode reward:", self.epi_reward)
                print("Global steps:", self.global_steps)

                if len(self.hiscore) > 100 and recent_score > self.hiscore:
                    print("*HISCORE UPDATED:", recent_score)
                    self.save_model()

                self.epi_reward = 0

            else:
                state = next_state

        self.state = state

        return trajectory

    def calc_logprob(self, actions, action_means, action_logvars):
        pass

    def update(self):

        def hvp_wrapper(g):
            def hvp_func(x):
                pass

            return hvp_func

    def save_model(self):

        self.policy.save_weights("checkpoints/actor")

        self.value_network.save_weights("checkpoints/critic")

    def load_model(self):

        self.policy.load_weights("checkpoints/actor")

        self.value_network.load_weights("checkpoints/critic")

    def test_play(self, n, monitordir, load_model=False):

        if load_model:
            self.load_model()

        if monitordir:
            env = wrappers.Monitor(gym.make(self.ENV_ID),
                                   monitordir, force=True,
                                   video_callable=(lambda ep: ep % 1 == 0))
        else:
            env = gym.make(self.ENV_ID)

        for i in range(n):

            total_reward = 0

            steps = 0

            done = False

            state = env.reset()

            while not done:

                action = self.policy.sample_action(state)

                next_state, reward, done, _ = env.step(action)

                state = next_state

                total_reward += reward

                steps += 1

            print()
            print(f"Test Play {i}: {total_reward}")
            print(f"Steps:", steps)
            print()


def main():

    agent = TRPOAgent()

    history = agent.play(n_iters=1)

    print(history)

    plt.plot(range(len(history)), history)
    plt.xlabel("Episodes")
    plt.ylabel("Total Rewards")
    plt.savefig("history/log.png")

    agent.test_play(n=1, monitordir="history", load_model=False)


if __name__ == "__main__":
    main()
