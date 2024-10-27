import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import functools
import random
from pathlib import Path
import collections

import gym
from gym import wrappers
import numpy as np
import pandas as pd
from multiprocessing import Process, Pipe
import matplotlib.pyplot as plt

from env import SubProcVecEnv, preprocess
from models import ActorCriticNet


def envfunc_proto(env_id):
    env = gym.make("BreakoutDeterministic-v4")
    env.seed = env_id
    return env


class A2CAgent:

    TRAJECTORY_SIZE = 5

    ACTION_SPACE = 4

    def __init__(self, n_procs, gamma=0.99, weights=None):

        self.n_procs = n_procs

        self.ACNet = ActorCriticNet(action_space=self.ACTION_SPACE)

        if weights:
            pass

        self.gamma = gamma

        self.vecenv = SubProcVecEnv(
            [functools.partial(envfunc_proto, env_id=i)
             for i in range(self.n_procs)])

        self.states = None

        self.batch_size = self.n_procs * self.TRAJECTORY_SIZE

        self.hiscore = 0

    def run(self, total_steps, test_freq=10000):

        self.states = self.vecenv.reset()

        test_scores = []

        steps = 0

        for _ in range(total_steps // (self.n_procs * self.TRAJECTORY_SIZE)):

            mb_states, mb_actions, mb_discounted_rewards = self.run_Nsteps()

            states = mb_states.reshape((self.batch_size, 84, 84, 4))

            selected_actions = mb_actions.reshape(self.batch_size, -1)

            discounted_rewards = mb_discounted_rewards.reshape(self.batch_size, -1)

            self.ACNet.update(states, selected_actions, discounted_rewards)

            if steps % test_freq == 0:

                scores = self.play(5)

                average_score = sum(scores) / len(scores)

                test_scores.append(average_score)

                print("Test Play:", average_score)

                if average_score > self.hiscore:
                    self.hiscore = average_score
                    self.save_model()
                    print("Hi Score Update:", self.hiscore)

                print("Current Hiscore:", self.hiscore)

            steps += self.n_procs * self.TRAJECTORY_SIZE

            print("Step:", steps)

        return test_scores

    def run_Nsteps(self):

        mb_states, mb_actions, mb_rewards, mb_dones = [], [], [], []

        for _ in range(self.TRAJECTORY_SIZE):

            states = np.array(self.states)

            actions = self.ACNet.sample_action(states)

            rewards, next_states, dones, _ = self.vecenv.step(actions)

            mb_states.append(states)
            mb_actions.append(actions)
            mb_rewards.append(rewards)
            mb_dones.append(dones)

            self.states = next_states

        mb_states = np.array(mb_states).swapaxes(0, 1)
        mb_actions = np.array(mb_actions).T
        mb_rewards = np.array(mb_rewards).T
        mb_dones = np.array(mb_dones).T

        """Calculate Discounted Rewards
        """
        last_values, _ = self.ACNet.predict(self.states)

        mb_discounted_rewards = np.zeros(mb_rewards.shape)
        for n, (rewards, dones, last_value) in enumerate(zip(mb_rewards, mb_dones, last_values.flatten())):
            rewards = rewards.tolist()
            dones = dones.tolist()

            discounted_rewards = self.discount_with_dones(rewards, dones, last_value)
            mb_discounted_rewards[n] = discounted_rewards

        return (mb_states, mb_actions, mb_discounted_rewards)

    def discount_with_dones(self, rewards, dones, last_value):

        R = 0 if dones[-1] else last_value

        discounted_rewards = []
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + (not done) * self.gamma * R
            discounted_rewards.append(R)

        discounted_rewards.reverse()

        return discounted_rewards

    def save_model(self):

        self.ACNet.save_weights("checkpoints/best")

    def load_model(self, weights_path):

        self.ACNet.load_weights(weights_path)

    def play(self, n=1, monitordir=None, log=False):

        if monitordir:
            env = wrappers.Monitor(gym.make("BreakoutDeterministic-v4"),
                                   monitordir, force=True,
                                   video_callable=(lambda ep: True))
        else:
            env = gym.make("BreakoutDeterministic-v4")

        total_rewards = []

        for i in range(n):

            frame = preprocess(env.reset())

            frames = collections.deque(maxlen=4)

            for _ in range(4):
                frames.append(frame)

            done = False

            total_reward = 0

            while not done:
                state = np.stack(frames, axis=2)[np.newaxis, ...]

                action = self.ACNet.sample_action(state)

                frame, reward, done, _ = env.step(action[0])

                frames.append(preprocess(frame))

                total_reward += reward

            total_rewards.append(total_reward)

            if log:
                print(i, total_reward)

        return total_rewards


def main():

    MONITOR_DIR = Path(__file__).parent / "log"

    total_steps = 25000000

    test_freq = 15000

    agent = A2CAgent(n_procs=15)

    log_testplay = agent.run(total_steps=total_steps, test_freq=test_freq)

    agent.load_model("checkpoints/best")
    agent.play(10, monitordir=MONITOR_DIR)

    steps = np.arange(1, total_steps+1, test_freq)
    steps = steps / 1000

    plt.plot(steps, log_testplay)
    plt.xlabel("steps")
    plt.ylabel("Total rewards")
    plt.savefig(MONITOR_DIR / "testplay.png")

    df = pd.DataFrame()
    df["steps"] = steps
    df["testscore"] = log_testplay
    df.to_csv(MONITOR_DIR / "testplay.csv")


def play():

    MONITOR_DIR = Path(__file__).parent / "log"

    agent = A2CAgent(n_procs=1)

    agent.load_model("checkpoints25M/best")

    agent.play(10, monitordir=MONITOR_DIR, log=True)


if __name__ == "__main__":
    main()
    #play()
