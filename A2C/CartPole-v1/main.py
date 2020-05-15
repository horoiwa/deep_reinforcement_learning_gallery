import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import functools
import random

import gym
import numpy as np
from multiprocessing import Process, Pipe

from env import SubProcVecEnv
from models import ActorCriticNet


def envfunc_proto(env_id):
    env = gym.make("CartPole-v1")
    env.seed = env_id
    return env


class A2CAgent:

    TRAJECTORY_SIZE = 5

    ACTION_SPACE = 2

    TEST_FREQ = 1000

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

    def run(self, total_steps):

        self.states = self.vecenv.reset()

        test_scores = []

        steps = 0

        for n in range(total_steps // (self.n_procs * self.TRAJECTORY_SIZE)):

            mb_states, mb_actions, mb_discounted_rewards = self.run_Nsteps()

            states = mb_states.reshape(self.batch_size, 4)

            selected_actions = mb_actions.reshape(self.batch_size, -1)

            discounted_rewards = mb_discounted_rewards.reshape(self.batch_size, -1)

            self.ACNet.update(states, selected_actions, discounted_rewards)

            steps += self.n_procs * self.TRAJECTORY_SIZE

            print("Step:", steps)

            if steps % self.TEST_FREQ == 0:

                test_score = self.play()

                test_scores.append(test_score)

                print("Test Play:", test_score)

        return test_scores

    def run_Nsteps(self):

        mb_states, mb_values, mb_actions, mb_rewards, mb_dones = [], [], [], [], []

        for _ in range(self.TRAJECTORY_SIZE):

            states = np.array(self.states)

            actions = self.ACNet.sample_action(states)

            rewards, next_states, dones, infos = self.vecenv.step(actions)

            mb_states.append(states)
            mb_actions.append(actions)
            mb_rewards.append(rewards)
            mb_dones.append(dones)

            self.states = next_states

        mb_states = np.array(mb_states).swapaxes(0, 1)
        mb_actions = np.array(mb_actions).T
        mb_rewards = np.array(mb_rewards).T
        mb_values = np.array(mb_values).T
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

    def play(self):

        env = gym.make("CartPole-v1")

        obs = env.reset()

        done = False

        total_rewards = 0

        while not done:

            action = self.ACNet.sample_action(obs)

            obs, reward, done, info = env.step(action[0])

            total_rewards += reward

        return total_rewards



def main():
    agent = A2CAgent(n_procs=10)
    test_history = agent.run(total_steps=40000)
    print(test_history)


if __name__ == "__main__":
    main()
