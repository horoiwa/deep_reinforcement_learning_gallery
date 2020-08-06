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


@dataclass
class Experience:

    state: np.ndarray

    action: int

    reward: float

    next_state: np.ndarray

    done: bool


class TRPOAgent:

    MAX_EXPERIENCES = 1000

    ENV_ID = "Pendulum-v0"

    def __init__(self):

        self.policy = PolicyNetwork()

        self.value_network = ValueNetwork()

        self.buffer = ReplayBuffer(max_experiences=self.MAX_EXPERIENCES)

        self.env = gym.make(self.ENV_ID)

        self.global_steps = 0

        self.hiscore = None

    def play(self, n_episodes):

        total_rewards = []

        recent_scores = collections.deque(maxlen=10)

        for n in range(n_episodes):

            total_reward, localsteps = self.play_episode()

            total_rewards.append(total_reward)

            recent_scores.append(total_reward)

            recent_average_score = sum(recent_scores) / len(recent_scores)

            print(f"Episode {n}: {total_reward}")
            print(f"Local steps {localsteps}")
            print(f"Experiences {len(self.buffer)}")
            print(f"Global step {self.global_steps}")
            print(f"recent average score {recent_average_score}")
            print()

            if (self.hiscore is None) or (recent_average_score > self.hiscore):
                self.hiscore = recent_average_score
                print(f"HISCORE Updated: {self.hiscore}")
                #self.save_model()

        return total_rewards

    def play_episode(self):

        total_reward = 0

        steps = 0

        done = False

        state = self.env.reset()

        while not done:

            action = self.policy.sample_action(state)

            next_state, reward, done, _ = self.env.step(action)

            exp = (state, action, reward, next_state, done)

            self.buffer.add_experience(exp)

            state = next_state

            total_reward += reward

            steps += 1

            self.global_steps += 1

        return total_reward, steps

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

    history = agent.play(n_episodes=30)

    print(history)
    plt.plot(range(len(history)), history)
    plt.xlabel("Episodes")
    plt.ylabel("Total Rewards")
    plt.savefig("history/log.png")

    agent.test_play(n=1, monitordir="history", load_model=False)


if __name__ == "__main__":
    main()
