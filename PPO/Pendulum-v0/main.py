import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pathlib import Path

import gym
from gym import wrappers
import numpy as np
from multiprocessing import Process, Pipe
import matplotlib.pyplot as plt

from env import VecEnv
from models import PolicyNetwork, ValueNetwork


def env_func():
    return gym.make("Pendulum-v0")


class PPOAgent:

    TRAJECTORY_SIZE = 64

    OBS_SPACE = 3

    GAMMA = 0.99

    LAMBDA = 0.95

    def __init__(self, env_id, action_space, n_envs):

        self.env_id = env_id

        self.n_envs = n_envs

        self.vecenv = VecEnv(env_id=self.env_id, n_envs=self.n_envs)

        self.policy = PolicyNetwork(action_space=action_space)

        self.value = ValueNetwork()

    def run(self, n_epochs, trajectory_size):

        history = {"steps": [], "scores": []}

        states = self.vecenv.reset()

        hiscore = 0

        for epoch in range(n_epochs):

            for _ in range(trajectory_size):

                actions = self.policy(states)

                states = self.vecenv.step(actions)

            trajectories = self.vecenv.get_trajectories()
            trajectories = self.compute_advantage(trajectories)

            states, actions, advantages = self.create_minibatch(trajectories)

            self.update_policy(states, actions, advantages)
            self.update_value(states, advantages)

            global_steps = (epoch+1) * trajectory_size * self.n_envs
            test_score = np.array(self.play(n=3)).mean()
            history["steos"].append(global_steps)
            history["scores"].append(test_score)

            ma_score = sum(history["scores"][-10:]) / 10
            if epoch > 10 and ma_score > hiscore:
                self.save_model()
                print("Model Saved")

            print(f"Epoch {epoch}, {global_steps//1000}K, {test_score}")

        return history

    def compute_advantage(self, trajectories):
        for trajectory in trajectories:
            pass
        return trajectories

    def create_minibatch(self, trajectory):

        states = None
        actions = None
        advantages = None

        return states, actions, advantages

    def update_policy(self, states, actions, advantages):
        raise NotImplementedError()

    def update_value(self, states, advantages):
        raise NotImplementedError()

    def save_model(self):

        self.policy.save_weights("checkpoints/policy")

        self.value.save_weights("checkpoints/value")

    def load_model(self, weights_path):

        self.policy.load_weights("checkpoints/policy")

        self.value.load_weights("checkpoints/value")

    def play(self, n=1, monitordir=None):

        if monitordir:
            env = wrappers.Monitor(gym.make(self.env_id),
                                   monitordir, force=True,
                                   video_callable=(lambda ep: True))
        else:
            env = gym.make(self.env_id)

        total_rewards = []

        for _ in range(n):

            obs = env.reset()

            done = False

            total_reward = 0

            while not done:

                action = self.policy.sample_action(obs)

                obs, reward, done, _ = env.step(action)

                total_reward += reward

            total_rewards.append(total_reward)

        return total_rewards


def main():

    MONITOR_DIR = Path(__file__).parent / "log"

    agent = PPOAgent(env_id="Pendulum-v0", action_space=1, n_envs=4)

    history = agent.run(n_epochs=10, trajectory_size=512)

    #agent.load_model()
    #agent.play(5, monitordir=MONITOR_DIR)

    plt.plot(history["steps"], history["scores"])
    plt.xlabel("steps")
    plt.ylabel("Total rewards")
    plt.savefig(MONITOR_DIR / "testplay.png")


if __name__ == "__main__":
    main()
