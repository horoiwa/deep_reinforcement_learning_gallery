from pathlib import Path
import sys
import random

import numpy as np

import gym
from gym import wrappers


if __name__ == "__main__":
    """https://github.com/openai/gym/wiki/BipedalWalker-v2

        コケたら（頭が地面についたら）-100点で終了
        しかしコケないが進まない状態に陥ることもある

        obs 24次元　各関節のトルクやレーダー情報など
        action 4次元　各関節のトルク値
    """

    ENV_NAME = "BipedalWalker-v3"
    #ENV_NAME = "BipedalWalkerHardcore-v3"
    env = gym.make(ENV_NAME)
    monitor_dir = Path(__file__).parent / "bipedal"
    env = wrappers.Monitor(env, monitor_dir, force=True)

    print("Environment:", ENV_NAME)
    print("Observation space:", env.observation_space)
    print("Action space", env.action_space)
    print("Action space", env.action_space.high)
    print("Action space", env.action_space.low)
    print()
    print("Observation space", env.observation_space)

    for n in range(3):
        done = False
        observation = env.reset()
        total_rewards = 0
        for i in range(1000):
            action = np.random.uniform(-1, 1, 4)
            action = np.array([1,0,0,1])

            observation, reward, done, info = env.step(action)
            print(i, reward)
            total_rewards += reward

            if done:
                print("Done!", total_rewards)
                break
