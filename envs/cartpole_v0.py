from pathlib import Path
import sys
import random

import gym
from gym import wrappers

"""
Breakoutv0 と v4, deterministicの違い
https://github.com/openai/gym/issues/1280
"""

if __name__ == "__main__":

    ENV_NAME = "CartPole-v0"
    env = gym.make(ENV_NAME)
    monitor_dir = Path(__file__).parent / "cartpole-v0"
    env = wrappers.Monitor(env, monitor_dir, force=True)

    print("Environment:", ENV_NAME)
    print("Observation space:", env.observation_space)
    print("Action space", env.action_space)

    done = False
    observation = env.reset()
    action = 0
    total_steps = 0

    for _ in range(5):
        print("RANDOM")
        action = random.choice([0, 1])
        observation, reward, done, info = env.step(action)
        total_steps += 1
        if done:
            print("DONE!")
            print("STEPS", total_steps)
            sys.exit()

    for step in range(10000):

        env.step(0)
        observation, reward, done, info = env.step(action)
        total_steps += 1

        print(f"====STEP{step}====")
        print("obs", observation, type(observation))
        print("reward", reward, type(reward))
        print("done", done)
        print("info", info)
        print()

        if done:
            print("DONE!")
            print("STEPS", step)
            break
