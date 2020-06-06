from pathlib import Path
from PIL import Image

import gym
from gym import wrappers


def main1():

    ENV_NAME = "SpaceInvadersDeterministic-v4"
    env = gym.make(ENV_NAME)
    monitor_dir = Path(__file__).parent / "spaceinvadorsDet-v4"
    env = wrappers.Monitor(env, monitor_dir, force=True)

    print("Environment:", ENV_NAME)
    print("Observation space:", env.observation_space)
    print("Action space", env.action_space)
    print("")

    """
    Actions
    0: Do nothing
    1: Fire (start/restart game)
    2: move right
    3: move left
    """

    done = False
    observation = env.reset()
    step = 0

    observation, reward, done, info = env.step(0)
    lives = info["ale.lives"]

    for _ in range(45):
        env.step(2)

    env.step(1)

    for i in range(1000):
        if i % 3 == 0:
            observation, reward, done, info = env.step(0)
        elif i % 3 == 1:
            observation, reward, done, info = env.step(0)
        else:
            observation, reward, done, info = env.step(0)

        print(reward, info)

        if done:
            break

"""
0: NoOp
1: Fire
2: Right
3: Left

初期位置は最左
ライフ３
入力なしで強制開始

はじめの３秒程度は入力を受け付けないので開始時ランダムは長めにとる必要がある
45-60くらいで調整

"""

import numpy as np

def preproc():
    env = gym.make("SpaceInvadersDeterministic-v4")
    frame = env.reset()
    for _ in range(40):
        frame, _, _, _ = env.step(1)

    frame = Image.fromarray(frame)
    frame = frame.convert("L")
    #frame = frame.crop((0, 0, 160, 200))
    frame = frame.crop((0, 20, 160, 200))
    frame = frame.resize((84, 84))
    frame.show()
    frame = np.array(frame, dtype=np.float32)
    frame = frame / 255

    return frame



if __name__ == "__main__":
    #main1()
    preproc()
