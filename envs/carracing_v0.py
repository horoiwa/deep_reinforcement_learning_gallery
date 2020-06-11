from pathlib import Path
import numpy as np
from PIL import Image
import sys

import gym
from gym import wrappers


def main():
    """
        pip install atari[all] Box2D
    """

    ENV_NAME = 'CarRacing-v0'
    env = gym.make(ENV_NAME)
    monitor_dir = Path(__file__).parent / "carracing-v0"
    env = wrappers.Monitor(env, monitor_dir, force=True)

    for i in range(1):
        play(env)


def play(env):
    print("Observation space:", env.observation_space) #(96,96,3)
    print("Action space", env.action_space) #3
    print("")

    """
    Actionは３要素のリストで渡す
    [ハンドル角度、アクセル、ブレーキ]

    ハンドル角度は -1(左) - 1(右)
    アクセルは 0-1
    ブレーキは 0-1


    1000stepで強制終了
    芝に高速で突入するとスリップする
    世界の外にでると-100点

    タイルの上を完璧に走り切ったら総得点は 1000点-0.1×タイム
    走り終わってから得点が入るのではなく

    アクセルとブレーキの同時踏みは許容される
    """

    done = False
    frame = env.reset()
    step = 0

    action1 = [0, 1, 0]
    frame, reward, done, info = env.step(action1)

    for i in range(100):
        observation, reward, done, info = env.step(action1)
        print(i, reward, done)

        if done:
            break

    action2 = [1, 0.2, 0.3]
    for i in range(1000):
        observation, reward, done, info = env.step(action2)
        print(i, reward, done)

        if done:
            break


def preproc():

    ENV_NAME = 'CarRacing-v0'
    env = gym.make(ENV_NAME)

    action = [0, 0.3, 0]
    frame = env.reset()

    for _ in range(40):
        frame, _, _, _ = env.step(action)

    frame = Image.fromarray(frame)
    frame = frame.convert("L")
    frame = frame.crop((0, 0, 96, 84))
    frame = frame.resize((84, 84))

    frame = np.array(frame, dtype=np.float32)
    frame = frame / 255



if __name__ == "__main__":
    #main()
    preproc()
