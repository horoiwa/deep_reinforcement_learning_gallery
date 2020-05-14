import collections
import random

import numpy as np
from PIL import Image
import tensorflow as tf
import gym
from gym import wrappers


def preprocess(frame):

    frame = Image.fromarray(frame)
    frame = frame.convert("L")
    frame = frame.crop((0, 20, 160, 210))
    frame = frame.resize((84, 84))
    frame = np.array(frame, dtype=np.float32)

    return frame


def main():

    env = gym.make("BreakoutDeterministic-v4")
    env = wrappers.Monitor(env, "tmp", force=True)

    state = collections.deque(maxlen=4)

    frame = env.reset()
    frame = preprocess(frame)
    for _ in range(4):
        state.append(frame)

    for _ in range(random.randint(0, 10)):
        frame, _, _, _ = env.step(1)
        frame = preprocess(frame)
        state.append(frame)

    print(frame.shape)
    print(np.stack(state, axis=2).shape)

    for i in range(100):
        #action = model.sample_action(np.stack(state, axis=2))
        frame, reward, done, info = env.step(3)
        frame = preprocess(frame)
        state.append(frame)
        print(info)

        if done:
            env.reset()

    arr = np.stack(state, axis=2)
    print(np.stack([arr, arr, arr]).shape)


if __name__ == "__main__":
    main()
