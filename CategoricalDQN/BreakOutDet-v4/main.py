from pathlib import Path
import shutil

import gym
import numpy as np
import tensorflow as tf
import collections

from model import CategoricalQNet
from buffer import Experience, ReplayBuffer
from util import frame_preprocess


def learn(env_name="BreakoutDeterministic-v4", n_episodes=5000,
          n_atoms=51, Vmin=-10, Vmax=10, gamma=0.98,
          n_frames=4, batch_size=32, lr=0.00025, init_epsilon=0.95,
          update_period=4, target_update_period=10000):

    logdir = Path(__file__).parent / "log"
    if logdir.exists():
        shutil.rmtree(logdir)
    summary_writer = tf.summary.create_file_writer(str(logdir))

    env = gym.make(env_name)
    action_space = env.action_space.n

    qnet = CategoricalQNet(action_space, n_atoms, Vmin, Vmax)
    target_qnet = CategoricalQNet(action_space, n_atoms, Vmin, Vmax)

    buffer = ReplayBuffer(max_len=1000000)

    optimizer = tf.keras.optimizers.Adam(lr=lr, epsilon=0.01/batch_size)

    steps = 0
    for n in range(n_episodes):
        print("Episode", n)
        #: initialize env
        frames = collections.deque(maxlen=4)
        frame = frame_preprocess(env.reset())
        for _ in range(n_frames):
            frames.append(frame)

        done = False
        lives = 5
        while not done:

            steps += 1
            epsilon = max(init_epsilon * (1000000 - steps) / 1000000, 0.1)

            state = np.stack(frames, axis=2)[np.newaxis, ...]
            action = qnet.sample_action(state, epsilon=epsilon)
            next_frame, reward, done, info = env.step(action)

            frames.append(frame_preprocess(next_frame))
            next_state = np.stack(frames, axis=2)[np.newaxis, ...]

            if done:
                exp = Experience(state, action, reward, next_state, done)
                buffer.push(exp)
                break
            else:
                if info["ale.lives"] != lives:
                    lives = info["ale.lives"]
                    exp = Experience(state, action, reward, next_state, True)
                else:
                    exp = Experience(state, action, reward, next_state, done)

                buffer.push(exp)

            if (len(buffer) > 25000) and (steps % update_period ==0):
                minibatch = buffer.get_minibatch(batch_size)
                update_qnetwork(qnet, target_qnet, minibatch)

            #: hard target update
            if steps % target_update_period == 0:
                target_qnet.set_weights(qnet.get_weights())

    qnetwork.save("checkpoints/categorical_qnet")


def update_qnetwork(qnet, target_qnet, minibatch):
    pass


if __name__ == '__main__':
    learn()
