import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.initializers import RandomUniform
import numpy as np


class ActorNetwork(tf.keras.Model):
    """memo
        batchnorm
    """

    def __init__(self, action_space):

        super(ActorNetwork, self).__init__()

        self.action_space = action_space

        self.conv1 = kl.Conv2D(32, 8, strides=4, activation="relu",
                               kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3))

        self.conv2 = kl.Conv2D(64, 4, strides=2, activation="relu",
                               kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3))

        self.conv3 = kl.Conv2D(64, 3, strides=1, activation="relu",
                               kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3))

        self.flat1 = kl.Flatten()

        self.dense1 = kl.Dense(512, activation="relu",
                               kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3))

        self.actions = kl.Dense(len(self.action_space), activation="tanh",
                                kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3))

    def call(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.flat1(x)

        x = self.dense1(x)

        action = self.actions(x)

        return action

    def sample_action(self, state, noise=True):
        """ノイズつきアクションのサンプリング
        """
        action = self(state)
        return action


class CriticNetwork(tf.keras.Model):

    def __init__(self, action_space):
        pass

    def call(self, X):
        return X


if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    frames = []

    for _ in range(4):
        frame = np.random.random(size=(84, 84))
        frames.append(frame)

    obs = np.stack(frames, axis=2)[np.newaxis, ...]

    actor = ActorNetwork(action_space=[(-1., 1.), (0., 1.), (0., 1.)])

    action = actor.sample_action(obs)

    print(action)

    #action = [-0.2, 0.8, 0.2]
