import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras.layers as kl
import numpy as np


class ActorNetwork(tf.keras.Model):

    ACTION_SPACE = [(-1, 1), (0, 1), (0, 1)]

    def __init__(self):

        super(ActorNetwork, self).__init__()

        self.n_action = len(self.ACTION_SPACE)

        self.conv1 = kl.Conv2D(32, 8, strides=4, activation="relu",
                               kernel_initializer="he_normal")

        self.conv2 = kl.Conv2D(64, 4, strides=2, activation="relu",
                               kernel_initializer="he_normal")

        self.conv3 = kl.Conv2D(64, 3, strides=1, activation="relu",
                               kernel_initializer="he_normal")

        self.flat1 = kl.Flatten()

        self.dense1 = kl.Dense(512, activation="relu",
                               kernel_initializer="he_normal")

        self.actions = kl.Dense(self.n_action, activation="tanh",
                                kernel_initializer="he_normal")

    def call(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.flat1(x)

        x = self.dense1(x)

        action = self.actions(x)

        return action


class CriticNetwork(tf.keras.Model):

    def __init__(self):
        pass

    def call(self, X):
        return X


if __name__ == "__main__":

    frames = []

    for _ in range(4):
        frame = np.random.random(size=(84, 84))
        frames.append(frame)

    obs = np.stack(frames, axis=2)[np.newaxis, ...]

    actor = ActorNetwork()

    action = actor.predict(obs)

    print(action)

    #action = [-0.2, 0.8, 0.2]
