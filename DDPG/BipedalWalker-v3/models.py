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

        self.dense1 = kl.Dense(128, activation="relu",
                               kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3))

        self.bn1 = kl.BatchNormalization()

        self.dense2 = kl.Dense(64, activation="relu",
                               kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3))

        self.bn2 = kl.BatchNormalization()

        self.actions = kl.Dense(len(self.action_space), activation="tanh",
                                kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3))

    def call(self, s, training=True):

        x = self.dense1(s)

        x = self.bn1(x, training=training)

        x = self.dense2(x)

        x = self.bn2(x, training=training)

        actions = self.actions(x)

        return actions

    def sample_action(self, state, noise):
        """ノイズつきアクションのサンプリング
        """
        state = np.atleast_2d(state).astype(np.float32)

        action = self(state, training=False).numpy()[0]

        if noise:
            action += np.random.normal(0, 0.1, size=len(self.action_space))
            action = np.clip(action, -1., 1.)

        return action


class CriticNetwork(tf.keras.Model):

    def __init__(self, action_space):

        super(CriticNetwork, self).__init__()

        self.action_space = action_space

        self.dense1 = kl.Dense(128, activation="relu",
                               kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3))

        self.bn1 = kl.BatchNormalization()

        self.dense2 = kl.Dense(64, activation="relu",
                               kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3))

        self.bn2 = kl.BatchNormalization()

        self.values = kl.Dense(1, kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3))

    def call(self, s, a, training=True):

        x = tf.concat([s, a], -1)

        x = self.dense1(x)

        x = self.bn1(x, training=training)

        x = self.dense2(x)

        x = self.bn2(x, training=training)

        values = self.values(x)

        return values

    def evaluate(self, s, a):

        s = np.atleast_2d(s).astype(np.float32)

        a = np.atleast_2d(a).astype(np.float32)

        q = self(s, a, training=False).numpy()[0][0]

        return q



if __name__ == "__main__":
    import gym
    #physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)

    actor = ActorNetwork(action_space=[(-1., 1.), (-1., 1.), (-1., 1.), (-1., 1.)])

    critic = CriticNetwork(action_space=[(-1., 1.), (-1., 1.), (-1., 1.), (-1., 1.)])

    env = gym.make("BipedalWalker-v3")

    obs = env.reset()

    action = actor.sample_action(obs, noise=True)

    q = critic.evaluate(obs, action)

    print(action)
    print(q)

