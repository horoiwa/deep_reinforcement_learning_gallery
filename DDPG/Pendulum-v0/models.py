import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.initializers import RandomUniform
import numpy as np


class ActorNetwork(tf.keras.Model):

    ACTION_RANGE = 2.0

    def __init__(self, action_space):

        super(ActorNetwork, self).__init__()

        self.action_space = action_space

        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)

        self.dense1 = kl.Dense(64, activation="relu")

        self.bn1 = kl.BatchNormalization()

        self.dense2 = kl.Dense(64, activation="relu")

        self.bn2 = kl.BatchNormalization()

        self.actions = kl.Dense(self.action_space, activation="tanh")

    def call(self, s, training=True):

        x = self.dense1(s)

        #x = self.bn1(x, training=training)

        x = self.dense2(x)

        #x = self.bn2(x, training=training)

        actions = self.actions(x)

        actions = actions * self.ACTION_RANGE

        return actions

    def sample_action(self, state, noise=None):
        """ノイズつきアクションのサンプリング
        """
        state = np.atleast_2d(state).astype(np.float32)

        action = self(state, training=False).numpy()[0]

        if noise:
            action += np.random.normal(0, noise*self.ACTION_RANGE, size=self.action_space)
            action = np.clip(action, -self.ACTION_RANGE, self.ACTION_RANGE)

        return action


class CriticNetwork(tf.keras.Model):

    def __init__(self):

        super(CriticNetwork, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)

        self.dense1 = kl.Dense(64, activation="relu")

        self.bn1 = kl.BatchNormalization()

        self.dense2 = kl.Dense(64, activation="relu")

        self.bn2 = kl.BatchNormalization()

        self.values = kl.Dense(1)

    def call(self, s, a, training=True):

        x = tf.concat([s, a], 1)

        x = self.dense1(x)

        #x = self.bn1(x, training=training)

        x = self.dense2(x)

        #x = self.bn2(x, training=training)

        values = self.values(x)

        return values


if __name__ == "__main__":
    import gym
    #physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)

    actor = ActorNetwork(action_space=1)

    critic = CriticNetwork()

    env = gym.make("Pendulum-v0")

    s = env.reset()

    a_ = actor.sample_action(s)
    print(a_)

    a = np.atleast_2d(a_)
    s = np.atleast_2d(s)

    q = critic(s, a)

    s2, r, done, _ = env.step([-1])

    print(s)
    print(a)
    print(q)

    w = actor.get_weights()
    print(np.array(w).shape)

    w = actor.get_weights()[0]
    print(w[0, 0])

    actor.set_weights(np.array(actor.get_weights()) +100)

    w = actor.get_weights()[0]
    print(w[0, 0])

    w = actor.get_weights()
    print(np.array(w)[0].shape)

    print(env.action_space.__dict__)
