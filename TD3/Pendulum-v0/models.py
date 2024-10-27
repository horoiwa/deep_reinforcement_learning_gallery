import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.initializers import RandomUniform
import numpy as np


class ActorNetwork(tf.keras.Model):

    def __init__(self, action_space, max_action):

        super(ActorNetwork, self).__init__()

        self.action_space = action_space

        self.max_action = max_action

        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)

        self.dense1 = kl.Dense(64, activation="relu")

        self.dense2 = kl.Dense(64, activation="relu")

        self.actions = kl.Dense(self.action_space, activation="tanh")

    def call(self, s, training=True):

        x = self.dense1(s)

        x = self.dense2(x)

        actions = self.actions(x)

        actions = actions * self.max_action

        return actions

    def sample_action(self, state, noise=None):
        """ノイズつきアクションのサンプリング
        """
        state = np.atleast_2d(state).astype(np.float32)

        action = self(state, training=False).numpy()[0]

        if noise:
            action += np.random.normal(0, noise*self.max_action,
                                       size=self.action_space)
            action = np.clip(action, -self.max_action, self.max_action)

        return action


class CriticNetwork(tf.keras.Model):

    def __init__(self):

        super(CriticNetwork, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)

        self.dense1 = kl.Dense(64, activation="relu")

        self.dense2 = kl.Dense(64, activation="relu")

        self.out1 = kl.Dense(1)

        self.dense3 = kl.Dense(64, activation="relu")

        self.dense4 = kl.Dense(64, activation="relu")

        self.out2 = kl.Dense(1)

    def call(self, s, a, training=True):

        x = tf.concat([s, a], 1)

        x1 = self.dense1(x)

        x1 = self.dense2(x1)

        q1 = self.out1(x1)

        x2 = self.dense3(x)

        x2 = self.dense4(x2)

        q2 = self.out2(x2)

        return q1, q2


if __name__ == "__main__":
    import gym
    #physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)

    actor = ActorNetwork(action_space=1, max_action=2)

    critic = CriticNetwork()

    env = gym.make("Pendulum-v0")

    s = env.reset()

    a_ = actor.sample_action(s)
    print(a_)

    a = np.atleast_2d(a_)
    s = np.atleast_2d(s)

    q = critic(s, a)

    s2, r, done, _ = env.step([-1])

    s_ = np.vstack([s]*5)
    a_ = np.vstack([a]*5)

    q1, q2 = critic(s_, a_)

    print(q1)
    for q in q1.numpy().flatten():
        print(q)
