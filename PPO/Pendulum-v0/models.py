import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow_probability as tfp
import numpy as np


class PolicyNetwork(tf.keras.Model):

    def __init__(self, action_space, lr=0.0003):

        super(PolicyNetwork, self).__init__()

        self.action_space = action_space

        self.dense1 = kl.Dense(64, activation="tanh",
                               kernel_initializer="Orthogonal")

        self.dense2 = kl.Dense(64, activation="tanh",
                               kernel_initializer="Orthogonal")

        self.pi_mean = kl.Dense(self.action_space, activation="tanh",
                                kernel_initializer="Orthogonal")

        self.pi_stdev = kl.Dense(self.action_space, activation="softplus",
                                 kernel_initializer="Orthogonal")

        self.optimizer = tf.optimizers.Adam(lr=lr)

    @tf.function
    def call(self, x):

        x = self.dense1(x)

        x = self.dense2(x)

        mean = 2 * self.pi_mean(x)

        stdev = self.pi_stdev(x)

        return mean, stdev

    def sample_action(self, states):

        states = np.atleast_2d(states).astype(np.float32)

        mean, stdev = self(states)

        sampled_action = mean + stdev * tf.random.normal(tf.shape(mean))

        sampled_action = tf.clip_by_value(sampled_action, -1.9, 1.9)

        return sampled_action.numpy()


class ValueNetwork(tf.keras.Model):

    def __init__(self, lr=0.0003):

        super(ValueNetwork, self).__init__()

        self.dense1 = kl.Dense(64, activation="relu")

        self.dense2 = kl.Dense(64, activation="relu")

        self.out = kl.Dense(1)

        self.optimizer = tf.keras.optimizers.Adam(lr=lr)

    def call(self, x):

        x = self.dense1(x)
        x = self.dense2(x)
        out = self.out(x)

        return out


if __name__ == "__main__":

    policy = PolicyNetwork(action_space=1)
    s = np.array([[1, 2, 3, 4]])
    out = policy(s)
    a = policy.sample_action(s)
    print(a)
