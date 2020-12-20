import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow_probability as tfp


class GaussianPolicy(tf.keras.Model):

    def __init__(self, action_space, action_bound, lr=3e-4):

        super(GaussianPolicy, self).__init__()

        self.action_space = action_space

        self.action_bound = action_bound

        self.dense1 = kl.Dense(256, activation="relu")

        self.dense2 = kl.Dense(256, activation="relu")

        self.mu = kl.Dense(self.action_space, activation="tanh")

        self.logstd = kl.Dense(self.action_space)

        self.optimizer = tf.keras.optimizers.Adam(lr=lr)

    @tf.function
    def call(self, x):

        x = self.dense1(x)

        x = self.dense2(x)

        mu = self.mu(x)

        logstd = self.logstd(x)

        return mu, logstd

    @tf.function
    def sample_action(self, states):

        mu, logstd = self(states)

        std = tf.math.exp(logstd)

        #: Reparameterization trick
        normal_noize = tf.random.normal(shape=mu.shape, mean=0., stddev=1.)

        actions = mu + std * normal_noize

        logprob = self._compute_logprob(mu, std, actions)

        actions_squashed = tf.tanh(actions)

        logprob_squashed = logprob - tf.reduce_sum(
            tf.math.log(1 - tf.tanh(actions)**2 + 1e-6), axis=1, keepdims=True)

        actions_squashed *= self.action_bound

        return actions_squashed, logprob_squashed

    @tf.function
    def _compute_logprob(self, means, stdevs, actions):
        logprob = - 0.5 * np.log(2*np.pi)
        logprob += - tf.math.log(stdevs)
        logprob += - 0.5 * tf.square((actions - means) / stdevs)
        logprob = tf.reduce_sum(logprob, axis=1, keepdims=True)
        return logprob


class DualQNetwork(tf.keras.Model):

    def __init__(self, lr=3e-4):

        super(DualQNetwork, self).__init__()

        self.dense_11 = kl.Dense(256, activation="relu")

        self.dense_12 = kl.Dense(256, activation="relu")

        self.q1 = kl.Dense(1)

        self.dense_21 = kl.Dense(256, activation="relu")

        self.dense_22 = kl.Dense(256, activation="relu")

        self.q2 = kl.Dense(1)

        self.optimizer = tf.keras.optimizers.Adam(lr=lr)

    def call(self, states, actions):

        inputs = tf.concat([states, actions], 1)

        x1 = self.dense_11(inputs)

        x1 = self.dense_12(x1)

        q1 = self.q1(x1)

        x2 = self.dense_21(inputs)

        x2 = self.dense_22(x2)

        q2 = self.q2(x2)

        return q1, q2
