from statistics import covariance
import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow_probability as tfp
from tensorflow_probability import distribution as tfd


class PolicyNetwork(tf.keras.Model):
    """
    Gussian Policy with full covariance matrix
    """

    def __init__(self, action_space):

        super(PolicyNetwork, self).__init__()

        self.action_space = action_space

        self.dense1 = kl.Dense(256, activation="relu")

        self.dense2 = kl.Dense(256, activation="relu")

        self.mu = kl.Dense(self.action_space, activation="tanh")

        self.cholesky_factor = kl.Dense(sum(range(self.action_space+1)))

    #@tf.function
    def call(self, x):

        x = self.dense1(x)

        x = self.dense2(x)

        mu = self.mu(x)

        A = tfp.math.fill_triangular(
            tf.math.softplus(self.cholesky_factor(x))
        )

        #: Note: tensorflow_propabilityの実装では共分散行列のupper-triangleは無視されるのでAをそのまま使ってもたぶん問題ない
        covariance_matrix = tf.matmul(A, tf.transpose(A)) + 1e-6

        dist = tfd.Independent(
            tfd.MultivariateNormalFullCovariance(
                loc=mu, covariance_matrix=covariance_matrix
                )
            )

        return dist

    def sample_actions(self, states):

        dist = self(states)

        actions = dist.sample()

        return actions


class CriticNetwork(tf.keras.Model):

    def __init__(self):

        super(CriticNetwork, self).__init__()

        self.dense_1 = kl.Dense(256, activation="relu")

        self.dense_2 = kl.Dense(256, activation="relu")

        self.q = kl.Dense(1)

    @tf.function
    def call(self, states, actions):

        inputs = tf.concat([states, actions], 1)

        x1 = self.dense_1(inputs)

        x1 = self.dense_2(x1)

        q = self.q1(x1)

        return q
