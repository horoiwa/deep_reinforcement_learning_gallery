import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


class PolicyNetwork(tf.keras.Model):
    """
    Gussian Policy with full covariance matrix
    """

    def __init__(self, action_space):

        super(PolicyNetwork, self).__init__()

        self.action_space = action_space

        self.dense1 = kl.Dense(256, activation="relu",
                               kernel_initializer="Orthogonal")

        self.dense2 = kl.Dense(256, activation="relu",
                               kernel_initializer="Orthogonal")

        self.mean = kl.Dense(self.action_space, activation="tanh",
                             kernel_initializer="Orthogonal")

        self.cholesky_factor = kl.Dense(sum(range(self.action_space+1)),
                                        kernel_initializer="Orthogonal")

    @tf.function
    def call(self, x):

        x = self.dense1(x)

        x = self.dense2(x)

        mean = self.mean(x)

        A = tfp.math.fill_triangular(
            tf.math.softplus(self.cholesky_factor(x))
        )

        covariance_matrix = tf.matmul(A, tf.transpose(A, perm=[0, 2, 1]))

        return mean, covariance_matrix

    def sample_action(self, states):

        mean, covariance_matrix = self(states)

        dist = tfd.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=covariance_matrix)

        actions = dist.sample()

        actions = tf.clip_by_value(actions, -1.0, 1.0)

        return actions


class CriticNetwork(tf.keras.Model):

    def __init__(self):

        super(CriticNetwork, self).__init__()

        self.dense_1 = kl.Dense(256, activation="relu",
                               kernel_initializer="Orthogonal")

        self.dense_2 = kl.Dense(256, activation="relu",
                               kernel_initializer="Orthogonal")

        self.q = kl.Dense(1, kernel_initializer="Orthogonal")

    @tf.function
    def call(self, states, actions):

        inputs = tf.concat([states, actions], 1)

        x1 = self.dense_1(inputs)

        x1 = self.dense_2(x1)

        q = self.q(x1)

        return q


if __name__ == "__main__":
    import gym

    env = gym.make("BipedalWalker-v3")

    policy = PolicyNetwork(action_space=4)
    critic = CriticNetwork()

    dummy_state = env.reset()
    dummy_state = (dummy_state[np.newaxis, ...]).astype(np.float32)

    dummy_action = np.random.normal(0, 0.1, size=4)
    dummy_action = (dummy_action[np.newaxis, ...]).astype(np.float32)

    action = policy(dummy_state)
    qvalue = critic(dummy_state, dummy_action)
