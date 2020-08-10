import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow_probability as tfp


class PolicyNetwork(tf.keras.Model):

    MAX_KL = 0.01

    def __init__(self, action_space):

        super(PolicyNetwork, self).__init__()

        self.action_space = action_space

        self.dense1 = kl.Dense(64, activation="tanh",
                               kernel_initializer="he_normal")

        self.dense2 = kl.Dense(64, activation="tanh",
                               kernel_initializer="he_normal")

        self.pi_mean = kl.Dense(self.action_space,
                                kernel_initializer="he_normal")

        self.pi_std = kl.Dense(self.action_space,
                               kernel_initializer="he_normal")

    def call(self, s):

        x = self.dense1(s)
        x = self.dense2(x)
        pi_mean = self.pi_mean(x)
        pi_std = self.pi_std(x)

        return pi_mean, pi_std

    def sample_action(self, state):

        state = np.atleast_2d(state).astype(np.float32)

        mean, std = self(state)

        sampled_action = mean + std * tf.random.normal(tf.shape(mean))

        return sampled_action.numpy()[0]


class ValueNetwork(tf.keras.Model):

    LR = 0.001

    def __init__(self):

        super(ValueNetwork, self).__init__()

        self.dense1 = kl.Dense(64, activation="relu",
                               kernel_initializer="he_normal")

        self.dense2 = kl.Dense(64, activation="relu",
                               kernel_initializer="he_normal")

        self.out = kl.Dense(1, kernel_initializer="he_normal")

        self.optimizer = tf.keras.optimizers.Adam(lr=self.LR)

    def call(self, s):

        x = self.dense1(s)
        x = self.dense2(x)
        out = self.out(x)

        return out


if __name__ == "__main__":

    policy = PolicyNetwork(action_space=1)
    s = np.array([[1, 2, 3, 4]])
    out = policy(s)
    a = policy.sample_action(s)
    print(a)

