import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


class PolicyNetwork(tf.keras.Model):

    ACTION_SPACE = 1

    MAX_KL = 0.01

    def __init__(self):

        super(PolicyNetwork, self).__init__()

        self.dense1 = kl.Dense(64, activation="tanh",
                               kernel_initializer="he_normal")

        self.dense2 = kl.Dense(64, activation="tanh",
                               kernel_initializer="he_normal")

        self.out_mean = kl.Dense(self.ACTION_SPACE,
                                 kernel_initializer="he_normal")

        self.out_logstd = kl.Dense(self.ACTION_SPACE,
                                   kernel_initializer="he_normal")

    def call(self, s):

        x = self.dense1(s)
        x = self.dense2(x)
        action_mean = self.out_mean(x)
        action_logstd = self.out_logstd(x)

        return action_mean, action_logstd

    def sample_action(self, state):
        return [np.random.uniform(-1, 1)]


class ValueNetwork(tf.keras.Model):

    def __init__(self):

        super(ValueNetwork, self).__init__()

        self.dense1 = kl.Dense(64, activation="relu",
                               kernel_initializer="he_normal")

        self.dense2 = kl.Dense(64, activation="relu",
                               kernel_initializer="he_normal")

        self.out = kl.Dense(1, kernel_initializer="he_normal")

    def call(self, s):

        x = self.dense1(s)
        x = self.dense2(x)
        out = self.out(x)

        return out


if __name__ == "__main__":

    policy = PolicyNetwork()
    s = np.array([[1, 2, 3, 4]])
    out = policy(s)
    print(out)
