import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow_probability as tfp


class PolicyNetwork(tf.keras.Model):

    def __init__(self, action_space):
        """
          Note: 出力層のactivationにtanhとかを使うのはKLが意味をなさなくなるのでNG
        """

        super(PolicyNetwork, self).__init__()

        self.action_space = action_space

        self.dense1 = kl.Dense(64, activation="tanh")

        self.dense2 = kl.Dense(64, activation="tanh")

        self.pi_mean = kl.Dense(self.action_space)

        self.pi_logstdev = kl.Dense(self.action_space)

    @tf.function
    def call(self, s):

        x = self.dense1(s)
        x = self.dense2(x)
        mean = self.pi_mean(x)
        logstdev = self.pi_logstdev(x)
        stdev = tf.exp(logstdev)

        return mean, stdev

    def sample_action(self, state):

        state = np.atleast_2d(state).astype(np.float32)

        mean, stdev = self(state)

        sampled_action = mean + stdev * tf.random.normal(tf.shape(mean))

        return sampled_action.numpy()[0]


class ValueNetwork(tf.keras.Model):

    LR = 0.005

    def __init__(self):

        super(ValueNetwork, self).__init__()

        self.dense1 = kl.Dense(64, activation="relu")

        self.dense2 = kl.Dense(64, activation="relu")

        self.out = kl.Dense(1)

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

