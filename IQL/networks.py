import tensorflow as  tf
import tensorflow.keras.layers as kl
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


class QNetwork(tf.keras.Model):

    def __init__(self):
        super(QNetwork, self).__init__()

        self.dense1 = kl.Dense(256, activation="relu")
        self.dense2 = kl.Dense(256, activation="relu")
        self.q = kl.Dense(1)

    def call(self, states, actions):
        x = tf.concat([states, actions], 1)
        x = self.dense1(x)
        x = self.dense2(x)
        q = self.q(x)
        return q


class DualQNetwork(tf.keras.Model):

    def __init__(self):
        super(DualQNetwork, self).__init__()
        self.qnet1 = QNetwork()
        self.qnet2 = QNetwork()

    @tf.function
    def call(self, states, actions):
        q1 = self.qnet1(states, actions)
        q2 = self.qnet2(states, actions)
        return q1, q2



class ValueNetwork(tf.keras.Model):

    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.dense1 = kl.Dense(256, activation="relu")
        self.dense2 = kl.Dense(256, activation="relu")
        self.v = kl.Dense(1)

    @tf.function
    def call(self, states):
        x = self.dense1(states)
        x = self.dense2(x)
        v = self.v(x)
        return v


class GaussianPolicy(tf.keras.Model):
    def __init__(self, action_space: int):
        super(GaussianPolicy, self).__init__()
        self.action_space = action_space

        self.dense1 = kl.Dense(256, activation="relu")
        self.dense2 = kl.Dense(256, activation="relu")
        self.mu = kl.Dense(self.action_space, activation="tanh")
        self.log_sigma= kl.Dense(self.action_space)

    @tf.function
    def call(self, states):
        x = self.dense1(states)
        x = self.dense2(x)
        mu = self.mu(x)
        sigma = tf.math.exp(tf.clip_by_value(self.log_sigma(x), -2.0, 10.0))

        dist = tfd.Independent(
            tfd.Normal(loc=mu, scale=sigma),
            reinterpreted_batch_ndims=1,
        )
        return dist

    def sample_actions(self, states):

        dist = self.call(states)
        action = dist.sample()
        return action
