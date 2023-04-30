import tensorflow as  tf


class Qnetwork(tf.keras.Model):

    def __init__(self):
        super(Qnetwork, self).__init__()

        self.dense1 = kl.Dense(256, activation="relu")
        self.dense2 = kl.Dense(256, activation="relu")
        self.q = kl.Dense(1)

    def call(self, states, actions):
        x = tf.concat([states, aactions], 1)
        x = self.dense1(x)
        x = self.dense2(x)
        q = self.q(x)
        return q


class DualQNetwork(tf.keras.Model):

    def __init__(self):
        super(DualQnetwork, self).__init__()
        self.qnet1 = Qnetwork()
        self.qnet2 = Qnetwork()

    def call(self, states, aactions):
        q1 = self.qnet1(states, actions)
        q2 = self.qnet2(states, actions)
        return q1, q2



class ValueNetwork(tf.keras.Model):

    def __init__(self):
        super(DualQnetwork, self).__init__()
        self.dense1 = kl.Dense(256, activation="relu")
        self.dense2 = kl.Dense(256, activation="relu")
        self.v = kl.Dense(1)

    def call(self, states):
        x = self.dense1(states)
        x = self.dense2(x)
        v = self.v(x)
        return v


class GaussianPolicy(tf.keras.Model):
    def __init__(self, action_space: int):
        super(GaussianPolicy, self).__init__()

        self.dense1 = kl.Dense(256, activation="relu")
        self.dense2 = kl.Dense(256, activation="relu")
        self.mu = kl.Dense(self.action_space, activation="tanh")
        self.log_sigma= kl.Dense(self.action_space)

    def call(self, states):
        x = self.dense1(states)
        x = self.dense2(x)
        mu = self.mu(x)
        sigma = tf.math.exp(self.log_sigma(x))
        return mu, sigma
