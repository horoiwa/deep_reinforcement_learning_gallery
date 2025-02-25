import tensorflow as tf
import tensorflow.keras.layers as kl


class PolicyValueNetwork(tf.keras.Model):
    def __init__(self):
        super(PolicyValueNetwork, self).__init__()

    def call(self, state):
        pass


class RewardModel(tf.keras.Model):
    def __init__(self):
        super(RewardModel, self).__init__()

    def call(self, state):
        pass


class TransitionModel(tf.keras.Model):
    def __init__(self):
        super(TransitionModel, self).__init__()

    def call(self, state, action):
        pass
