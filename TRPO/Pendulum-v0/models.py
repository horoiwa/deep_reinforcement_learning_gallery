import numpy as np
import tensorflow as tf


class PolicyNetwork(tf.keras.Model):

    def __init__(self):

        super(PolicyNetwork, self).__init__()

    def sample_action(self):
        return np.random.uniform(-1, 1)


class ValueNetwork(tf.keras.Model):

    def __init__(self):

        super(ValueNetwork, self).__init__()
