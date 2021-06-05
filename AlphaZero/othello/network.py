import tensorflow as tf


class AlphaZeroNetwork(tf.keras.Model):
    """ For 6 * 6 othello """

    def __init__(self):
        super(AlphaZeroNetwork, self).__init__()

    def call(self, x):
        return x


class ResBlock(tf.keras.layers.Layer):

    def __init__(self):
        super(ResBlock, self).__init__()

    def call(self, x):
        return x
