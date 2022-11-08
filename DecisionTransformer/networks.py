import tensorflow as tf


class DecisionTransformer(tf.keras.Model):

    def __init__(self):
        super(DecisionTransformer, self).__init__()

    def call(self, x):
        return x

