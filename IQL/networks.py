import tensorflow as  tf


class Qnetwork(tf.keras.Model):

    def __init__(self):
        super(Qnetwork, self).__init__()

    def call(self, x):
        return x
