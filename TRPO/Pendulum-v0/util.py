import tensorflow as tf
import numpy as np


def main():
    pass


class NormalDist:
    def __init__(self):
        self.mean = 10
        self.std = 1

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
           + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
           + tf.reduce_sum(self.logstd, axis=-1)

if __name__ == "__main__":
    main()
