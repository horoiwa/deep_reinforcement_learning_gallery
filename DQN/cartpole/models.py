import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as kl


class QNetwork(tf.keras.Model):

    def __init__(self, action_space):

        super(QNetwork, self).__init__()

        self.dense1 = kl.Dense(64, activation="relu", name="dense1")

        self.dense2 = kl.Dense(64, activation="relu", name="dense2")

        self.out = kl.Dense(action_space, activation="relu", name="output")

        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)

    @tf.function
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.out(x)
        return out

    def predict(self, state):
        state = np.atleast_2d(state).astype(np.float32)
        return self(state).numpy

    def copy_from(self, src_model):

        target_variables = self.trainable_variables
        src_variables = src_model.trainable_variables

        for var1, var2 in zip(target_variables, src_variables):
            var1.assign(var2.numpy())


if __name__ == "__main__":
    state = [-0.10430691, -1.55866031, 0.19466207, 2.51363456]
    qnet = QNetwork(action_space=2)
    pred = qnet.predict(state)
    print(pred)
