import tensorflow as tf
import tensorflow.keras.layers as kl
import numpy as np


class ValueNetwork(tf.keras.Model):

    def __init__(self):

        super(ValueNetwork, self).__init__()

        self.dense1 = kl.Dense(32, activation="relu", name="dense1")

        self.dense2 = kl.Dense(32, activation="relu", name="dense2")

        self.out = kl.Dense(1, name="output")

        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)

    @tf.function
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.out(x)
        return out

    def predict(self, states):
        states = np.atleast_2d(states).astype(np.float32)
        return self(states).numpy()

    def compute_grads(self, states, target_values):

        with tf.GradientTape() as tape:

            estimated_values = self(states)

            loss = tf.reduce_mean(
                tf.square(target_values - estimated_values))

        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)

        return gradients


class PolicyNetwork(tf.keras.Model):

    def __init__(self, action_space):

        super(PolicyNetwork, self).__init__()


if __name__ == "__main__":
    states = np.array([[-0.10430691, -1.55866031, 0.19466207, 2.51363456],
                       [-0.10430691, -1.55866031, 0.19466207, 2.51363456],
                       [-0.10430691, -1.55866031, 0.19466207, 2.51363456]])
    states.astype(np.float32)
    actions = [0, 1, 1]

    target_values = [1, 1, 1]
    value_model = ValueModel()
