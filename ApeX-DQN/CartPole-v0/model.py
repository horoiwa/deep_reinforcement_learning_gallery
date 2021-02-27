import random

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


class QNetwork(tf.keras.Model):

    def __init__(self, action_space):
        super(QNetwork, self).__init__()

        self.action_space = action_space
        self.input_layer = kl.Dense(
            64, activation="relu", kernel_initializer="he_normal")
        self.hidden_layer = kl.Dense(
            64, activation="relu", kernel_initializer="he_normal")
        self.output_layer = kl.Dense(
            action_space, kernel_initializer="he_normal")

    def call(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x

    def sample_action(self, state, epsilon):

        if random.random() > epsilon:
            state = np.atleast_2d(state)
            qvalues = self(state)
            action = np.argmax(qvalues)
        else:
            action = np.random.choice(self.action_space)

        return action
