import random

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


class RecurrentDuelingQNetwork(tf.keras.Model):

    def __init__(self, action_space):

        super(RecurrentDuelingQNetwork, self).__init__()

        self.action_space = action_space

        self.conv1 = kl.Conv2D(32, 8, strides=4, activation="relu",
                               kernel_initializer="he_normal")
        self.conv2 = kl.Conv2D(64, 4, strides=2, activation="relu",
                               kernel_initializer="he_normal")
        self.conv3 = kl.Conv2D(64, 3, strides=1, activation="relu",
                               kernel_initializer="he_normal")
        self.flatten1 = kl.Flatten()

        self.lstm = kl.LSTMCell(512)

        self.value = kl.Dense(1, kernel_initializer="he_normal")

        self.advantages = kl.Dense(self.action_space,
                                   kernel_initializer="he_normal")

    def call(self, x, states, prev_action):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten1(x)

        prev_action_onehot = tf.one_hot(prev_action, self.action_space)
        x = tf.concat([x, prev_action_onehot], axis=1)
        x, states = self.lstm(x, states=states)

        value = self.value(x)
        advantages = self.advantages(x)
        advantages_mean = tf.reduce_mean(advantages, axis=1, keepdims=True)
        advantages_scaled = advantages - advantages_mean

        qvalues = value + advantages_scaled

        return qvalues, states

    def sample_action(self, x, c, h, prev_action, epsilon):

        qvalues, state = self(x, states=[c, h], prev_action=[prev_action])

        if random.random() > epsilon:
            action = np.argmax(qvalues)
        else:
            action = np.random.choice(self.action_space)

        return action, state

