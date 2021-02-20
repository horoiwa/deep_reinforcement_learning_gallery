import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


class SamplingMixin:

    def sample_action(self, x, epsilon):
        if np.random.random() > epsilon:
            selected_actions, _ = self.sample_actions(x)
            selected_action = selected_actions.numpy()[0]
        else:
            selected_action = np.random.choice(self.action_space)

        return selected_action

    def sample_actions(self, x):
        qvalues = self(x)
        selected_actions = tf.cast(tf.argmax(qvalues, axis=1), tf.int32)
        return selected_actions, qvalues


class DuelingQNetwork(tf.keras.Model, SamplingMixin):

    def __init__(self, action_space):

        super(DuelingQNetwork, self).__init__()

        self.action_space = action_space

        self.conv1 = kl.Conv2D(32, 8, strides=4, activation="relu",
                               kernel_initializer="he_normal")
        self.conv2 = kl.Conv2D(64, 4, strides=2, activation="relu",
                               kernel_initializer="he_normal")
        self.conv3 = kl.Conv2D(64, 3, strides=1, activation="relu",
                               kernel_initializer="he_normal")
        self.flatten1 = kl.Flatten()

        self.dense1 = kl.Dense(512, activation="relu",
                               kernel_initializer="he_normal")

        self.value = kl.Dense(1, kernel_initializer="he_normal")

        self.dense2 = kl.Dense(512, activation="relu",
                               kernel_initializer="he_normal")

        self.advantages = kl.Dense(self.action_space,
                                   kernel_initializer="he_normal")

    def call(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten1(x)

        x1 = self.dense1(x)
        value = self.value(x1)

        x2 = self.dense2(x)
        advantages = self.advantages(x2)

        advantages_scaled = advantages - tf.math.reduce_mean(advantages)
        q_values = value + advantages_scaled

        return q_values

