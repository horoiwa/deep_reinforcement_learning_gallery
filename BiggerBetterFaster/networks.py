import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


class BBFNetwork(tf.keras.Model):

    def __init__(self, action_space: int, N: int = 200):

        super(BBFNetwork, self).__init__()

        self.action_space = action_space

        self.N = N

        self.conv1 = kl.Conv2D(
            32, 8, strides=4, activation="relu", kernel_initializer="he_normal"
        )
        self.conv2 = kl.Conv2D(
            64, 4, strides=2, activation="relu", kernel_initializer="he_normal"
        )
        self.conv3 = kl.Conv2D(
            64, 3, strides=1, activation="relu", kernel_initializer="he_normal"
        )

        self.flatten1 = kl.Flatten()
        self.dense1 = kl.Dense(512, activation="relu", kernel_initializer="he_normal")
        self.out = kl.Dense(self.action_space * self.N, kernel_initializer="he_normal")

    # @tf.function
    def call(self, x):

        batch_size = x.shape[0]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten1(x)
        x = self.dense1(x)

        out = self.out(x)
        quantile_values = tf.reshape(out, (batch_size, self.action_space, self.N))

        return quantile_values

    def compute_z(self, x):
        pass

    def sample_action(self, x, epsilon):
        """quantilesを均等幅で取っている場合はE[Z(s, a)]は単純平均と一致"""

        if np.random.random() > epsilon:
            quantile_qvalues = self(x)
            q_means = tf.reduce_mean(quantile_qvalues, axis=2, keepdims=True)
            selected_action = tf.argmax(q_means, axis=1)
            selected_action = selected_action[0][0].numpy()
        else:
            selected_action = np.random.choice(self.action_space)

        return selected_action


class Encoder(tf.keras.Model):
    def __init__(self):
        pass


class TransitionHead(tf.keras.Model):
    def __init__(self):
        pass
