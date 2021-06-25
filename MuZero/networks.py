import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu


class RepresentationNetwork(tf.keras.Model):

    def __init__(self, action_space):
        super(RepresentationNetwork, self).__init__()
        self.action_space = action_space

        self.conv1 = kl.Conv2D(128, kernel_size=3, strides=2,
                               padding="same", activation="relu",
                               use_bias=False, kernel_regularizer=l2(0.001),
                               kernel_initializer="he_normal")

        self.resblock1 = ResidualBlock(filters=128)
        self.resblock2 = ResidualBlock(filters=128)

        self.conv2 = kl.Conv2D(256, kernel_size=3, strides=2,
                               padding="same", activation="relu",
                               use_bias=False, kernel_regularizer=l2(0.001),
                               kernel_initializer="he_normal")

        self.resblock3 = ResidualBlock(filters=256)
        self.resblock4 = ResidualBlock(filters=256)
        self.resblock5 = ResidualBlock(filters=256)

        self.pool1 = kl.AveragePooling2D(pool_size=3, strides=2, padding="same")

        self.resblock6 = ResidualBlock(filters=256)
        self.resblock7 = ResidualBlock(filters=256)
        self.resblock8 = ResidualBlock(filters=256)

        self.pool2 = kl.AveragePooling2D(pool_size=3, strides=2, padding="same")

    def call(self, observations, training=False):
        """
            observations: <batch_size, 96, 96, 3*n_frames + n_frames> for atari.
              first (3 * frames) planes are for RGB * n_frames
              and (+ frames) planes are for hitorical actions
        """
        x = self.conv1(observations)
        x = self.resblock1(x, training=training)
        x = self.resblock2(x, training=training)

        x = self.conv2(x)
        x = self.resblock3(x, training=training)
        x = self.resblock4(x, training=training)
        x = self.resblock5(x, training=training)

        x = self.pool1(x)
        x = self.resblock6(x, training=training)
        x = self.resblock7(x, training=training)
        x = self.resblock8(x, training=training)

        states = self.pool2(x)

        return states

    def predict(self, frame_history: list, action_history: list):

        (h, w), L = frame_history[0][..., 0].shape, len(frame_history)

        frames = np.concatenate(frame_history, axis=2)

        actions = np.ones((h, w, L), dtype=np.float32)
        action_history = np.array(action_history, dtype=np.float32)
        actions = actions * action_history / (self.action_space - 1)

        observations = np.concatenate([frames, actions], axis=2)
        observations = observations[np.newaxis, ...]

        states = self(observations)

        return states


class PVNetwork(tf.keras.Model):

    def __init__(self, action_space):
        super(DynamicsNetwork, self).__init__()

        self.action_space = action_space
        self.n_supports = 600

        self.conv_p = kl.Conv2D(2, kernel_size=1, strides=1,
                                padding="same", activation="relu",
                                use_bias=False, kernel_regularizer=l2(0.001),
                                kernel_initializer="he_normal")
        self.bn_p = kl.BatchNormalization()
        self.flat_p = kl.Flatten()
        self.logits_policy = kl.Dense(action_space,
                                      kernel_regularizer=l2(0.001),
                                      kernel_initializer="he_normal")

        self.conv_v = kl.Conv2D(1, kernel_size=1, strides=1,
                                padding="same", activation="relu",
                                use_bias=False, kernel_regularizer=l2(0.001),
                                kernel_initializer="he_normal")
        self.bn_v = kl.BatchNormalization()
        self.flat_v = kl.Flatten()
        self.dense_v = kl.Dense(256, activation="relu",
                                kernel_regularizer=l2(0.001),
                                kernel_initializer="he_normal")
        self.logits_value = kl.Dense(self.n_supports,
                                     kernel_regularizer=l2(0.001),
                                     kernel_initializer="he_normal")

    def call(self, state, training=False):
        x1 = relu(self.bn_p(self.conv_p(state)))
        x1 = self.flat_p(x1)
        logits_policy = self.logits_policy(x1)
        policy = tf.nn.softmax(logits_policy)

        x2 = relu(self.bn_v(self.conv_v(state)))
        x2 = self.flat_v(x2)
        x2 = self.dense_v(x2)
        logits_value = self.logits_value(x2)
        value = tf.nn.softmax(logits_value)

        return policy, value


class DynamicsNetwork(tf.keras.Model):

    def __init__(self, action_space):
        super(DynamicsNetwork, self).__init__()

    def call(self, states, traning=False):
        pass

    def predict(self, states, actions):
        pass


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, filters):
        super(ResidualBlock, self).__init__()

        self.conv1 = kl.Conv2D(filters, kernel_size=3, padding="same",
                               use_bias=False, kernel_regularizer=l2(0.001),
                               kernel_initializer="he_normal")
        self.bn1 = kl.BatchNormalization()
        self.conv2 = kl.Conv2D(filters, kernel_size=3, padding="same",
                               use_bias=False, kernel_regularizer=l2(0.001),
                               kernel_initializer="he_normal")
        self.bn2 = kl.BatchNormalization()

    def call(self, x, training=False):

        inputs = x

        x = relu(self.bn1(self.conv1(x), training=training))

        x = self.bn2(self.conv2(x), training=training)
        x = x + inputs  #: skip connection
        x = relu(x)

        return x


if __name__ == '__main__':
    import time
    import gym

    import util

    n_frames = 8
    env_name = "BreakoutDeterministic-v4"
    f = util.get_preprocess_func(env_name)

    env = gym.make(env_name)
    action_space = env.action_space.n

    frame = f(env.reset())

    frame_history = [frame] * n_frames
    action_history = [0, 1, 2, 3, 0, 1, 2, 3]

    repr_function = RepresentationNetwork(action_space=action_space)

    t = time.time()
    for _ in range(10):
        repr_function.predict(frame_history, action_history)
    print(time.time() - t)
