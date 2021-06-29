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
        #self.resblock4 = ResidualBlock(filters=256)
        #self.resblock5 = ResidualBlock(filters=256)

        self.pool1 = kl.AveragePooling2D(pool_size=3, strides=2, padding="same")

        self.resblock6 = ResidualBlock(filters=256)
        #self.resblock7 = ResidualBlock(filters=256)
        #self.resblock8 = ResidualBlock(filters=256)

        self.pool2 = kl.AveragePooling2D(pool_size=3, strides=2, padding="same")

    @tf.function
    def call(self, x, training=False):
        """
            x: <batch_size, 96, 96, 3*n_frames + n_frames> for atari.
              first (3 * frames) planes are for RGB * n_frames
              and (+ frames) planes are for hitorical actions
        """
        x = self.conv1(x)
        x = self.resblock1(x, training=training)
        x = self.resblock2(x, training=training)

        x = self.conv2(x)
        x = self.resblock3(x, training=training)
        #x = self.resblock4(x, training=training)
        #x = self.resblock5(x, training=training)

        x = self.pool1(x)
        x = self.resblock6(x, training=training)
        #x = self.resblock7(x, training=training)
        #x = self.resblock8(x, training=training)

        encoded_states = self.pool2(x)

        return encoded_states

    def predict(self, frame_history: list, action_history: list):
        """
        Utility for encoding K step of frames and actions to hidden state

        Args:
            frame_history (list): list of Grayscale image
            action_history (list[int]): list of actions
        """

        (h, w), L = frame_history[0].shape, len(frame_history)

        frames = np.stack(frame_history, axis=2)

        actions = np.ones((h, w, L), dtype=np.float32)
        action_history = np.array(action_history, dtype=np.float32)
        actions = actions * action_history / (self.action_space - 1)

        observation = np.concatenate([frames, actions], axis=2)
        observation = observation[np.newaxis, ...]

        latent_state = self(observation)

        return latent_state


class PVNetwork(tf.keras.Model):

    def __init__(self, action_space: int, V_min: int, V_max: int):

        super(PVNetwork, self).__init__()

        self.action_space = action_space

        self.V_min, self.V_max = V_min, V_max

        self.n_supports = V_max - V_min + 1

        self.supports = tf.range(V_min, V_max+1, dtype=tf.float32)

        self.conv_p = kl.Conv2D(2, kernel_size=1, strides=1, padding="valid",
                                use_bias=False, kernel_regularizer=l2(0.001),
                                kernel_initializer="he_normal")
        self.bn_p = kl.BatchNormalization()
        self.flat_p = kl.Flatten()
        self.logits_policy = kl.Dense(action_space,
                                      kernel_regularizer=l2(0.001),
                                      kernel_initializer="he_normal")

        self.conv_v = kl.Conv2D(1, kernel_size=1, strides=1, padding="valid",
                                use_bias=False, kernel_regularizer=l2(0.001),
                                kernel_initializer="he_normal")
        self.bn_v = kl.BatchNormalization()
        self.flat_v = kl.Flatten()

        self.logits_value = kl.Dense(self.n_supports,
                                     kernel_regularizer=l2(0.001),
                                     kernel_initializer="he_normal")

    def call(self, x, training=False):
        x1 = relu(self.bn_p(self.conv_p(x), training=training))
        x1 = self.flat_p(x1)
        logits_policy = self.logits_policy(x1)
        policy = tf.nn.softmax(logits_policy)

        x2 = relu(self.bn_v(self.conv_v(x), training=training))
        x2 = self.flat_v(x2)
        logits_value = self.logits_value(x2)
        value = tf.nn.softmax(logits_value)

        return policy, value

    def predict(self, latent_state):
        policy, value_dist = self(latent_state)

        policy = policy.numpy()[0]
        value = tf.reduce_mean(value_dist * self.supports).numpy()

        return policy, value


class DynamicsNetwork(tf.keras.Model):

    def __init__(self, action_space: int, V_min: int, V_max: int):

        super(DynamicsNetwork, self).__init__()

        self.action_space = action_space

        self.V_min, self.V_max = V_min, V_max

        self.n_supports = V_max - V_min + 1

        self.supports = tf.range(V_min, V_max+1, dtype=tf.float32)

        self.conv1 = kl.Conv2D(256, kernel_size=3, strides=1,
                               padding="same", activation="relu",
                               use_bias=False, kernel_regularizer=l2(0.001),
                               kernel_initializer="he_normal")

        self.resblock1 = ResidualBlock(filters=256)
        self.resblock2 = ResidualBlock(filters=256)
        self.resblock3 = ResidualBlock(filters=256)
        self.resblock4 = ResidualBlock(filters=256)
        #self.resblock5 = ResidualBlock(filters=256)
        #self.resblock6 = ResidualBlock(filters=256)
        #self.resblock7 = ResidualBlock(filters=256)
        #self.resblock9 = ResidualBlock(filters=256)

        self.conv2 = kl.Conv2D(1, kernel_size=1, strides=1, padding="same",
                               use_bias=False, kernel_regularizer=l2(0.001),
                               kernel_initializer="he_normal")
        self.bn1 = kl.BatchNormalization()
        self.flat = kl.Flatten()
        self.logits = kl.Dense(self.n_supports,
                               kernel_regularizer=l2(0.001),
                               kernel_initializer="he_normal")

    def call(self, x, training=False):

        x = self.conv1(x)

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)

        #x = self.resblock5(x)
        #x = self.resblock6(x)
        #x = self.resblock7(x)
        #x = self.resblock8(x)

        next_states = x

        x = relu(self.bn1(self.conv2(x), training=training))
        x = self.flat(x)
        logits = self.logits(x)
        rewards = tf.nn.softmax(logits)

        return next_states, rewards

    def predict(self, state, action: int):

        assert len(state.shape) == 4 and state.shape[0] == 1
        assert action in range(self.action_space)

        action_onehot = np.zeros(
            state.shape[:3]+(self.action_space,), dtype=np.float32)
        action_onehot[..., action] += 1.0

        state = tf.concat([state, action_onehot], axis=3)
        next_state, reward_dist = self(state)
        reward = tf.reduce_mean(reward_dist * self.supports)

        return next_state, reward

    def predict_all(self, state):
        """
        Utility function for predicting transition of all actions

        Args:
            state: shape == (1, 6, 6, 256)
        """
        assert len(state.shape) == 4 and state.shape[0] == 1

        states = []

        for i in range(self.action_space):
            #: (1, 6, 6, action_space)
            action_onehot = np.zeros(
                state.shape[:3]+(self.action_space,), dtype=np.float32)
            action_onehot[..., i] += 1.0

            #: (1, 6, 6, 256 + action_space)
            states.append(tf.concat([state, action_onehot], axis=3))

        #: create batch
        states = tf.concat(states, axis=0)
        next_states, rewards_dist = self(states)

        supports = tf.tile(
            tf.reshape(self.supports, shape=(1, -1)), (4, 1))
        rewards = tf.reduce_mean(rewards_dist * supports, axis=1)

        return next_states, rewards


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
    dynamics_function = DynamicsNetwork(action_space=action_space, n_supports=61)
    pv_network = PVNetwork(action_space=action_space, n_supports=61)

    state = repr_function.predict(frame_history, action_history)
    policy, value = pv_network(state)
    next_states, rewards = dynamics_function.predict_all(state)
