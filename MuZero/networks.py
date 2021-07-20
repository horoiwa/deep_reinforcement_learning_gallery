import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu

from util import value_rescaling, inverse_value_rescaling


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
        #self.resblock5 = ResidualBlock(filters=256)

        self.pool1 = kl.AveragePooling2D(pool_size=3, strides=2, padding="same")

        self.resblock6 = ResidualBlock(filters=256)
        self.resblock7 = ResidualBlock(filters=256)
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
        x = self.resblock4(x, training=training)
        #x = self.resblock5(x, training=training)

        x = self.pool1(x)
        x = self.resblock6(x, training=training)
        x = self.resblock7(x, training=training)
        #x = self.resblock8(x, training=training)

        x = self.pool2(x)

        bs, h, w, d = x.shape

        s_min = tf.reduce_min(tf.reshape(x, (bs, -1)), axis=1, keepdims=True)
        s_max = tf.reduce_max(tf.reshape(x, (bs, -1)), axis=1, keepdims=True)

        s_min = s_min * tf.ones((bs, h*w*d), dtype=tf.float32)
        s_max = s_max * tf.ones((bs, h*w*d), dtype=tf.float32)

        s_min = tf.reshape(s_min, (bs, h, w, d))
        s_max = tf.reshape(s_max, (bs, h, w, d))

        hidden_states = (x - s_min) / (s_max - s_min)

        return hidden_states

    def predict(self, frame_history, action_history):

        observation = self.make_observation(frame_history, action_history)

        hidden_state = self(observation)

        return hidden_state, observation

    def make_observation(self, frame_history, action_history):
        """
        Utility for encoding K step of frames and actions to hidden state

        Args:
            frame_history (list): list of Grayscale image
            action_history (list[int]): list of actions
            action_space(int)
        """

        (h, w), L = frame_history[0].shape, len(frame_history)
        frames = np.stack(frame_history, axis=2)

        actions = np.ones((h, w, L), dtype=np.float32)
        action_history = np.array(action_history, dtype=np.float32)
        actions = actions * action_history / (self.action_space - 1)

        observation = np.concatenate([frames, actions], axis=2)
        observation = tf.convert_to_tensor(observation[np.newaxis, ...])

        return observation


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

        self.value = kl.Dense(1,
                              kernel_regularizer=l2(0.001),
                              kernel_initializer="he_normal")

    @tf.function
    def call(self, hidden_states, training=False):
        x1 = relu(self.bn_p(self.conv_p(hidden_states), training=training))
        x1 = self.flat_p(x1)
        logits_policy = self.logits_policy(x1)
        policy = tf.nn.softmax(logits_policy)

        x2 = relu(self.bn_v(self.conv_v(hidden_states), training=training))
        x2 = self.flat_v(x2)
        value = self.value(x2)

        return policy, value

    def predict(self, hidden_state):

        assert len(hidden_state.shape) == 4

        policy, value = self(hidden_state)

        if hidden_state.shape[0] == 1:
            policy = policy.numpy()[0]
            value = inverse_value_rescaling(value).numpy()[0][0]
        else:
            value = inverse_value_rescaling(value)

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
        self.reward = kl.Dense(1,
                               kernel_regularizer=l2(0.001),
                               kernel_initializer="he_normal")

    @tf.function
    def call(self, hidden_states, actions, training=False):
        """
        Args:
            hidden_states: (batchsize, 6, 6, 256)
            actions: (bathsize, )
        """

        bs, h, w, d = hidden_states.shape

        #: Make onehot action plane
        actions_onehot = tf.one_hot(actions, self.action_space)         #: (batchsize, action_space)
        actions_onehot = tf.repeat(
            actions_onehot, repeats=h * w, axis=1)                      #: (batchsize, action_space * 6 * 6)
        actions_onehot = tf.reshape(
            actions_onehot, (bs, self.action_space, h * w))             #: (batchsize, action_space, 6 * 6)
        actions_onehot = tf.reshape(
            actions_onehot, (bs, self.action_space, h,  w))             #: (batchsize, action_space, 6, 6)
        actions_onehot = tf.transpose(
            actions_onehot, perm=[0, 2, 3, 1])                          #: (batchsize, 6, 6, action_space)

        #: Concat action plane with hidden state
        x = tf.concat([hidden_states, actions_onehot], axis=3)          #: (batchsize, 6, 6, 256+action_space)

        x = self.conv1(x)
        x = self.resblock1(x, training=training)
        x = self.resblock2(x, training=training)
        x = self.resblock3(x, training=training)
        x = self.resblock4(x, training=training)

        #x = self.resblock5(x, training=training)
        #x = self.resblock6(x, training=training)
        #x = self.resblock7(x, training=training)
        #x = self.resblock8(x, training=training)

        s_min = tf.reduce_min(tf.reshape(x, (bs, -1)), axis=1, keepdims=True)
        s_max = tf.reduce_max(tf.reshape(x, (bs, -1)), axis=1, keepdims=True)

        s_min = s_min * tf.ones((bs, h*w*d), dtype=tf.float32)
        s_max = s_max * tf.ones((bs, h*w*d), dtype=tf.float32)

        s_min = tf.reshape(s_min, (bs, h, w, d))
        s_max = tf.reshape(s_max, (bs, h, w, d))

        next_hidden_states = (x - s_min) / (s_max - s_min)

        x = relu(self.bn1(self.conv2(x), training=training))
        x = self.flat(x)
        rewards = self.reward(x)

        return next_hidden_states, rewards

    def predict(self, hidden_state, action: int):

        assert len(hidden_state.shape) == 4 and hidden_state.shape[0] == 1
        assert action in range(self.action_space)

        next_state, reward = self(hidden_state, tf.convert_to_tensor([action]))

        return next_state, reward

    def predict_all(self, hidden_state):
        """
        Utility function for predicting dynamics of all actions

        Args:
            state: shape == (1, 6, 6, 256)
        """
        assert len(hidden_state.shape) == 4 and hidden_state.shape[0] == 1

        actions = tf.range(self.action_space)
        hidden_states = tf.repeat(hidden_state, repeats=4, axis=0)
        next_states, rewards = self(hidden_states, actions)

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
    dynamics_function = DynamicsNetwork(action_space=action_space, V_min=-30, V_max=30)
    pv_network = PVNetwork(action_space=action_space, V_min=-30, V_max=30)

    hidden_state, obs = repr_function.predict(frame_history, action_history)
    hidden_states = tf.repeat(hidden_state, repeats=4, axis=0)

    policy, value = pv_network.predict(hidden_state)

    action = tf.convert_to_tensor([1])
    #next_states, rewards = dynamics_function(hidden_state, action)
    #next_states, rewards = dynamics_function.predict(hidden_state, 0)
    next_states, rewards = dynamics_function.predict_all(hidden_state)
