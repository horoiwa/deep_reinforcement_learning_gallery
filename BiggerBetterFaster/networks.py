import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


def renormalize(x):
    shape = x.shape
    x = tf.reshape(x, shape=[shape[0], -1])
    max_value = tf.reduce_max(x, axis=-1, keepdims=True)
    min_value = tf.reduce_min(x, axis=-1, keepdims=True)
    x = (x - min_value) / (max_value - min_value + 1e-5)
    x = tf.reshape(x, shape=shape)
    return x


class BBFNetwork(tf.keras.Model):

    def __init__(
        self,
        action_space: int,
        n_supports: int = 200,
        width_scale: int = 4,
        hidden_dim=2048,
    ):

        super(BBFNetwork, self).__init__()

        self.action_space = action_space
        self.n_supports = n_supports
        self.width_scale = width_scale
        self.hidden_dim = hidden_dim

        self.encoder = ImpalaCNN(width_scale=self.width_scale)
        self.flatten = kl.Flatten()
        self.project = kl.Dense(
            self.hidden_dim, activation=None, kernel_initializer="he_normal"
        )
        self.q_head = kl.Dense(
            self.action_space * self.n_supports, kernel_initializer="he_normal"
        )

        latent_dim: int = self.encoder.base_dims[-1] * self.width_scale
        self.transition_model = TransitionModel(
            action_space=self.action_space, latent_dim=latent_dim
        )
        self.predict = kl.Dense(
            self.hidden_dim, activation=None, kernel_initializer="he_normal"
        )

    @tf.function
    def call(self, state):
        B = state.shape[0]  # batch size
        z_t = renormalize(self.encoder(state))  # (B, 11, 11, 128)
        g = self.project(self.flatten(z_t))
        quantile_qvalues = tf.reshape(
            self.q_head(g),  # (B, N * action_space)
            shape=[B, self.action_space, -1],
        )  # (B, action_space, N)
        return quantile_qvalues, z_t, g

    def sample_action(self, x, epsilon):
        """quantilesを均等幅で取っている場合はE[Z(s, a)]は単純平均と一致"""
        assert x.shape[0] == 1

        if np.random.random() > epsilon:
            quantile_qvalues, _, _ = self(x)
            q_means = tf.reduce_mean(quantile_qvalues, axis=2)
            selected_action = tf.argmax(q_means, axis=1)
            selected_action = selected_action[0].numpy()
        else:
            selected_action = np.random.choice(self.action_space)

        return selected_action

    def compute_quantile_values(self, states, actions=None):
        quantile_values_all, z_t, g = self.call(states)  # (B, action_space, N)
        if actions is None:
            qvalues = tf.reduce_mean(
                quantile_values_all, axis=2, keepdims=True
            )  # (B, action_space, 1)
            actions = tf.argmax(qvalues, axis=1)  # (B, 1)
        actions_onehot = tf.one_hot(
            actions, self.action_space, axis=1
        )  # (B, action_space, 1)

        quantile_values = tf.reduce_sum(
            quantile_values_all * actions_onehot, axis=1, keepdims=False
        )  # (B, N)
        return quantile_values, z_t, g

    @tf.function
    def compute_prediction(self, z_t, actions):
        z_t_plus_k = self.transition_model(z_t, actions)
        g = self.project(self.flatten(z_t_plus_k))
        q = self.predict(g)
        return q


class ImpalaCNN(tf.keras.Model):
    def __init__(self, width_scale: int):
        super(ImpalaCNN, self).__init__()
        self.base_dims = (16, 32, 32)
        self.width_scale = width_scale
        self.resblock_1 = ResidualBlock(dims=self.base_dims[0] * self.width_scale)
        self.resblock_2 = ResidualBlock(dims=self.base_dims[1] * self.width_scale)
        self.resblock_3 = ResidualBlock(dims=self.base_dims[2] * self.width_scale)

    def call(self, x):
        x = self.resblock_1(x)
        x = self.resblock_2(x)
        x = self.resblock_3(x)
        x = tf.nn.relu(x)
        return x


class ResidualBlock(tf.keras.Model):
    def __init__(self, dims: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = kl.Conv2D(
            dims,
            kernel_size=3,
            strides=1,
            kernel_initializer="he_normal",
            padding="same",
            activation=None,
        )
        self.conv2 = kl.Conv2D(
            dims,
            kernel_size=3,
            strides=1,
            kernel_initializer="he_normal",
            padding="same",
            activation=None,
        )
        self.conv3 = kl.Conv2D(
            dims,
            kernel_size=3,
            strides=1,
            kernel_initializer="he_normal",
            padding="same",
        )

        self.conv4 = kl.Conv2D(
            dims,
            kernel_size=3,
            strides=1,
            kernel_initializer="he_normal",
            padding="same",
            activation=None,
        )
        self.conv5 = kl.Conv2D(
            dims,
            kernel_size=3,
            strides=1,
            kernel_initializer="he_normal",
            padding="same",
        )

    def call(self, x_init):
        x = self.conv1(x_init)  # -> (B, 84, 84, 64)
        x = tf.nn.max_pool(x, ksize=3, strides=2, padding="SAME")  # -> (B, 42, 42, 64)

        block_input = x
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x += block_input

        block_input = x
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x += block_input

        return x


class TransitionModel(tf.keras.Model):

    def __init__(self, action_space: int, latent_dim: int):

        super(TransitionModel, self).__init__()
        self.action_space = action_space
        self.latent_dim = latent_dim
        self.transition_cell = TransitionCell(
            action_space=self.action_space, latent_dim=self.latent_dim
        )

    def call(self, z_t, actions):
        T = actions.shape[-1]
        for i in range(T):
            z_t = self.transition_cell(z_t, action=actions[:, i : i + 1])
        return z_t


class TransitionCell(tf.keras.Model):

    def __init__(self, action_space: int, latent_dim: int):
        super(TransitionCell, self).__init__()
        self.action_space = action_space
        self.latent_dim = latent_dim

        self.conv1 = kl.Conv2D(
            self.latent_dim,
            kernel_size=3,
            strides=1,
            kernel_initializer="he_normal",
            padding="same",
            activation="relu",
        )
        self.conv2 = kl.Conv2D(
            self.latent_dim,
            kernel_size=3,
            strides=1,
            kernel_initializer="he_normal",
            padding="same",
            activation="relu",
        )

    def call(self, z_t, action):
        # MuZeroスタイルの遷移
        B, H, W, C = z_t.shape

        action_onehot = tf.one_hot(action, depth=self.action_space)  # (B, action_space)
        action_onehot = tf.broadcast_to(
            tf.reshape(action_onehot, (B, 1, 1, self.action_space)),
            (B, H, W, self.action_space),
        )

        x = tf.concat([z_t, action_onehot], axis=-1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = renormalize(x)
        return x
