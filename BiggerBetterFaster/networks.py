import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


class BBFNetwork(tf.keras.Model):

    def __init__(
        self,
        action_space: int,
        width_scale: int = 4,
        hidden_dim=2048,
        renormalize: bool = True,
    ):

        super(BBFNetwork, self).__init__()

        self.action_space = action_space

        self.encoder = ImpalaCNN(width_scale=width_scale)
        self.flatten1 = kl.Flatten()
        self.projecter = Projecter(hidden_dim=hidden_dim)
        self.predictor = kl.Dense(
            hidden_dim, activation="relu", kernel_initializer="he_normal"
        )
        self.head = kl.Dense(self.action_space * 200, kernel_initializer="he_normal")

    # @tf.function
    def call(self, x):

        batch_size = x.shape[0]

        x = self.encoder(x)
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


class ImpalaCNN(tf.keras.Model):
    def __init__(self, width_scale: int):
        super(ImpalaCNN, self).__init__()
        self.width_scale = width_scale
        self.resblock_1 = ResidualBlock(dims=16 * self.width_scale)
        self.resblock_2 = ResidualBlock(dims=32 * self.width_scale)
        self.resblock_3 = ResidualBlock(dims=32 * self.width_scale)

    def call(self, x):
        x = self.resblock_1(x)
        x = self.resblock_2(x)
        x = self.resblock_3(x)
        import pdb; pdb.set_trace()  # fmt: skip
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


class Projecter(tf.keras.Model):
    def __init__(self, hidden_dim: int):
        super(Projecter, self).__init__()
