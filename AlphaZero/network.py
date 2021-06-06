import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.activations import relu


class AlphaZeroNetwork(tf.keras.Model):

    def __init__(self, action_space, n_blocks=10, filters=64):
        """
            Note:
            In AlphaZero Go paper, n_blocks = 20 (or 40) and filters = 256
        """
        super(AlphaZeroNetwork, self).__init__()

        self.action_space = action_space
        self.filters = filters
        self.n_blocks = n_blocks

        self.conv1 = kl.Conv2D(filters, kernel_size=3, padding="same",
                               kernel_initializer="he_normal")
        self.bn1 = kl.BatchNormalization()

        #: residual tower
        for n in range(self.n_blocks):
            setattr(self, f"resblock{n}", ResBlock(filters=self.filters))

        #: policy head
        self.conv_p = kl.Conv2D(2, kernel_size=1,
                                kernel_initializer="he_normal")
        self.bn_p = kl.BatchNormalization()
        self.flat_p = kl.Flatten()
        self.logits = kl.Dense(action_space,
                               kernel_initializer="he_normal")

        #: value head
        self.conv_v = kl.Conv2D(1, kernel_size=1,
                                kernel_initializer="he_normal")
        self.bn_v = kl.BatchNormalization()
        self.flat_v = kl.Flatten()
        self.dense_v = kl.Dense(64, kernel_initializer="he_normal")  #: units=256 in alphazero Go paper
        self.value = kl.Dense(1, activation="tanh",
                              kernel_initializer="he_normal")

    def call(self, x, training=False):

        x = self.conv1(x)
        x = self.bn1(x)
        x = relu(x)

        for n in range(self.n_blocks):
            x = getattr(self, f"resblock{n}")(x)

        #: policy head
        x1 = self.conv_p(x)
        x1 = self.bn_p(x1)
        x1 = relu(x1)
        x1 = self.flat_p(x1)
        logits = self.logits(x1)
        policy = tf.nn.softmax(logits)

        #: value head
        x2 = self.conv_v(x)
        x2 = self.bn_v(x2)
        x2 = relu(x2)
        x2 = self.flat_v(x2)

        x2 = self.dense_v(x2)
        x2 = relu(x2)
        value = self.value(x2)

        return policy, value


class ResBlock(tf.keras.layers.Layer):

    def __init__(self, filters):
        super(ResBlock, self).__init__()

        self.conv1 = kl.Conv2D(filters, kernel_size=3, padding="same",
                               kernel_initializer="he_normal")
        self.bn1 = kl.BatchNormalization()
        self.conv2 = kl.Conv2D(filters, kernel_size=3, padding="same",
                               kernel_initializer="he_normal")
        self.bn2 = kl.BatchNormalization()
        self.relu = kl.Activation("relu")

    def call(self, x, training=False):

        inputs = x

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = relu(x)

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = x + inputs  #: skip connection
        x = relu(x)

        return x


if __name__ == "__main__":
    import othello

    state = othello.get_initial_state()
    x = othello.encode_state(state, current_player=1)
    x = x[np.newaxis, ...]
    print(x.shape)
    action_space = othello.N_ROWS * othello.N_COLS
    network = AlphaZeroNetwork(action_space=action_space, n_blocks=5, filters=64)
    network(x)
