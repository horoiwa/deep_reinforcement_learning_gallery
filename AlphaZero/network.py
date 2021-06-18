import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu



class SimpleCNN(tf.keras.Model):
    """ See https://github.com/suragnair/alpha-zero-general/
    """

    def __init__(self, action_space, filters=512, use_bias=False):

        super(SimpleCNN, self).__init__()

        self.action_space = action_space
        self.filters = filters

        self.conv1 = kl.Conv2D(filters, 3, padding='same', use_bias=use_bias)
        self.bn1 = kl.BatchNormalization()

        self.conv2 = kl.Conv2D(filters, 3, padding='same', use_bias=use_bias)
        self.bn2 = kl.BatchNormalization()

        self.conv3 = kl.Conv2D(filters, 3, padding='valid', use_bias=use_bias)
        self.bn3 = kl.BatchNormalization()

        self.conv4 = kl.Conv2D(filters, 3, padding='valid', use_bias=use_bias)
        self.bn4 = kl.BatchNormalization()

        self.flat = kl.Flatten()

        self.dense5 = kl.Dense(1024, use_bias=use_bias)
        self.bn5 = kl.BatchNormalization()
        self.drop5 = kl.Dropout(0.3)

        self.dense6 = kl.Dense(512, use_bias=use_bias)
        self.bn6 = kl.BatchNormalization()
        self.drop6 = kl.Dropout(0.3)

        self.pi = kl.Dense(self.action_space, activation='softmax')

        self.value =  kl.Dense(1, activation='tanh')

    def call(self, x, training=False):

        x = relu(self.bn1(self.conv1(x), training=training))
        x = relu(self.bn2(self.conv2(x), training=training))
        x = relu(self.bn3(self.conv3(x), training=training))
        x = relu(self.bn4(self.conv4(x), training=training))

        x = self.flat(x)

        x = relu(self.bn5(self.dense5(x), training=training))
        x = self.drop5(x, training=training)

        x = relu(self.bn6(self.dense6(x), training=training))
        x = self.drop6(x, training=training)

        pi = self.pi(x)
        v = self.value(x)

        return pi, v

    def predict(self, state):
        if len(state.shape) == 3:
            state = state[np.newaxis, ...]

        policy, value = self(state)

        return policy, value


class AlphaZeroResNet(tf.keras.Model):

    def __init__(self, action_space, n_blocks=3, filters=256, use_bias=False):
        """
            Note:
            In AlphaZero Go paper, n_blocks = 20 (or 40) and filters = 256
        """
        super(AlphaZeroResNet, self).__init__()

        self.action_space = action_space
        self.filters = filters
        self.n_blocks = n_blocks

        self.conv1 = kl.Conv2D(filters, kernel_size=3, padding="same",
                               use_bias=use_bias, kernel_regularizer=l2(0.001),
                               kernel_initializer="he_normal")
        self.bn1 = kl.BatchNormalization()

        #: residual tower
        for n in range(self.n_blocks):
            setattr(self, f"resblock{n}",
                    ResBlock(filters=self.filters, use_bias=use_bias))

        #: policy head
        self.conv_p = kl.Conv2D(2, kernel_size=1,
                                use_bias=use_bias, kernel_regularizer=l2(0.001),
                                kernel_initializer="he_normal")
        self.bn_p = kl.BatchNormalization()
        self.flat_p = kl.Flatten()
        self.logits = kl.Dense(action_space,
                               kernel_regularizer=l2(0.001),
                               kernel_initializer="he_normal")

        #: value head
        self.conv_v = kl.Conv2D(1, kernel_size=1,
                                use_bias=use_bias, kernel_regularizer=l2(0.001),
                                kernel_initializer="he_normal")
        self.bn_v = kl.BatchNormalization()
        self.flat_v = kl.Flatten()
        self.value = kl.Dense(1, activation="tanh",
                              kernel_regularizer=l2(0.001),
                              kernel_initializer="he_normal")

    def call(self, x, training=False):

        x = relu(self.bn1(self.conv1(x), training=training))

        for n in range(self.n_blocks):
            x = getattr(self, f"resblock{n}")(x, training=training)

        #: policy head
        x1 = relu(self.bn_p(self.conv_p(x), training=training))
        x1 = self.flat_p(x1)
        logits = self.logits(x1)
        policy = tf.nn.softmax(logits)

        #: value head
        x2 = relu(self.bn_v(self.conv_v(x), training=training))
        x2 = self.flat_v(x2)
        value = self.value(x2)

        return policy, value

    def predict(self, state):
        if len(state.shape) == 3:
            state = state[np.newaxis, ...]

        policy, value = self(state)

        return policy, value


class ResBlock(tf.keras.layers.Layer):

    def __init__(self, filters, use_bias):
        super(ResBlock, self).__init__()

        self.conv1 = kl.Conv2D(filters, kernel_size=3, padding="same",
                               use_bias=use_bias, kernel_regularizer=l2(0.001),
                               kernel_initializer="he_normal")
        self.bn1 = kl.BatchNormalization()
        self.conv2 = kl.Conv2D(filters, kernel_size=3, padding="same",
                               use_bias=use_bias, kernel_regularizer=l2(0.001),
                               kernel_initializer="he_normal")
        self.bn2 = kl.BatchNormalization()

    def call(self, x, training=False):

        inputs = x

        x = relu(self.bn1(self.conv1(x), training=training))

        x = self.bn2(self.conv2(x), training=training)
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
    #network = AlphaZeroResNet(action_space=action_space, n_blocks=5, filters=64)
    network = SimpleCNN(action_space=action_space, filters=512)
    print(network(x))
