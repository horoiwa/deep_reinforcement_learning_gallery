import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow_probability as tfp


def create_network(action_space, use_dueling, use_noisy, use_categorical):
    if use_dueling and use_noisy and use_categorical:
        pass
    elif use_noisy:
        return NoisyQNetwork(action_space)
    elif use_dueling:
        return DuelingQNetwork(action_space)
    else:
        return QNetwork(action_space)


class NoisyDense(tf.keras.layers.Layer):
    """ Factorized Gaussian Noisy Dense Layer
    """
    def __init__(self, units, activation=None,
                 kernel_initializer="random_normal", trainable=True):
        super(NoisyDense, self).__init__()
        self.units = units
        self.initializer = kernel_initializer
        self.trainable = trainable
        self.normal = tfp.distributions.Normal(loc=0, scale=1)
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):

        p = input_shape[0]

        self.w_mu = self.add_weight(
            name="w_mu",
            shape=(int(input_shape[-1]), self.units),
            initializer=tf.keras.initializers.RandomUniform(
                -1. / np.sqrt(p), 1. / np.sqrt(p)),
            trainable=self.trainable)

        self.w_sigma = self.add_weight(
            name="w_sigma",
            shape=(int(input_shape[-1]), self.units),
            initializer=tf.keras.initializers.Constant(0.5 / np.sqrt(p)),
            trainable=self.trainable)

        self.b_mu = self.add_weight(
            name="b_mu",
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomUniform(
                -1. / np.sqrt(p), 1. / np.sqrt(p)),
            trainable=self.trainable)

        self.b_sigma = self.add_weight(
            name="b_sigma",
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(0.5 / np.sqrt(p)),
            trainable=self.trainable)

    @tf.function
    def call(self, inputs, noise=True):

        epsilon_in = self.f(self.normal.sample((self.w_mu.shape[0], 1)))
        epsilon_out = self.f(self.normal.sample((1, self.w_mu.shape[1])))

        w_epsilon = tf.matmul(epsilon_in, epsilon_out)
        b_epsilon = epsilon_out

        w = self.w_mu + self.w_sigma * w_epsilon
        b = self.b_mu + self.b_sigma * b_epsilon

        out = tf.matmul(inputs, w) + b

        if self.activation is not None:
            out = self.activation(out)

        return out

    @staticmethod
    def f(x):
        x = tf.sign(x) * tf.sqrt(tf.abs(x))
        return x


class SamplingMixin:

    def sample_action(self, x, epsilon=0):
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


class QNetwork(tf.keras.Model, SamplingMixin):

    def __init__(self, actions_space):

        super(QNetwork, self).__init__()

        self.action_space = actions_space

        self.conv1 = kl.Conv2D(32, 8, strides=4, activation="relu",
                               kernel_initializer="he_normal")
        self.conv2 = kl.Conv2D(64, 4, strides=2, activation="relu",
                               kernel_initializer="he_normal")
        self.conv3 = kl.Conv2D(64, 3, strides=1, activation="relu",
                               kernel_initializer="he_normal")
        self.flatten1 = kl.Flatten()
        self.dense1 = kl.Dense(512, activation="relu",
                               kernel_initializer="he_normal")
        self.qvalues = kl.Dense(self.action_space,
                                kernel_initializer="he_normal")

    @tf.function
    def call(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        qvalues = self.qvalues(x)

        return qvalues


class DuelingQNetwork(tf.keras.Model, SamplingMixin):

    def __init__(self, actions_space):

        super(DuelingQNetwork, self).__init__()

        self.action_space = actions_space

        self.conv1 = kl.Conv2D(32, 8, strides=4, activation="relu",
                               kernel_initializer="he_normal")
        self.conv2 = kl.Conv2D(64, 4, strides=2, activation="relu",
                               kernel_initializer="he_normal")
        self.conv3 = kl.Conv2D(64, 3, strides=1, activation="relu",
                               kernel_initializer="he_normal")
        self.flatten1 = kl.Flatten()

        self.dense1 = kl.Dense(512, activation="relu",
                               kernel_initializer="he_normal")
        self.value = kl.Dense(1, activation="relu",
                              kernel_initializer="he_normal")

        self.dense2 = kl.Dense(512, activation="relu",
                               kernel_initializer="he_normal")

        self.advantages = kl.Dense(self.action_space, activation="relu",
                                   kernel_initializer="he_normal")

        self.qvalues = kl.Dense(self.action_space,
                                kernel_initializer="he_normal")

    @tf.function
    def call(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten1(x)

        x1 = self.dense1(x)
        value = self.value(x1)

        x2 = self.dense2(x)
        advantages = self.advantages(x2)

        scaled_advantages = advantages - tf.reduce_mean(advantages)
        q_values = value + scaled_advantages

        return q_values


class NoisyQNetwork(tf.keras.Model, SamplingMixin):

    def __init__(self, actions_space):

        super(NoisyQNetwork, self).__init__()

        self.action_space = actions_space

        self.conv1 = kl.Conv2D(32, 8, strides=4, activation="relu",
                               kernel_initializer="he_normal")
        self.conv2 = kl.Conv2D(64, 4, strides=2, activation="relu",
                               kernel_initializer="he_normal")
        self.conv3 = kl.Conv2D(64, 3, strides=1, activation="relu",
                               kernel_initializer="he_normal")
        self.flatten1 = kl.Flatten()
        self.dense1 = NoisyDense(512, activation="relu")
        self.qvalues = NoisyDense(self.action_space)

    @tf.function
    def call(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        qvalues = self.qvalues(x)

        return qvalues


class CategoricalQNet(tf.keras.Model):

    def __init__(self, actions_space, n_atoms, Z):

        super(CategoricalQNet, self).__init__()

        self.action_space = actions_space

        self.n_atoms = n_atoms

        #: 各ビンのしきい値(support)
        self.Z = Z

        self.conv1 = kl.Conv2D(32, 8, strides=4, activation="relu",
                               kernel_initializer="he_normal")
        self.conv2 = kl.Conv2D(64, 4, strides=2, activation="relu",
                               kernel_initializer="he_normal")
        self.conv3 = kl.Conv2D(64, 3, strides=1, activation="relu",
                               kernel_initializer="he_normal")

        self.flatten1 = kl.Flatten()
        self.dense1 = kl.Dense(512, activation="relu",
                               kernel_initializer="he_normal")
        self.logits = kl.Dense(self.action_space * self.n_atoms,
                               kernel_initializer="he_normal")

    @tf.function
    def call(self, x):

        batch_size = x.shape[0]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.flatten1(x)
        x = self.dense1(x)

        logits = self.logits(x)
        logits = tf.reshape(logits, (batch_size, self.action_space, self.n_atoms))
        probs = tf.nn.softmax(logits, axis=2)

        return probs

    def sample_action(self, x, epsilon=None):

        if (epsilon is None) or (np.random.random() > epsilon):
            selected_actions, _ = self.sample_actions(x)
            selected_action = selected_actions[0][0].numpy()
        else:
            selected_action = np.random.choice(self.action_space)

        return selected_action

    def sample_actions(self, x):
        probs = self(x)
        q_means = tf.reduce_sum(probs * self.Z, axis=2, keepdims=True)
        selected_actions = tf.argmax(q_means, axis=1)
        return selected_actions, probs
