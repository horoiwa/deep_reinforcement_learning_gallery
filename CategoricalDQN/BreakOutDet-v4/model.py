import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


class CategoricalQNet(tf.keras.Model):

    def __init__(self, actions_space, n_atoms, Z):

        super(CategoricalQNet, self).__init__()

        self.action_space = actions_space

        self.n_atoms = n_atoms

        self.Z = Z  #: 各ビンのしきい値(support)

        self.conv1 = kl.Conv2D(32, 8, strides=4, activation="relu",
                               kernel_initializer="he_normal")
        self.conv2 = kl.Conv2D(64, 4, strides=2, activation="relu",
                               kernel_initializer="he_normal")
        self.conv3 = kl.Conv2D(64, 3, strides=1, activation="relu",
                               kernel_initializer="he_normal")

        self.flatten1 = kl.Flatten()
        self.dense1 = kl.Dense(512, activation="relu",
                               kernel_initializer="he_normal")
        self.logits = kl.Dense(self.action_space * self.n_atoms)

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
