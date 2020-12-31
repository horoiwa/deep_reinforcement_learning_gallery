import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


class CategoricalQNet(tf.keras.Model):

    def __init__(self, actions_space, n_atoms, Vmin, Vmax):

        super(CategoricalQNet, self).__init__()

        self.action_space = actions_space

        self.Vmin, self.Vmax = Vmin, Vmax

        self.n_atoms = n_atoms
        self.atom_weights = np.linspace(self.Vmin, self.Vmax, self.n_atoms)
        self.delta_z = (self.Vmax - self.Vmin) / self.n_atoms

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
            probs = self(x)[0]
            q_means = tf.reduce_sum(probs * self.atom_weights, axis=1, keepdims=True)
            selected_action = tf.argmax(q_means).numpy()[0]
        else:
            selected_action = np.random.choice(self.action_space)

        return selected_action
