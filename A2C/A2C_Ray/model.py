import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow_probability as tfp


class PolicyWithValue(tf.keras.Model):

    def __init__(self, action_space):
        """ PolicyとValueがネットワークを共有するA3Cアーキテクチャ
        """
        super(PolicyWithValue, self).__init__()

        self.dense1 = kl.Dense(64, activation="relu")

        self.dense2_1 = kl.Dense(64, activation="relu")

        self.dense2_2 = kl.Dense(64, activation="relu")

        self.values = kl.Dense(1)

        self.logits = kl.Dense(action_space)

    @tf.function
    def call(self, x):

        x = self.dense1(x)

        x1 = self.dense2_1(x)

        logits = self.logits(x1)

        action_probs = tf.nn.softmax(logits)

        x2 = self.dense2_2(x)

        values = self.values(x2)

        return values, action_probs

    def sample_actions(self, states):

        states = np.atleast_2d(states)

        _, action_probs = self(states)

        cdist = tfp.distributions.Categorical(probs=action_probs)

        action = cdist.sample()

        return action.numpy()
