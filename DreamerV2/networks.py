import random

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow_probability as tfp


class Encoder(tf.keras.Model):

    def __init__(self):

        super(Encoder, self).__init__()

        self.conv1 = kl.Conv2D(2 ** 0 * 48, 4, strides=2, activation="elu")
        self.conv2 = kl.Conv2D(2 ** 1 * 48, 4, strides=2, activation="elu")
        self.conv3 = kl.Conv2D(2 ** 2 * 48, 4, strides=2, activation="elu")
        self.conv4 = kl.Conv2D(2 ** 3 * 48, 4, strides=2, activation="elu")

    def call(self, x):
        """ x: (None, 64, 64, 1) -> (None, 2, 2, 384) -> out: (None, 1536)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        out = tf.reshape(x, [1, -1])
        return out


class PolicyNetwork(tf.keras.Model):
    """ Policy network for Discrete action space
        predicts the logits of a categorical distribution
    """

    def __init__(self, action_space):

        super(PolicyNetwork, self).__init__()

        self.action_space = action_space

        self.logits = kl.Dense(self.action_space)

    def call(self, state):

        logits = self.logits(state)

        return logits

    def sample(self, state, epsilon=0.):

        if random.random() > epsilon:
            logits = self.call(state)
            action_probs = tf.nn.softmax(logits)
            cdist = tfp.distributions.Categorical(probs=action_probs)
            action = cdist.sample()
        else:
            #: random action
            action = random.randint(0, self.action_space-1)

        return action


class ValueNetwork(tf.keras.Model):

    def __init__(self):
        super(ValueNetwork, self).__init__()

    def call(self):
        pass


if __name__ == '__main__':
    import gym
    from PIL import Image
    import util

    envname = "BreakoutDeterministic-v4"
    env = gym.make(envname)
    preprocess_func = util.get_preprocess_func(envname)
    obs = preprocess_func(env.reset())
    print(obs.shape)
    obs = obs[np.newaxis, ...]
    print(obs.shape)

    encoder = Encoder()
    s = encoder(obs)
    print(s.shape)
