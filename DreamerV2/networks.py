import random

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


class WorldModel(tf.keras.Model):

    def __init__(self, config):

        super(WorldModel, self).__init__()

        self.config = config

        self.latent_dim = self.config.latent_dim

        self.n_atoms = self.config.n_atoms

        self.encoder = Encoder()

        self.decoder = Decoder()

        self.rssm = RecurrentStateSpaceModel(self.latent_dim, self.n_atoms)

        self.reward_head = MLPHead()

        self.discount_head = MLPHead()

    def call(self, obs, prev_z, prev_h, prev_a):
        """ ! Redundunt method
        """

        embed = self.encoder(obs)
        h = self.rssm.step_h(prev_z, prev_h, prev_a)
        z = self.rssm.sample_z(h, embed)
        _z = tf.reshape(z, [z.shape[0], -1])

        feat = tf.concat([_z, h], axis=-1)

        decoded = self.decoder(z)
        reward = self.reward_head(feat)
        gamma = self.discount_head(feat)

        return h, z, feat, decoded, reward, gamma

    def get_initial_state(self, batch_size):
        z_init = tf.zeros([batch_size, self.latent_dim, self.n_atoms])
        h_init = self.rssm.gru_cell.get_initial_state(
            batch_size=batch_size, dtype=tf.float32
            )
        return z_init, h_init

    def step_h(self, prev_z, prev_h, prev_a):
        """ Step deterministic latent H of reccurent state space model
        """
        h = self.rssm.step_h(prev_z, prev_h, prev_a)

        return h

    def get_feature(self, obs, h):
        embed = self.encoder(obs)
        z = self.rssm.sample_z(h, embed)
        z = tf.reshape(z, [z.shape[0], -1])
        feat = tf.concat([z, h], axis=-1)
        return feat


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
        out = tf.reshape(x, [x.shape[0], -1])
        return out


class Decoder(tf.keras.Model):

    def __init__(self):

        super(Decoder, self).__init__()

    def call(self, x):
        return x


class RecurrentStateSpaceModel(tf.keras.Model):

    def __init__(self, latent_dim, n_atoms):

        super(RecurrentStateSpaceModel, self).__init__()

        self.latent_dim, self.n_atoms = latent_dim, n_atoms

        self.units = 600

        self.dense_z1 = kl.Dense(self.units, activation="elu")

        self.dense_z2 = kl.Dense(self.latent_dim*self.n_atoms)

        self.dense_h1 = kl.Dense(self.units, activation="elu")

        self.gru_cell = kl.GRUCell(self.units, activation="tanh")

    def call(self, prev_z, prev_h, prev_a, embed):
        """ ! Redundunt method
        """
        h = self.step_h(prev_z, prev_h, prev_a)
        z = self.sample_z(h, embed)
        return z, h

    @tf.function
    def sample_z(self, h, embed):

        x = tf.concat([h, embed], axis=-1)
        x = self.dense_z1(x)
        logits = self.dense_z2(x)
        logits = tf.reshape(
            logits, [logits.shape[0], self.latent_dim, self.n_atoms]
            )

        z_probs = tf.nn.softmax(logits, axis=2)

        #: batch_shape=[batch_size] event_shape=[32, 32]
        dist = tfd.Independent(
            tfd.OneHotCategorical(probs=z_probs),
            reinterpreted_batch_ndims=1)

        z = tf.cast(dist.sample(), tf.float32)

        #: Reparameterization trick for OneHotCategorcalDist
        z = z + z_probs - tf.stop_gradient(z_probs)

        return z

    @tf.function
    def step_h(self, prev_z, prev_h, prev_a):
        prev_z = tf.reshape(prev_z, [prev_z.shape[0], -1])
        x = tf.concat([prev_z, prev_a], axis=-1)
        x = self.dense_h1(x)
        h, _ = self.gru_cell(x, [prev_h])

        return h


class MLPHead(tf.keras.Model):

    def __init__(self):

        super(Head, self).__init__()

    def call(self, x):
        return x


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
