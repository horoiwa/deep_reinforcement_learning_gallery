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

        self.reward_head = MLPHead(out_shape=1)

        self.discount_head = MLPHead(out_shape=1)

    @tf.function
    def call(self, obs, prev_z, prev_h, prev_a):

        embed = self.encoder(obs)

        h = self.rssm.step_h(prev_z, prev_h, prev_a)

        z_prior, z_prior_probs = self.rssm.sample_z_prior(h)
        z_post, z_post_probs = self.rssm.sample_z_post(h, embed)
        z_post = tf.reshape(z_post, [z_post.shape[0], -1])

        feat = tf.concat([z_post, h], axis=-1)

        img_decoded = self.decoder(feat)
        reward_mean = self.reward_head(feat)
        discount_logit = self.discount_head(feat)

        return (h, z_prior, z_prior_probs, z_post, z_post_probs,
                feat, img_decoded, reward_mean, discount_logit)

    def get_initial_state(self, batch_size):
        z_init = tf.zeros([batch_size, self.latent_dim, self.n_atoms])
        z_init = tf.reshape(z_init, [z_init.shape[0], -1])
        h_init = self.rssm.gru_cell.get_initial_state(
            batch_size=batch_size, dtype=tf.float32
            )
        return z_init, h_init

    @tf.function
    def step_h(self, prev_z, prev_h, prev_a):
        """ Step deterministic latent H of reccurent state space model
        """
        h = self.rssm.step_h(prev_z, prev_h, prev_a)

        return h

    @tf.function
    def get_feature(self, obs, h):
        embed = self.encoder(obs)
        z_post, z_post_probs = self.rssm.sample_z_post(h, embed)
        z_post = tf.reshape(z_post, [z_post.shape[0], -1])
        feat = tf.concat([z_post, h], axis=-1)
        return feat, z_post


class Encoder(tf.keras.Model):

    def __init__(self):

        super(Encoder, self).__init__()

        self.conv1 = kl.Conv2D(2 ** 0 * 48, 4, strides=2, activation="elu")
        self.conv2 = kl.Conv2D(2 ** 1 * 48, 4, strides=2, activation="elu")
        self.conv3 = kl.Conv2D(2 ** 2 * 48, 4, strides=2, activation="elu")
        self.conv4 = kl.Conv2D(2 ** 3 * 48, 4, strides=2, activation="elu")

    @tf.function
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

        self.dense1 = kl.Dense(1536, activation="elu")
        self.convT1 = kl.Conv2DTranspose(192, 5, strides=2, activation="elu")
        self.convT2 = kl.Conv2DTranspose(98, 5, strides=2, activation="elu")
        self.convT3 = kl.Conv2DTranspose(48, 6, strides=2, activation="elu")
        self.convT4 = kl.Conv2DTranspose(1, 6, strides=2)

    @tf.function
    def call(self, feat):
        """
            feat: (batch_size, 1624)
            out: (batch_size, 64, 64, 1)
        """
        x = self.dense1(feat)                # (bs, 1536)
        x = tf.reshape(x, [-1, 1, 1, 1536])  # (bs, 1, 1, 1536)
        x = self.convT1(x)                   # (bs, 5, 5, 192)
        x = self.convT2(x)                   # (bs, 13, 13, 98)
        x = self.convT3(x)                   # (bs, 30, 30, 48)
        x = self.convT4(x)                   # (bs, 64, 64, 1)

        return x


class RecurrentStateSpaceModel(tf.keras.Model):

    def __init__(self, latent_dim, n_atoms):

        super(RecurrentStateSpaceModel, self).__init__()

        self.latent_dim, self.n_atoms = latent_dim, n_atoms

        self.units = 600

        self.dense_z_prior1 = kl.Dense(self.units, activation="elu")

        self.dense_z_prior2 = kl.Dense(self.latent_dim*self.n_atoms)

        self.dense_z_post1 = kl.Dense(self.units, activation="elu")

        self.dense_z_post2 = kl.Dense(self.latent_dim*self.n_atoms)

        self.dense_h1 = kl.Dense(self.units, activation="elu")

        self.gru_cell = kl.GRUCell(self.units, activation="tanh")

    def call(self, prev_z, prev_h, prev_a, embed):
        """ ! Redundunt method
        """
        h = self.step_h(prev_z, prev_h, prev_a)
        z_prior, z_prior_probs = self.sample_z_prior(h)
        z_post, z_post_probs = self.sample_z_post(h, embed)
        return h, z_post, z_post_probs, z_prior, z_prior_probs

    @tf.function
    def sample_z_prior(self, h):

        x = self.dense_z_prior1(h)
        logits = self.dense_z_prior2(x)
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

        return z, z_probs

    @tf.function
    def sample_z_post(self, h, embed):

        x = tf.concat([h, embed], axis=-1)
        x = self.dense_z_post1(x)
        logits = self.dense_z_post2(x)
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

        return z, z_probs

    @tf.function
    def step_h(self, prev_z, prev_h, prev_a):
        prev_z = tf.reshape(prev_z, [prev_z.shape[0], -1])
        x = tf.concat([prev_z, prev_a], axis=-1)
        x = self.dense_h1(x)
        x, [h] = self.gru_cell(x, [prev_h])  #: x == h

        return h


class MLPHead(tf.keras.Model):

    def __init__(self, out_shape):

        super(MLPHead, self).__init__()

        self.d1 = kl.Dense(400, activation='elu',
                           kernel_initializer="Orthogonal")
        self.d2 = kl.Dense(400, activation='elu'
                           kernel_initializer="Orthogonal")
        self.d3 = kl.Dense(400, activation='elu'
                           kernel_initializer="Orthogonal")
        #self.d4 = kl.Dense(400, activation='elu')

        self.out = kl.Dense(out_shape, kernel_initializer="Orthogonal")

    @tf.function
    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        #x = self.d4(x)
        out = self.out(x)
        return out


class PolicyNetwork(tf.keras.Model):
    """ Policy network for Discrete action space
        predicts the logits of a categorical distribution
    """

    def __init__(self, action_space):

        super(PolicyNetwork, self).__init__()

        self.action_space = action_space

        self.mlp = MLPHead(out_shape=self.action_space)

    def call(self, feat):
        logits = self.mlp(feat)
        probs = tf.nn.softmax(logits, axis=1)
        return logits, probs

    def sample_action(self, feat, epsilon=0.):

        assert feat.shape[0] == 1

        if random.random() > epsilon:
            action_onehot = self.sample(feat)
            action = np.argmax(action_onehot)
        else:
            #: random action
            action = random.randint(0, self.action_space-1)

        return action

    def sample(self, feat):
        logits, probs = self(feat)
        dist = tfd.Independent(
            tfd.OneHotCategorical(probs=probs),
            reinterpreted_batch_ndims=0)
        actions_onehot = dist.sample()
        return actions_onehot


class ValueNetwork(tf.keras.Model):
    """ Policy network for Discrete action space
        predicts the logits of a categorical distribution
    """

    def __init__(self, action_space):

        super(ValueNetwork, self).__init__()

        self.mlp = MLPHead(out_shape=1)

    @tf.function
    def call(self, feat):

        value = self.mlp(feat)

        return value


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
