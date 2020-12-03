import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow_probability as tfp


class Actor(tf.keras.Model):

    def __init__(self, action_space, action_scale, lr=3e-4):

        super(Actor, self).__init__()

        self.action_space = 1

        self.action_scale = 2

        self.dense1 = kl.Dense(256, activation="relu")

        self.dense2 = kl.Dense(256, activation="relu")

        self.mu = kl.Dense(action_space)

        self.logstd = kl.Dense(action_space)

        self.optimizer = tf.keras.optimizers.Adam(lr=lr)

    def call(self, x):

        x = self.dense1(x)

        x = self.dense2(x)

        mu = self.mu(x)

        logstd = self.logstd(x)

        return mu, logstd

    def sample_action(self, state):

        state = np.atleast_2d(state).astype(np.float32)

        mu, logstd = self(state)

        std = np.exp(logstd)

        dist = tfp.distributions.Normal(loc=mu, scale=std)

        sampled_action = tf.tanh(dist.sample()) * self.action_scale

        return sampled_action.numpy()[0]

    def logprob(self, states, actions, noize):

        mu, logstd = self(states)

        return None


class Critic(tf.keras.Model):

    def __init__(self, lr=3e-4):

        super(Critic, self).__init__()

        self.dense_1 = kl.Dense(256, activation="relu")

        self.dense_2 = kl.Dense(256, activation="relu")

        self.out = kl.Dense(1)

        self.optimizer = tf.keras.optimizers.Adam(lr=lr)

    def call(self, observations, actions):

        x = tf.concat([observations, actions], 1)

        x = self.dense_1(x)

        x = self.dense_2(x)

        q = self.out(x)

        return q


if __name__ == "__main__":

    actor = Actor(action_space=1, action_scale=2)
    critic = Critic()

    import gym
    env = gym.make("Pendulum-v0")
    state = env.reset()
    action = actor.sample_action(state)
    print(action)
