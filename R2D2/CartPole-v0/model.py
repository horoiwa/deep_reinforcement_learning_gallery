import random

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


class RecurrentQNetwork(tf.keras.Model):

    def __init__(self, action_space):
        super(RecurrentQNetwork, self).__init__()

        self.action_space = action_space

        self.input_layer = kl.Dense(
            128, activation="relu", kernel_initializer="he_normal")
        self.lstm = kl.LSTMCell(128)
        self.output_layer = kl.Dense(
            action_space, kernel_initializer="he_normal")

    def call(self, x, states):
        x = self.input_layer(x)
        x, states = self.lstm(x, states=states)
        out = self.output_layer(x)
        return out, states

    def sample_action(self, x, c, h, epsilon):

        x = np.atleast_2d(x).astype(np.float32)
        qvalues, state = self(x, states=[c, h])

        if random.random() > epsilon:
            action = np.argmax(qvalues)
        else:
            action = np.random.choice(self.action_space)

        return action, state


if __name__ == "__main__":
    import gym

    burn_in = 4
    trace_length = 8

    env = gym.make("CartPole-v0")
    recurrent_qnet = RecurrentQNetwork(action_space=2)
    buffer = []

    episode_rewards = 0
    epsilon = 0.3

    state = env.reset()
    c, h = recurrent_qnet.lstm.get_initial_state(batch_size=1, dtype=tf.float32)
    for i in range(999):
        action, (c, h) = recurrent_qnet.sample_action(state, c, h, epsilon)
        next_state, reward, done, _ = env.step(action)
        episode_rewards += reward
        transition = (state, action, reward, next_state, done)
        buffer.append(transition)

        if done:
            print(episode_rewards, state, next_state)
            episode_rewards = 0
            state = env.reset()
            c, h = recurrent_qnet.lstm.get_initial_state(batch_size=1, dtype=tf.float32)
        else:
            state = next_state
