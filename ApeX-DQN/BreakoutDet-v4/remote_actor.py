import collections
import pickle
import zlib

import ray
import gym
import numpy as np
import tensorflow as tf

from model import DuelingQNetwork
from buffer import LocalReplayBuffer
from util import preprocess_frame


@ray.remote(num_cpus=1)
class Actor:

    def __init__(self, pid, env_name, epsilon, alpha,
                 buffer_size, n_frames,
                 gamma, nstep, reward_clip):

        self.pid = pid

        self.env = gym.make(env_name)

        self.epsilon = epsilon

        self.gamma = gamma

        self.alpha = alpha

        self.n_frames = n_frames

        self.action_space = self.env.action_space.n

        self.frames = collections.deque(maxlen=n_frames)

        self.nstep = nstep

        self.buffer_size = buffer_size

        self.local_buffer = LocalReplayBuffer(
            reward_clip=reward_clip, gamma=gamma, nstep=nstep)

        self.local_qnet = DuelingQNetwork(action_space=self.action_space)

        self.lives = 5  #: Breakout only

        self.define_network()

    def define_network(self):

        #: hide GPU from remote actor
        tf.config.set_visible_devices([], 'GPU')

        #: define by run
        frame = preprocess_frame(self.env.reset())
        for _ in range(self.n_frames):
            self.frames.append(frame)

        state = np.stack(self.frames, axis=2)[np.newaxis, ...]
        self.local_qnet(state)

    def rollout(self, current_weights):

        self.local_qnet.set_weights(current_weights)

        state = np.stack(self.frames, axis=2)[np.newaxis, ...]

        while True:

            state = np.stack(self.frames, axis=2)[np.newaxis, ...]

            action = self.local_qnet.sample_action(state, self.epsilon)

            next_frame, reward, done, info = self.env.step(action)

            self.frames.append(preprocess_frame(next_frame))

            next_state = np.stack(self.frames, axis=2)[np.newaxis, ...]

            if self.lives != info["ale.lives"]:
                #: loss of life as episode ends
                transition = (state, action, reward, next_state, True)
                self.lives = info["ale.lives"]
            else:
                transition = (state, action, reward, next_state, done)

            self.local_buffer.push(transition)

            if done:
                self.lives = 5
                frame = preprocess_frame(self.env.reset())
                for _ in range(self.n_frames):
                    self.frames.append(frame)

            if len(self.local_buffer) == self.buffer_size:

                experiences = self.local_buffer.pull()

                states = np.vstack(
                    [exp.state for exp in experiences]).astype(np.float32)
                actions = np.vstack(
                    [exp.action for exp in experiences]).astype(np.float32)
                rewards = np.array(
                    [exp.reward for exp in experiences]).reshape(-1, 1)
                next_states = np.vstack(
                    [exp.next_state for exp in experiences]
                    ).astype(np.float32)
                dones = np.array(
                    [exp.done for exp in experiences]).reshape(-1, 1)

                next_actions, next_qvalues = self.local_qnet.sample_actions(next_states)

                next_actions_onehot = tf.one_hot(next_actions, self.action_space)

                max_next_qvalues = tf.reduce_sum(
                    next_qvalues * next_actions_onehot, axis=1, keepdims=True)

                TQ = rewards + self.gamma ** (self.nstep) * (1 - dones) * max_next_qvalues

                qvalues = self.local_qnet(states)
                actions_onehot = tf.one_hot(
                    actions.flatten().astype(np.int32), self.action_space)
                Q = tf.reduce_sum(qvalues * actions_onehot, axis=1, keepdims=True)

                priorities = ((np.abs(TQ - Q) + 0.001) ** self.alpha).flatten()

                experiences = [zlib.compress(pickle.dumps(exp)) for exp in experiences]

                return priorities, experiences, self.pid


@ray.remote(num_cpus=1)
class TestActor:

    def __init__(self, env_name, n_frames=4):

        self.env = gym.make(env_name)

        self.action_space = self.env.action_space.n

        self.epsilon = 0.05

        self.n_frames = n_frames

        self.frames = collections.deque(maxlen=n_frames)

        self.qnet = DuelingQNetwork(action_space=self.action_space)

        self.define_network()

    def define_network(self):

        #: hide GPU from remote actor
        tf.config.set_visible_devices([], 'GPU')

        #: define by run
        frame = preprocess_frame(self.env.reset())
        for _ in range(self.n_frames):
            self.frames.append(frame)

        state = np.stack(self.frames, axis=2)[np.newaxis, ...]

        self.qnet(state)

    def play(self, current_weights):

        self.qnet.set_weights(current_weights)

        episode_steps, episode_rewards = 0, 0

        state = np.stack(self.frames, axis=2)[np.newaxis, ...]

        while True:

            state = np.stack(self.frames, axis=2)[np.newaxis, ...]

            action = self.qnet.sample_action(state, epsilon=0.05)

            next_frame, reward, done, _ = self.env.step(action)

            self.frames.append(preprocess_frame(next_frame))

            episode_steps += 1

            episode_rewards += reward

            if done or (episode_steps > 1000 and episode_rewards < 10):
                frame = preprocess_frame(self.env.reset())
                for _ in range(self.n_frames):
                    self.frames.append(frame)
                break

        return episode_steps, episode_rewards
