import collections
import shutil
from pathlib import Path
import pickle

import gym
import numpy as np
import ray
import tensorflow as tf
import lz4.frame as lz4f

import util
from buffer import EpisodeBuffer
from model import RecurrentDuelingQNetwork


@ray.remote(num_cpus=1)
class Actor:

    def __init__(self, pid, env_name, n_frames,
                 epsilon, gamma, eta, alpha,
                 nstep, burnin_length, unroll_length):

        self.pid = pid
        self.env_name = env_name
        self.action_space = gym.make(env_name).action_space.n
        self.frame_process_func = util.get_preprocess_func(self.env_name)
        self.n_frames = n_frames

        self.q_network = RecurrentDuelingQNetwork(self.action_space)
        self.epsilon = epsilon
        self.gamma = gamma

        self.eta = eta
        self.alpha = alpha  # priority exponent

        self.nstep = nstep
        self.burnin_len = burnin_length
        self.unroll_len = unroll_length

        self.define_network()

    def define_network(self):

        #: hide GPU from remote actor
        tf.config.set_visible_devices([], 'GPU')

        env = gym.make(self.env_name)

        frame = self.frame_process_func(env.reset())
        frames = [frame] * self.n_frames
        state = np.stack(frames, axis=2)[np.newaxis, ...]

        c, h = self.q_network.lstm.get_initial_state(batch_size=1, dtype=tf.float32)
        self.q_network(np.atleast_2d(state), states=[c, h], prev_action=[0])

    def sync_weights_and_rollout(self, current_weights):

        #: グローバルQ-Networkと重みを同期
        self.q_network.set_weights(current_weights)

        priorities, segments = [], []

        while len(segments) < 10:
            _priorities, _segments = self._rollout()
            priorities += _priorities
            segments += _segments

        return priorities, segments, self.pid

    def _rollout(self) -> (list, list):

        env = gym.make(self.env_name)
        episode_buffer = EpisodeBuffer(nstep=self.nstep, gamma=self.gamma,
                                       burnin_length=self.burnin_len,
                                       unroll_length=self.unroll_len)

        frame = self.frame_process_func(env.reset())
        frames = collections.deque(
            [frame] * self.n_frames, maxlen=self.n_frames)

        c, h = self.q_network.lstm.get_initial_state(
            batch_size=1, dtype=tf.float32)
        done = False
        prev_action = 0
        lives = util.get_initial_lives(self.env_name)
        episode_rewards = 0
        while not done:

            state = np.stack(frames, axis=2)[np.newaxis, ...]

            action, (next_c, next_h) = self.q_network.sample_action(
                state, c, h, prev_action, self.epsilon)
            next_frame, reward, done, info = env.step(action)
            frames.append(self.frame_process_func(next_frame))
            next_state = np.stack(frames, axis=2)[np.newaxis, ...]

            if done:
                #: Episode terminal
                transition = (state, action, reward, next_state, done,
                              c, h, prev_action, True)
            elif lives != info["ale.lives"]:
                #: Life loss (roll)
                lives = info["ale.lives"]
                transition = (state, action, reward, next_state, True,
                              c, h, prev_action, False)
            else:
                transition = (state, action, reward, next_state, done,
                              c, h, prev_action, False)
            episode_buffer.add(transition)

            episode_rewards += reward
            c, h, prev_action = next_c, next_h, action

        segments = episode_buffer.pull_segments()

        """ Compute initial priority
        """
        states = np.stack([np.vstack(seg.states) for seg in segments], axis=1)  # (burnin+unroll_len, batch_size, obs_dim)
        actions = np.stack([seg.actions for seg in segments], axis=1)  # (burnin+unroll_len, batch_size)
        rewards = np.stack([seg.rewards for seg in segments], axis=1)  # (unroll_len, batch_size)
        dones = np.stack([seg.dones for seg in segments], axis=1)      # (unroll_len, batch_size)
        last_state = np.vstack([seg.last_state for seg in segments])   # (batch_size, obs_dim)

        c0 = tf.convert_to_tensor(
             np.vstack([seg.c_init for seg in segments]), dtype=tf.float32)  # (batch_size, lstm_out_dim)
        h0 = tf.convert_to_tensor(
             np.vstack([seg.h_init for seg in segments]), dtype=tf.float32)  # (batch_size, lstm_out_dim)

        a0 = np.atleast_2d([seg.prev_action_init for seg in segments])  # (1, bacth_size)
        prev_actions = np.vstack([a0, actions])[:-1]          # (burnin+unroll_len, batch_size)
        assert prev_actions.shape == actions.shape

        #: burn-in with stored-state
        c, h = c0, h0
        for t in range(self.burnin_len):
            _, (c, h) = self.q_network(
                states[t], states=[c, h], prev_action=prev_actions[t])

        qvalues = []
        for t in range(self.burnin_len, self.burnin_len+self.unroll_len):
            q, (c, h) = self.q_network(
                states[t], states=[c, h], prev_action=prev_actions[t])
            qvalues.append(q)
        qvalues = tf.stack(qvalues)                                          # (unroll_len, batch_size, action_space)
        actions_onehot = tf.one_hot(
            actions[self.burnin_len:], self.action_space)
        Q = tf.reduce_sum(qvalues * actions_onehot, axis=2, keepdims=False)  # (unroll_len, batch_size)

        #: compute qvlalue of last next-state in segment
        remaining_qvalue, _ = self.q_network(
            last_state, states=[c, h], prev_action=actions[-1])
        remaining_qvalue = tf.expand_dims(remaining_qvalue, axis=0)          # (1, batch_size, action_space)

        next_qvalues = tf.concat([qvalues[1:], remaining_qvalue], axis=0)    # (unroll_len, batch_size, action_space)
        next_actions = tf.argmax(next_qvalues, axis=2)                       # (unroll_len, batch_size)
        next_actions_onehot = tf.one_hot(next_actions, self.action_space)    # (unroll_len, batch_size, action_space)
        next_maxQ = tf.reduce_sum(
            next_qvalues * next_actions_onehot, axis=2, keepdims=False)      # (unroll_len, batch_size)

        TQ = rewards + self.gamma * (1 - dones) * next_maxQ  # (unroll_len, batch_size)

        td_errors = TQ - Q
        td_errors_abs = tf.abs(td_errors)

        priorities = self.eta * tf.reduce_max(td_errors_abs, axis=0) \
            + (1 - self.eta) * tf.reduce_mean(td_errors_abs, axis=0)
        priorities = (priorities + 0.001) ** self.alpha

        #: RAM節約のためにデータ圧縮
        compressed_segments = [lz4f.compress(pickle.dumps(seg))
                               for seg in segments]

        return priorities.numpy().tolist(), compressed_segments


@ray.remote(num_cpus=1)
class Tester:

    def __init__(self, env_name, n_frames):

        self.env_name = env_name
        self.frame_process_func = util.get_preprocess_func(env_name)
        self.n_frames = n_frames
        self.action_space = gym.make(self.env_name).action_space.n
        self.q_network = RecurrentDuelingQNetwork(self.action_space)
        self.define_network()

    def define_network(self):

        #: hide GPU from remote actor
        tf.config.set_visible_devices([], 'GPU')

        env = gym.make(self.env_name)

        frame = self.frame_process_func(env.reset())
        frames = [frame] * self.n_frames
        state = np.stack(frames, axis=2)[np.newaxis, ...]

        c, h = self.q_network.lstm.get_initial_state(batch_size=1, dtype=tf.float32)
        self.q_network(np.atleast_2d(state), states=[c, h], prev_action=[0])

    def test_play(self, current_weights, epsilon):

        self.q_network.set_weights(current_weights)

        env = gym.make(self.env_name)

        frame = self.frame_process_func(env.reset())
        frames = collections.deque(
            [frame] * self.n_frames, maxlen=self.n_frames)

        episode_rewards, steps = 0, 0

        c, h = self.q_network.lstm.get_initial_state(
            batch_size=1, dtype=tf.float32)
        prev_action = 0
        done = False
        while not done:
            steps += 1
            state = np.stack(frames, axis=2)[np.newaxis, ...]
            action, (next_c, next_h) = self.q_network.sample_action(state, c, h, prev_action, epsilon)
            next_frame, reward, done, _ = env.step(action)
            frames.append(self.frame_process_func(next_frame))
            episode_rewards += reward
            c, h, prev_action = next_c, next_h, action
            if steps > 500 and episode_rewards < 5:
                break

        return episode_rewards

    def test_with_video(self, checkpoint_path, monitor_dir, epsilon):

        monitor_dir = Path(monitor_dir)
        if monitor_dir.exists():
            shutil.rmtree(monitor_dir)
        monitor_dir.mkdir()
        env = gym.wrappers.Monitor(
            gym.make(self.env_name), monitor_dir, force=True,
            video_callable=(lambda ep: True))

        frame = self.frame_process_func(env.reset())
        frames = collections.deque(
            [frame] * self.n_frames, maxlen=self.n_frames)

        episode_rewards, steps = 0, 0

        c, h = self.q_network.lstm.get_initial_state(
            batch_size=1, dtype=tf.float32)
        prev_action = 0
        done = False
        while not done:
            steps += 1
            state = np.stack(frames, axis=2)[np.newaxis, ...]
            action, (next_c, next_h) = self.q_network.sample_action(state, c, h, prev_action, epsilon)
            next_frame, reward, done, _ = env.step(action)
            frames.append(self.frame_process_func(next_frame))
            episode_rewards += reward
            c, h, prev_action = next_c, next_h, action
            if steps > 500 and episode_rewards < 5:
                break

        return episode_rewards
