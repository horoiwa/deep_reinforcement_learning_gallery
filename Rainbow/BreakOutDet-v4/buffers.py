from dataclasses import dataclass
import functools
import collections

import numpy as np
import pickle
import zlib


def create_replaybuffer(use_priority, use_multistep, max_len, reward_clip,
                        alpha, beta, total_steps, nstep_return, gamma):

    if use_priority and use_multistep:
        return NstepPrioritizedReplayBuffer(
            max_len=max_len, reward_clip=reward_clip,
            alpha=alpha, beta=beta, total_steps=total_steps,
            nstep_return=nstep_return, gamma=gamma)

    elif use_priority:
        return PrioritizedReplayBuffer(
            max_len=max_len, reward_clip=reward_clip,
            alpha=alpha, beta=beta, total_steps=total_steps)

    elif use_multistep:
        return NstepReplayBuffer(
            max_len=max_len, reward_clip=reward_clip,
            nstep_return=nstep_return, gamma=gamma)
    else:
        return ReplayBuffer(max_len=max_len, reward_clip=reward_clip)


@dataclass
class Experience:

    state: np.ndarray

    action: float

    reward: float

    next_state: np.ndarray

    done: bool


class ReplayBuffer:

    def __init__(self, max_len, reward_clip, compress=True):

        self.max_len = max_len

        self.buffer = []

        self.compress = compress

        self.reward_clip = reward_clip

        self.count = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, transition):
        """
            transition : tuple(state, action, reward, next_state, done)
        """

        exp = Experience(*transition)

        exp.reward = np.clip(exp.reward, -1, 1) if self.reward_clip else exp.reward

        if self.compress:
            exp = zlib.compress(pickle.dumps(exp))

        if self.count == self.max_len:
            self.count = 0

        try:
            self.buffer[self.count] = exp
        except IndexError:
            self.buffer.append(exp)

        self.count += 1

    def get_minibatch(self, batch_size):

        N = len(self.buffer)

        indices = np.random.choice(
            np.arange(N), replace=False, size=batch_size)

        if self.compress:
            selected_experiences = [
                pickle.loads(zlib.decompress(self.buffer[idx])) for idx in indices]
        else:
            selected_experiences = [self.buffer[idx] for idx in indices]

        states = np.vstack(
            [exp.state for exp in selected_experiences]).astype(np.float32)

        actions = np.vstack(
            [exp.action for exp in selected_experiences]).astype(np.float32)

        rewards = np.array(
            [exp.reward for exp in selected_experiences]).reshape(-1, 1)

        next_states = np.vstack(
            [exp.next_state for exp in selected_experiences]
            ).astype(np.float32)

        dones = np.array(
            [exp.done for exp in selected_experiences]).reshape(-1, 1)

        return (states, actions, rewards, next_states, dones)


class NstepReplayBuffer(ReplayBuffer):

    def __init__(self, max_len, reward_clip, nstep_return, gamma, compress=True):

        super().__init__(max_len, reward_clip, compress)

        self.nstep_return = nstep_return

        self.gamma = gamma

        self.temp_buffer = collections.deque(maxlen=nstep_return)

    def push(self, transition):
        """
        Args:
            transition : tuple(state, action, reward, next_state, done)
        """

        self.temp_buffer.append(Experience(*transition))

        if len(self.temp_buffer) == self.nstep_return:

            nstep_return = 0
            has_done = False
            for i, onestep_exp in enumerate(self.temp_buffer):
                reward, done = onestep_exp.reward, onestep_exp.done
                reward = np.clip(reward, -1, 1) if self.reward_clip else reward
                nstep_return += self.gamma ** i * (1 - done) * reward
                if done:
                    has_done = True
                    break

            nstep_exp = Experience(self.temp_buffer[0].state,
                                   self.temp_buffer[0].action,
                                   nstep_return,
                                   self.temp_buffer[-1].next_state,
                                   has_done)

            if self.compress:
                nstep_exp = zlib.compress(pickle.dumps(nstep_exp))

            if self.count == self.max_len:
                self.counter = 0

            try:
                self.buffer[self.count] = nstep_exp
            except IndexError:
                self.buffer.append(nstep_exp)

            self.count += 1


class PrioritizedReplayBuffer:

    def __init__(self, max_len, reward_clip, alpha=0.6, beta=0.4,
                 total_steps=2500000, compress=True):

        self.max_len = max_len

        self.buffer = []

        self.priorities = []

        self.alpha = alpha

        self.beta_scheduler = (
            lambda steps: beta + (1 - beta) * steps / total_steps)

        self.epsilon = 0.01

        self.max_priority = 1.0

        self.reward_clip = reward_clip

        self.compress = compress

        self.count = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, transition):
        """
        Args:
            transition : tuple(state, action, reward, next_state, done)
        """

        assert len(self.buffer) == len(self.priorities)

        exp = Experience(*transition)

        exp.reward = np.clip(exp.reward, -1, 1) if self.reward_clip else exp.reward

        if self.compress:
            exp = zlib.compress(pickle.dumps(exp))

        if self.count == self.max_len:
            self.count = 0

        try:
            self.buffer[self.count] = exp
            self.priorities[self.count] = self.max_priority
        except IndexError:
            self.buffer.append(exp)
            self.priorities.append(self.max_priority)

        self.count += 1

    def get_minibatch(self, batch_size, steps):

        N = len(self.buffer)

        beta = self.beta_scheduler(steps)

        probs = np.array(self.priorities) / sum(self.priorities)

        indices = np.random.choice(np.arange(N), p=probs,
                                   replace=False, size=batch_size)

        weights = (probs[indices] * N) ** (-1 * beta)
        weights /= weights.max()
        weights = weights.reshape(-1, 1).astype(np.float32)

        if self.compress:
            selected_experiences = [
                pickle.loads(zlib.decompress(self.buffer[idx])) for idx in indices]
        else:
            selected_experiences = [self.buffer[idx] for idx in indices]

        states = np.vstack(
            [exp.state for exp in selected_experiences]).astype(np.float32)

        actions = np.vstack(
            [exp.action for exp in selected_experiences]).astype(np.float32)

        rewards = np.array(
            [exp.reward for exp in selected_experiences]).reshape(-1, 1)

        next_states = np.vstack(
            [exp.next_state for exp in selected_experiences]
            ).astype(np.float32)

        dones = np.array(
            [exp.done for exp in selected_experiences]).reshape(-1, 1)

        return indices, weights, (states, actions, rewards, next_states, dones)

    def update_priority(self, indices, td_errors):
        """
        Args:
            indices : 1D-array
            td_errors : 1D-array
        """
        assert len(indices) == len(td_errors)

        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha

        #: update priority
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

        self.max_priority = max(self.max_priority, priorities.max())


class NstepPrioritizedReplayBuffer:

    def __init__(self, max_len, gamma, reward_clip,
                 nstep_return=3, alpha=0.6, beta=0.4,
                 total_steps=2500000, compress=True):

        self.max_len = max_len

        self.gamma = gamma

        self.buffer = []

        self.priorities = []

        self.nstep_return = nstep_return

        self.temp_buffer = collections.deque(maxlen=nstep_return)

        self.alpha = alpha

        self.beta_scheduler = (
            lambda steps: beta + (1 - beta) * steps / total_steps)

        self.epsilon = 0.01

        self.max_priority = 1.0

        self.reward_clip = reward_clip

        self.compress = compress

        self.counter = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, transition):
        """
        Args:
            transition : tuple(state, action, reward, next_state, done)
        """

        assert len(self.buffer) == len(self.priorities)

        self.temp_buffer.append(Experience(*transition))

        if len(self.temp_buffer) == self.nstep_return:

            nstep_return = 0
            has_done = False
            for i, exp in enumerate(self.temp_buffer):
                reward, done = exp.reward, exp.done
                reward = np.clip(reward, -1, 1) if self.reward_clip else reward
                nstep_return += self.gamma ** i * (1 - done) * reward
                if done:
                    has_done = True
                    break

            nstep_exp = Experience(self.temp_buffer[0].state,
                                   self.temp_buffer[0].action,
                                   nstep_return,
                                   self.temp_buffer[-1].next_state,
                                   has_done)

            if self.compress:
                nstep_exp = zlib.compress(pickle.dumps(nstep_exp))

            if self.counter == self.max_len:
                self.counter = 0

            try:
                self.buffer[self.counter] = nstep_exp
                self.priorities[self.counter] = self.max_priority
            except IndexError:
                self.buffer.append(nstep_exp)
                self.priorities.append(self.max_priority)

            self.counter += 1

    def get_minibatch(self, batch_size, steps):

        beta = self.beta_scheduler(steps)

        probs = np.array(self.priorities) / sum(self.priorities)

        indices = np.random.choice(np.arange(len(self.buffer)), p=probs,
                                   replace=False, size=batch_size)

        weights = (probs[indices] * len(self.buffer)) ** (-1 * beta)
        weights /= weights.max()
        weights = weights.reshape(-1, 1).astype(np.float32)

        if self.compress:
            selected_experiences = [
                pickle.loads(zlib.decompress(self.buffer[idx])) for idx in indices]
        else:
            selected_experiences = [self.buffer[idx] for idx in indices]

        states = np.vstack(
            [exp.state for exp in selected_experiences]).astype(np.float32)

        actions = np.vstack(
            [exp.action for exp in selected_experiences]).astype(np.float32)

        rewards = np.array(
            [exp.reward for exp in selected_experiences]).reshape(-1, 1)

        next_states = np.vstack(
            [exp.next_state for exp in selected_experiences]
            ).astype(np.float32)

        dones = np.array(
            [exp.done for exp in selected_experiences]).reshape(-1, 1)

        return indices, weights, (states, actions, rewards, next_states, dones)

    def update_priority(self, indices, td_errors):
        """
        Args:
            indices : 1D-array
            td_errors : 1D-array
        """
        assert len(indices) == len(td_errors)

        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha

        #: update priority
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

        self.max_priority = max(self.max_priority, priorities.max())
