from dataclasses import dataclass
import random
import zlib
import pickle
import time

import numpy as np

import util
import segment_tree


@dataclass
class Experience:

    state: np.ndarray
    action: float
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:

    def __init__(self, max_len: int, reward_clip: bool, compress: bool):
        """Create Prioritized Replay buffer.

        Args:
            size (int): Max number of timesteps to store in the FIFO buffer.
        """
        self.buffer = []
        self.max_len = max_len
        self.reward_clip = reward_clip
        self.compress = compress

        self.next_idx = 0

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

        if self.next_idx >= len(self.buffer):
            self.buffer.append(exp)
        else:
            self.buffer[self.next_idx] = exp

        if len(self.buffer) >= self.max_len:
            self.next_idx = 0
        else:
            self.next_idx += 1

    def sample_minibatch(self, batch_size: int):

        indices = [random.randint(0, len(self.buffer) - 1) for _ in range(batch_size)]

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


class PrioritizedReplayBuffer:

    def __init__(self, max_len: int, reward_clip: bool, compress: bool,
                 alpha: float, nstep=3):

        #:SumSegmentTreeの都合上、max_lenは2の倍数に設定する
        assert max_len % 2 == 0

        self.buffer = []

        self.max_len = max_len

        self.reward_clip = reward_clip

        self.compress = compress

        self.alpha = alpha

        self.sumtree = segment_tree.SumSegmentTree(capacity=max_len)

        self.mintree = segment_tree.MinSegmentTree(capacity=max_len)

        self.nstep = nstep

        self.next_idx = 0

    def push(self, transition, td_error=None):
        """
            transition : tuple(state, action, reward, next_state, done)
        """

        exp = Experience(*transition)
        exp.reward = np.clip(exp.reward, -1, 1) if self.reward_clip else exp.reward

        if self.compress:
            exp = zlib.compress(pickle.dumps(exp))

        if self.next_idx >= len(self.buffer):
            self.buffer.append(exp)
        else:
            self.buffer[self.next_idx] = exp

        priority = (np.abs(td_error) + 0.001) ** self.alpha
        self.sumtree[self.next_idx] = priority
        self.mintree[self.next_idx] = priority

        if len(self.buffer) >= self.max_len:
            self.next_idx = 0
        else:
            self.next_idx += 1

    def sample_minibatch(self, batch_size: int, beta: float):
        assert beta >= 0.0

        indices = []
        for _ in range(batch_size):
            mass = random.random() * self.sumtree.sum(0, len(self.buffer))
            idx = self.sumtree.find_prefixsum_idx(mass)
            indices.append(idx)

        """
            PER論文ではミニバッチの最大重みで重みのスケーリングをしていたが、
            rllibにならってバッファ内最小重みでスケーリングする
        """
        p_min = self.mintree.min() / self.sumtree.sum()
        max_weight = (p_min * len(self.buffer))**(-beta)

        weights = []
        for idx in indices:
            prob = self.sumtree[idx] / self.sumtree.sum()
            weight = (prob * len(self.buffer))**(-beta) / max_weight
            weights.append(weight)

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

    def update_priorities(self, indices, td_errors):
        """ Update priorities of sampled transitions.
        """
        assert len(indices) == len(td_errors)

        for idx, td_error in zip(indices, td_errors):
            priority = (np.abs(td_error) + 0.001) ** self.alpha
            self.sumtree[idx] = priority**self.alpha
            self.mintree[idx] = priority**self.alpha

            self._max_priority = max(self._max_priority, priority)
