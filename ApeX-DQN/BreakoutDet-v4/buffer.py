"""
Simplified reimplementation of ray/rllib/execution/replay_buffer.py
https://github.com/ray-project/ray/blob/master/rllib/execution/replay_buffer.py

Original Licence;
# Copyright 2018, OpenCensus Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from dataclasses import dataclass
import random
import zlib
import pickle
import time

import numpy as np

import util


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


class DistributedPrioritizedReplayBuffer:

    def __init__(self, max_len: int, reward_clip: bool, compress: bool,
                 alpha: float, nstep=3):

        #:SumSegmentTreeの都合上、max_lenは2のべき乗に設定する
        assert max_len & (max_len - 1) == 0

        self.buffer = []

        self.max_len = max_len

        self.reward_clip = reward_clip

        self.compress = compress

        self.alpha = alpha

        self.sumtree = util.SumSegmentTree(size=max_len)

        self.mintree = util.MinSegmentTree(size=max_len)

        self.nstep = nstep

        self.next_idx = 0

    def push(self, transition, td_error):
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


def performance_test():

    def dummy_exp():
        state = np.random.rand(84*84*4).reshape(84,84,4)
        next_state = np.random.rand(84*84*4).reshape(84,84,4)
        return (state, 2, 1., next_state, False)

    N = 800
    with util.Timer("準備"):
        experiences = [dummy_exp() for _ in range(N)]

    buffer = ReplayBuffer(max_len=1000000, reward_clip=True, compress=False)

    with util.Timer("push"):
        for exp in experiences:
            buffer.push(exp)

    M = int(N / 4)
    with util.Timer("pull: batch_size=32"):
        for _ in range(M):
            _ = buffer.sample_minibatch(batch_size=32)

    priorities = list(np.random.uniform(0, 5, size=1000000))

    with util.Timer("RANDOM ER"):
        for _ in range(M):
            indices = np.random.choice(np.arange(len(priorities)), replace=False)

    with util.Timer("numpy.random.choice PER"):
        for _ in range(M):
            probs = np.array(priorities) / sum(priorities)
            indices = np.random.choice(np.arange(len(probs)), replace=False, p=probs)


    with util.Timer("SumTree"):
        for _ in range(M):
            indices = None


if __name__ == '__main__':
    performance_test()
