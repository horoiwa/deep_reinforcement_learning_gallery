import random
from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayElement:
    state: np.ndarray
    action: int
    reward: float
    is_terminal: int


class PrioritizedReplayBuffer:
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.buffer = [None] * maxlen
        self.priorities = SumTree(capacity=self.mexlen)
        self.beta = 0.6  # importance sampling exponent
        self.cursor = 0
        self.full = False

    def __len__(self):
        return len(self.segment_buffer) if self.full else self.count

    def add(self, priorities: list[float], elements: list[ReplayElement]):

        for priority, element in zip(priorities, elements, strict=True):
            self.priorities[self.cursor] = priority
            self.segment_buffer[self.cursor] = element

            self.cursor += 1
            if self.cursor == self.buffer_size:
                self.cursor = 0
                self.full = True

    def update_priority(self, indices: list[int], priorities: list[float]):
        for idx, priority in zip(indices, priorities, strict=True):
            self.priorities[idx] = priority

    def sample_batch(self, batch_size: int):
        indices = [self.priorities.sample() for _ in range(batch_size)]

        #: Compute importance sampling weights
        weights = []
        current_size = len(self.buffer) if self.full else self.count
        for idx in indices:
            prob = self.priorities[idx] / self.priorities.sum()
            weight = (prob * current_size) ** (-self.beta)
            weights.append(weight)
        weights = np.array(weights) / max(weights)

        elements = [self.segment_buffer[idx] for idx in indices]

        return indices, weights, elements


class SumTree:
    """See https://github.com/ray-project/ray/blob/master/rllib/execution/segment_tree.py"""

    def __init__(self, capacity: int):
        #: 2のべき乗チェック
        assert capacity & (capacity - 1) == 0
        self.capacity = capacity
        self.values = [0 for _ in range(2 * capacity)]

    def __str__(self):
        return str(self.values[self.capacity :])

    def __setitem__(self, idx, val):
        idx = idx + self.capacity
        self.values[idx] = val

        current_idx = idx // 2
        while current_idx >= 1:
            idx_lchild = 2 * current_idx
            idx_rchild = 2 * current_idx + 1
            self.values[current_idx] = self.values[idx_lchild] + self.values[idx_rchild]
            current_idx //= 2

    def __getitem__(self, idx):
        idx = idx + self.capacity
        return self.values[idx]

    def sum(self):
        return self.values[1]

    def sample(self):
        z = random.uniform(0, self.sum())
        try:
            assert 0 <= z <= self.sum(), z
        except AssertionError:
            print(z)
            import pdb

            pdb.set_trace()

        current_idx = 1
        while current_idx < self.capacity:

            idx_lchild = 2 * current_idx
            idx_rchild = 2 * current_idx + 1

            #: 左子ノードよりzが大きい場合は右子ノードへ
            if z > self.values[idx_lchild]:
                current_idx = idx_rchild
                z = z - self.values[idx_lchild]
            else:
                current_idx = idx_lchild

        #: 見かけ上のインデックスにもどす
        idx = current_idx - self.capacity
        return idx
