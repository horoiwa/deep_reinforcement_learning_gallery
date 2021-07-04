import random

import numpy as np


class PrioritizedReplay:

    def __init__(self, capacity, alpha=1.0, beta=1.0):

        assert capacity & (capacity - 1) == 0

        self.capacity = capacity

        self.buffer = [None] * self.capacity

        self.sumtree = SumTree(capacity=capacity)

        self.alpha = alpha

        self.beta = beta

        self.next_idx = 0

        self.full = False

    def __len__(self):

        return len(self.buffer) if self.full else self.next_idx

    def add_samples(self, priorities, samples):

        assert len(priorities) == len(samples)

        for priority, exp in zip(priorities, samples):

            self.sumtree[self.next_idx] = priority
            self.buffer[self.next_idx] = exp
            self.next_idx += 1

            if self.next_idx == self.capacity:
                self.full = True
                self.next_idx = 0

    def sample_minibatch(self, batchsize):

        indices = [self.sumtree.sample() for _ in range(batchsize)]

        weights = []
        for idx in indices:
            prob = self.sumtree[idx] / self.sumtree.sum()
            weight = (prob * len(self.buffer))**(-self.beta)
            weights.append(weight)
        weights = np.array(weights) / max(weights)

        samples = [self.buffer[idx] for idx in indices]

        return indices, weights, samples

    def update_priorities(self, indices, priorities):
        """ Update priorities of sampled transitions.
        """
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            self.sumtree[idx] = priority ** (self.alpha)


class SumTree:
    """ See https://github.com/ray-project/ray/blob/master/rllib/execution/segment_tree.py
    """

    def __init__(self, capacity: int):
        #: 2のべき乗チェック
        assert capacity & (capacity - 1) == 0
        self.capacity = capacity
        self.values = [0 for _ in range(2 * capacity)]

    def __str__(self):
        return str(self.values[self.capacity:])

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

    def sample(self, z=None):
        z = random.uniform(0, self.sum()) if z is None else z
        assert 0 <= z <= self.sum()

        current_idx = 1
        while current_idx < self.capacity:

            idx_lchild = 2 * current_idx
            idx_rchild = 2 * current_idx + 1

            #: 左子ノードよりzが大きい場合は右子ノードへ
            if z > self.values[idx_lchild]:
                current_idx = idx_rchild
                z = z -self.values[idx_lchild]
            else:
                current_idx = idx_lchild

        #: 見かけ上のインデックスにもどす
        idx = current_idx - self.capacity
        return idx
