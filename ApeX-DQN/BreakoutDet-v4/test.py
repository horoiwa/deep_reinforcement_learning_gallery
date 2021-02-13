import random

import numpy as np

from buffer import ReplayBuffer, PrioritizedReplayBuffer
import segment_tree
import util


def performance_test_1(compress):

    def dummy_exp():
        state = np.random.rand(84*84*4).reshape(84,84,4)
        next_state = np.random.rand(84*84*4).reshape(84,84,4)
        return (state, 2, 1., next_state, False)

    N = 800
    with util.Timer("準備"):
        experiences = [dummy_exp() for _ in range(N)]

    buffer = ReplayBuffer(max_len=1000000, reward_clip=True, compress=compress)

    with util.Timer("push"):
        for exp in experiences:
            buffer.push(exp)

    M = int(N / 4)
    with util.Timer("pull"):
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


def performance_test_2(compress):

    def dummy_exp():
        state = np.random.rand(84*84*4).reshape(84,84,4)
        next_state = np.random.rand(84*84*4).reshape(84,84,4)
        return (state, 2, 1., next_state, False)

    N = 800
    with util.Timer("PER 準備"):
        experiences = [dummy_exp() for _ in range(N)]
        priorities = list(np.random.uniform(-5, 5, size=N))

    buffer = PrioritizedReplayBuffer(
        max_len=2**20, reward_clip=True, compress=compress, alpha=0.6)

    with util.Timer("PER push"):
        for exp, prior in zip(experiences, priorities):
            buffer.push(exp, prior)

    M = int(N / 4)
    with util.Timer("PER pull"):
        for _ in range(M):
            _ = buffer.sample_minibatch(batch_size=32, beta=0.4)

    sumtree = segment_tree.SumSegmentTree(capacity=2**20)
    priorities = list(np.random.uniform(0, 5, size=1000000))
    for i, prior in enumerate(priorities):
        sumtree[i] = prior

    with util.Timer("sumtree PER"):
        for _ in range(M):
            indices = []
            for _ in range(32):
                mass = random.random() * sumtree.sum(0, 1000000)
                idx = sumtree.find_prefixsum_idx(mass)
                indices.append(idx)


if __name__ == '__main__':
    compress = False
    performance_test_1(compress=compress)
    performance_test_2(compress=compress)
