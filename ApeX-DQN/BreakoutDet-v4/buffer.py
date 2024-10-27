from dataclasses import dataclass
import collections
import math

import numpy as np

import util


@dataclass
class Experience:

    state: np.ndarray
    action: float
    reward: float
    next_state: np.ndarray
    done: bool


class LocalReplayBuffer:

    def __init__(self, reward_clip, gamma, nstep):

        self.buffer = []

        self.reward_clip = reward_clip

        self.nstep = nstep

        self.gamma = gamma

        self.temp_buffer = collections.deque(maxlen=nstep)

        self.next_idx = 0

    def __len__(self):

        return len(self.buffer)

    def push(self, transition):
        """
        Args:
            transition : tuple(state, action, reward, next_state, done)
        """

        self.temp_buffer.append(Experience(*transition))

        if len(self.temp_buffer) == self.nstep:

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

            self.buffer.append(nstep_exp)

    def pull(self):

        experiences = self.buffer

        self.buffer = []

        return experiences


class GlobalReplayBuffer:

    def __init__(self, capacity, alpha, beta):

        assert capacity & (capacity - 1) == 0

        self.capacity = capacity

        self.buffer = [None] * self.capacity

        self.sumtree = util.SumTree(capacity=capacity)

        self.alpha = alpha

        self.beta = beta

        self.next_idx = 0

        self.full = False

    def __len__(self):

        return len(self.buffer) if self.full else self.next_idx

    def push(self, priorities, experiences):

        assert len(priorities) == len(experiences)

        for priority, exp in zip(priorities, experiences):
            self.sumtree[self.next_idx] = priority
            self.buffer[self.next_idx] = exp
            self.next_idx += 1
            if self.next_idx == self.capacity:
                self.full = True
                self.next_idx = 0

    def sample_batch(self, batch_size):

        indices = [self.sumtree.sample() for _ in range(batch_size)]

        weights = []
        for idx in indices:
            prob = self.sumtree[idx] / self.sumtree.sum()
            weight = (prob * len(self.buffer))**(-self.beta)
            weights.append(weight)
        weights = np.array(weights) / max(weights)

        experiences = [self.buffer[idx] for idx in indices]

        return indices, weights, experiences

    def update_priorities(self, indices, td_errors):
        """ Update priorities of sampled transitions.
        """
        assert len(indices) == len(td_errors)

        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 0.001) ** self.alpha
            self.sumtree[idx] = priority**self.alpha


if __name__ == "__main__":
    pass
