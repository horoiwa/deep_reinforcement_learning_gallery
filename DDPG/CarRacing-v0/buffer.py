import collections
from dataclasses import dataclass

import numpy as np


class ReplayBuffer:


    def __init__(self, max_experiences):

        self.max_experiences = max_experiences

        self.count = 0

        self.experiences = []

        self.priorities = np.zeros(self.max_experiences, dtype=np.float32)

        self.max_priority = 1.0

    def add_experience(self, exp):

        if len(self.experiences) == self.max_experiences:
            self.experiences[self.count] = exp
        else:
            self.experiences.append(exp)

        self.priorities[self.count] = self.max_priority

        if self.count == self.max_experiences-1:
            self.count = 0
        else:
            self.count += 1

    def get_minibatch(self, batch_size, beta):

        N = len(self.experiences)

        probs = (self.priorities / self.priorities.sum())[:N]

        indices = np.random.choice(np.arange(N), p=probs,
                                   replace=False, size=batch_size)

        selected_probs = np.array(
            [probs[idx] for idx in indices], dtype=np.float32)

        selected_weights = (selected_probs * N) ** -beta

        selected_weights /= selected_weights.max()

        selected_experiences = [self.experiences[idx] for idx in indices]

        return indices, selected_weights, selected_experiences

    def update_priority(self, indices, td_errors):

        assert len(indices) == len(td_errors)

        priorities = (np.abs(td_errors) + self.EPSILON) ** self.ALPHA

        self.priorities[indices] = priorities

        self.max_priority = max(self.max_priority, priorities.max())

    def __len__(self):
        return len(self.experiences)
