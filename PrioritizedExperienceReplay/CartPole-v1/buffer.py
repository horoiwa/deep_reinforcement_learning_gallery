import collections
from dataclasses import dataclass

import numpy as np


class PrioritizedReplayBuffer:

    ALPHA = 0.7

    EPSILON = 0.01

    def __init__(self, max_experiences):

        self.max_experiences = max_experiences

        self.count = 0

        self.experiences = []

        self.priorities = np.zeros(self.max_experiences)

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

        weights = (np.array([probs[idx] for idx in indices]) * N) ** -beta

        weights /= weights.max()

        experiences = [self.experiences[idx] for idx in indices]

        return indices, weights, experiences

    def update_priority(self, indices, td_errors):

        assert len(indices) == len(td_errors)

        priorities = (np.abs(td_errors) + self.EPSILON) ** self.ALPHA

        self.priorities[indices] = priorities

        self.max_priority = max(self.max_priority, priorities.max())

    def __len__(self):
        return len(self.experiences)


if __name__ == "__main__":
    import numpy as np
    import random

    buffer = PrioritizedReplayBuffer(max_experiences=16)

    Exp = collections.namedtuple("Experience",
                                 ["state", "action",
                                  "reward", "next_state", "done", "priority"])
    for i in range(32):

        s1 = [np.random.randint(100) for _ in range(4)]

        a = [np.random.randint(2)]

        r = i

        s2 = [np.random.randint(100) for _ in range(4)]

        done = random.choice([False, True])

        exp = Experience(s1, a, r, s2, done)

        buffer.add_experience(exp)

    print(len(buffer))

    print()

    for exp in buffer.experiences:
        print(exp.reward)

    indices, weights, experiences = buffer.get_minibatch(4, 0.5)
    print(indices)
    print(weights)

    td_errors = np.random.randint(-100, 100, size=len(indices))
    print(td_errors)
    buffer.update_priority(indices, td_errors)
    print()
    indices, weights, experiences = buffer.get_minibatch(4, 0.5)
    print(indices)
    print(weights)
    print()
    print(buffer.max_priority)
