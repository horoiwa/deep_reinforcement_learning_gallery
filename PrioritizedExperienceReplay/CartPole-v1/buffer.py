import collections
from dataclasses import dataclass

import numpy as np


@dataclass
class Experience:

    state: np.ndarray

    action: int

    reward: float

    next_state: np.ndarray

    done: bool


class ReplayBuffer:

    ALPHA = 0.7

    EPSILON = 0.01

    def __init__(self, max_experiences):

        self.max_experiences = max_experiences

        self.count = 0

        self.experiences = []

        self.priorities = np.zeros(self.max_experiences) + self.EPSILON

    def add_experience(self, exp):

        if len(self.experiences) == self.max_experiences:
            self.experiences[self.count] = exp
        else:
            self.experiences.append(exp)

        self.priorities[self.count] = self.priorities.max()

        if self.count == self.max_experiences-1:
            self.count = 0
        else:
            self.count += 1

    def get_minibatch(self, batch_size):
        selected_indices = np.random.choice(
            np.arange(len(self.experiences)), size=batch_size)

        selected_experiences = [self.experiences[idx] for idx in selected_indices]

        return selected_experiences

    def get_prioritized_minibatch(self, batch_size):

        weights = self.priorities / self.priorities.sum()

        selected_indices = np.random.choice(
            np.arange(len(self.experiences)), p=weights, size=batch_size)

        selected_weights = [weights[idx] for idx in selected_indices]

        selected_experiences = [self.experiences[idx] for idx in selected_indices]

        return selected_indices, selected_weights, selected_experiences

    def update_priority(self, indices, new_priorities):

        self.priorities[indices] = new_priorities

    def __len__(self):
        return len(self.experiences)



if __name__ == "__main__":
    import numpy as np
    import random

    buffer = ReplayBuffer(max_experiences=8)

    Exp = collections.namedtuple("Experience",
                                 ["state", "action",
                                  "reward", "next_state", "done", "priority"])
    for i in range(20):

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

