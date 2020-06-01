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

    priority: float


class PrioritizedReplayBuffer:

    def __init__(self, max_experiences):

        self.max_experiences = max_experiences

        self.n = 0

        self.experiences = []

    def add(self, exp):

        exp.priority = 0

        if len(self.experiences) < self.max_experiences:
            self.experiences.append(exp)

        elif len(self.experiences) == self.max_experiences:
            self.n = 0
            self.experiences[self.n] = exp
            self.n += 1

        else:
            self.experiences[self.n] = exp
            self.n += 1

    @property
    def max_priority(self):
        return max([exp.priority for exp in self.experiences])

    def get_prioritized_minibatch(self, batch_size):
        pass

    def update_priority(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.experiences[i].priority = priority

    def __len__(self):
        return len(self.experiences)



if __name__ == "__main__":
    import numpy as np
    import random

    buffer = PrioritizedReplayBuffer(max_experiences=8)

    Exp = collections.namedtuple("Experience",
                                 ["state", "action",
                                  "reward", "next_state", "done", "priority"])
    for _ in range(10):

        s1 = [np.random.randint(100) for _ in range(4)]

        a = [np.random.randint(2)]

        r = 1

        s2 = [np.random.randint(100) for _ in range(4)]

        done = random.choice([False, True])

        priority = 1

        exp = Experience(s1, a, r, s2, done, priority)

        buffer.add(exp)

    print(len(buffer))

    print*
