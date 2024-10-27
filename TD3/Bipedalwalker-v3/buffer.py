import collections
from dataclasses import dataclass

import numpy as np


class ReplayBuffer:

    def __init__(self, max_experiences):

        self.max_experiences = max_experiences

        self.count = 0

        self.experiences = []

    def add_experience(self, exp):

        if len(self.experiences) == self.max_experiences:
            self.experiences[self.count] = exp
        else:
            self.experiences.append(exp)

        if self.count == self.max_experiences-1:
            self.count = 0
        else:
            self.count += 1

    def get_minibatch(self, batch_size):

        N = len(self.experiences)

        indices = np.random.choice(np.arange(N), replace=False,
                                   size=batch_size)

        selected_experiences = [self.experiences[idx] for idx in indices]

        states = np.vstack([exp.state for exp in selected_experiences]).astype(np.float32)

        actions = [exp.action for exp in selected_experiences]

        rewards = [exp.reward for exp in selected_experiences]

        next_states = np.vstack([exp.next_state for exp in selected_experiences]).astype(np.float32)

        dones = [exp.done for exp in selected_experiences]

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.experiences)
