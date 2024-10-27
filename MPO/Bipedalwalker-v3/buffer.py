import collections

import numpy as np


class ReplayBuffer:

    def __init__(self, maxlen):

        self.buffer = collections.deque(maxlen=maxlen)

    def __len__(self):
        return len(self.buffer)

    def add(self, exp):

        self.buffer.append(exp)

    def get_minibatch(self, batch_size):

        N = len(self.buffer)

        indices = np.random.choice(
            np.arange(N), replace=False, size=batch_size)

        selected_experiences = [self.buffer[idx] for idx in indices]

        states = np.vstack(
            [exp.state for exp in selected_experiences]).astype(np.float32)

        actions = np.vstack(
            [exp.action for exp in selected_experiences]).astype(np.float32)

        rewards = np.array(
            [exp.reward for exp in selected_experiences]
            ).reshape(-1, 1).astype(np.float32)

        next_states = np.vstack(
            [exp.next_state for exp in selected_experiences]
            ).astype(np.float32)

        dones = np.array(
            [exp.done for exp in selected_experiences]
            ).reshape(-1, 1).astype(np.float32)

        return (states, actions, rewards, next_states, dones)

