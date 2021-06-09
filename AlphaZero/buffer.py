import collections

import numpy as np

import othello


class ReplayBuffer:

    def __init__(self, buffer_size):

        self.buffer = collections.deque(maxlen=buffer_size)

    def get_minibatch(self, batch_size):

        indices = np.random.choice(range(len(self.buffer)), size=batch_size)

        samples = [self.buffer[idx] for idx in indices]

        states = [othello.encode_state(s.state, s.player) for s in samples]

        rewards = [s.reward for s in samples]


        return None

    def add_record(self, record):
        for sample in record:
            self.buffer.append(sample)
