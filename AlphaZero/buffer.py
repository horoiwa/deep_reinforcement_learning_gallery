import collections

import numpy as np

import othello


class ReplayBuffer:

    def __init__(self, buffer_size):

        self.buffer = collections.deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def get_minibatch(self, batch_size):

        indices = np.random.choice(range(len(self.buffer)), size=batch_size)

        samples = [self.buffer[idx] for idx in indices]

        states = np.stack(
            [othello.encode_state(s.state, s.player) for s in samples],
            axis=0)

        mcts_policy = np.array(
            [s.mcts_policy for s in samples], dtype=np.float32)

        rewards = np.array(
            [s.reward for s in samples], dtype=np.float32).reshape(-1, 1)

        return (states, mcts_policy, rewards)

    def add_record(self, record):
        for sample in record:
            self.buffer.append(sample)
