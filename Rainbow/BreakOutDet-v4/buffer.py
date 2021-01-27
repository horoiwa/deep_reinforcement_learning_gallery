from dataclasses import dataclass

import numpy as np
import pickle
import zlib


@dataclass
class Experience:

    state: np.ndarray

    action: float

    reward: float

    next_state: np.ndarray

    done: bool


class ReplayBuffer:

    def __init__(self, max_len, compress=True):

        self.max_len = max_len

        self.buffer = []

        self.compress = compress

        self.count = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, transition):
        """
            transition : tuple(state, action, reward, next_state, done)
        """

        exp = Experience(*transition)

        if self.compress:
            exp = zlib.compress(pickle.dumps(exp))

        if self.count == self.max_len:
            self.count = 0

        try:
            self.buffer[self.count] = exp
        except IndexError:
            self.buffer.append(exp)

        self.count += 1

    def get_minibatch(self, batch_size):

        N = len(self.buffer)

        indices = np.random.choice(
            np.arange(N), replace=False, size=batch_size)

        if self.compress:
            selected_experiences = [
                pickle.loads(zlib.decompress(self.buffer[idx])) for idx in indices]
        else:
            selected_experiences = [self.buffer[idx] for idx in indices]

        states = np.vstack(
            [exp.state for exp in selected_experiences]).astype(np.float32)

        actions = np.vstack(
            [exp.action for exp in selected_experiences]).astype(np.float32)

        rewards = np.array(
            [exp.reward for exp in selected_experiences]).reshape(-1, 1)

        next_states = np.vstack(
            [exp.next_state for exp in selected_experiences]
            ).astype(np.float32)

        dones = np.array(
            [exp.done for exp in selected_experiences]).reshape(-1, 1)

        return (states, actions, rewards, next_states, dones)
