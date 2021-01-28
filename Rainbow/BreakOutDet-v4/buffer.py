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


class PrioritizedReplayBuffer:

    def __init__(self, max_len, alpha=0.6, epsilon=0.01, compress=True):

        self.max_len = max_len

        self.buffer = []

        self.priorities = []

        self.alpha = alpha

        self.epsilon = epsilon

        self.max_priority = 1.0

        self.compress = compress

        self.counter = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, transition):
        """
        Args:
            transition : tuple(state, action, reward, next_state, done)
        """

        assert len(self.buffer) == len(self.priorities)

        exp = Experience(*transition)

        if self.compress:
            exp = zlib.compress(pickle.dumps(exp))

        if self.counter == self.max_len:
            self.counter = 0

        try:
            self.buffer[self.counter] = exp
            self.priorities[self.counter] = self.max_priority
        except IndexError:
            self.buffer.append(exp)
            self.priorities.append(self.max_priority)

        self.counter += 1

    def get_minibatch(self, batch_size, beta):

        probs = np.array(self.priorities) / sum(self.priorities)

        indices = np.random.choice(np.arange(len(self.buffer)), p=probs,
                                   replace=False, size=batch_size)

        weights = (probs[indices] * len(self.buffer)) ** (-1 * beta)

        weights /= weights.max()

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

        return indices, weights, (states, actions, rewards, next_states, dones)

    def update_priority(self, indices, td_errors):
        """
        Args:
            indices : 1D-array
            td_errors : 1D-array
        """
        assert len(indices) == len(td_errors)

        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha

        #: update priority
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

        self.max_priority = max(self.max_priority, priorities.max())


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


if __name__ == "__main__":
    import numpy as np
    import random

    buffer = PrioritizedReplayBuffer(max_len=16, compress=False)

    for i in range(32):

        s1 = [np.random.randint(100) for _ in range(4)]

        a = [np.random.randint(2)]

        r = i

        s2 = [np.random.randint(100) for _ in range(4)]

        done = random.choice([False, True])

        transition = (s1, a, r, s2, done)

        buffer.push(transition)

    print("LEN", len(buffer))
    print()

    for exp in buffer.buffer:
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