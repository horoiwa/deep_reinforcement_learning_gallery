from dataclasses import dataclass
import collections
import zlib
import pickle

import numpy as np
import ray

import segment_tree


@dataclass
class Experience:

    state: np.ndarray
    action: float
    reward: float
    next_state: np.ndarray
    done: bool


class LocalReplayBuffer:

    def __init__(self, reward_clip, nstep, gamma, compress):

        self.buffer = []

        self.reward_clip = reward_clip

        self.nstep = nstep

        self.gamma = gamma

        self.temp_buffer = collections.deque(maxlen=nstep)

        self.compress = compress

        self.next_idx = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, transition):
        """
        Args:
            transition : tuple(state, action, reward, next_state, done)
        """

        self.temp_buffer.append(Experience(*transition))

        if len(self.temp_buffer) == self.nstep:

            nstep_return = 0
            has_done = False
            for i, onestep_exp in enumerate(self.temp_buffer):
                reward, done = onestep_exp.reward, onestep_exp.done
                reward = np.clip(reward, -1, 1) if self.reward_clip else reward
                nstep_return += self.gamma ** i * (1 - done) * reward
                if done:
                    has_done = True
                    break

            nstep_exp = Experience(self.temp_buffer[0].state,
                                   self.temp_buffer[0].action,
                                   nstep_return,
                                   self.temp_buffer[-1].next_state,
                                   has_done)

            self.buffer.append(nstep_exp)

    def pull(self):

        experiences = [self.buffer[idx] for idx in range(len(self.buffer))]

        states = np.vstack(
            [exp.state for exp in experiences]).astype(np.float32)

        actions = np.vstack(
            [exp.action for exp in experiences]).astype(np.float32)

        rewards = np.array(
            [exp.reward for exp in experiences]).reshape(-1, 1)

        next_states = np.vstack(
            [exp.next_state for exp in experiences]
            ).astype(np.float32)

        dones = np.array(
            [exp.done for exp in experiences]).reshape(-1, 1)

        if self.compress:
            experiences = [zlib.compress(pickle.dumps(exp)) for exp in experiences]

        self.buffer = []

        return (states, actions, rewards, next_states, dones), experiences


class GlobalPrioritizedReplayBuffer:

    def __init__(self, max_len, capacity, alpha, compress):

        assert capacity >= max_len, f"{capacity}  >= {max_len}"
        assert capacity & (capacity - 1) == 0

        self.max_len = max_len

        self.capacity = capacity

        self.sumtree = segment_tree.SumTree(capacity=capacity)

        self.buffer = []

        self.alpha = alpha

        self.compress = compress

        self.next_idx = 0

    def __len__(self):
        return len(self.buffer)

    def push_on_batch(self, experiences, priorities):

        assert len(experiences) == len(priorities)

        for exp, priority in zip(experiences, priorities):
            self.sumtree[self.next_idx] = priority
            self.buffer.append(exp)
            self.next_idx += 1

    def sample_minibatch(self, batch_size: int, beta: float):
        assert beta >= 0.0

        indices = [self.sumtree.sample() for _ in range(batch_size)]

        weights = []
        for idx in indices:
            prob = self.sumtree[idx] / self.sumtree.sum()
            weight = (prob * len(self.buffer))**(-beta)
            weights.append(weight)
        weights = np.array(weights) / max(weights)

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

    def update_priorities(self, indices, td_errors):
        """ Update priorities of sampled transitions.
        """
        assert len(indices) == len(td_errors)

        for idx, td_error in zip(indices, td_errors):
            priority = (np.abs(td_error) + 0.001) ** self.alpha
            self.sumtree[idx] = priority**self.alpha
            self.max_priority = max(self.max_priority, priority)

    def cut(self):
        if self.next_idx > self.max_len:
            for i in range(self.max_len, self.next_idx):
                self.sumtree[i] = 0

            self.buffer = self.buffer[:self.max_len]
            self.next_idx = 0


if __name__ == "__main__":
    pass
