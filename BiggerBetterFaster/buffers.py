import random
import tensorflow as tf
from dataclasses import dataclass

import numpy as np


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    is_done: int


class ReplayBuffer:
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def append(self, exp: Experience):
        self.buffer.append(exp)
        if len(self.buffer) > self.maxlen:
            self.buffer = self.buffer[-self.maxlen :]

    def sample_batch(self, batch_size: int, n_step: int, gamma: float):
        assert n_step > 0
        indices = [random.randint(0, len(self.buffer) - 10) for _ in range(batch_size)]
        trajectories = [self.buffer[idx : idx + n_step] for idx in indices]

        states = tf.concat(
            [traj[0].state for traj in trajectories], axis=0
        )  # (B, H, W, 4)
        actions = tf.convert_to_tensor(
            [[traj[0].action] for traj in trajectories], dtype=tf.int64
        )  # (B, 1)

        rewards, is_dones = [], []
        for trajectory in trajectories:
            reward, is_done = 0, 0
            for i, transition in enumerate(trajectory, start=0):
                reward += transition.reward * gamma**i
                is_done = max(is_done, transition.is_done)
                if is_done == 1:
                    break
            rewards.append([reward])
            is_dones.append([is_done])

        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        is_dones = tf.convert_to_tensor(is_dones, dtype=tf.float32)

        next_states = tf.concat(
            [traj[-1].state for traj in trajectories], axis=0
        )  # (B, H, W, 4)

        return (states, actions, rewards, is_dones, next_states)
