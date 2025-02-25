import random
import tensorflow as tf
from dataclasses import dataclass

import numpy as np


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    done: int


class ReplayBuffer:
    def __init__(self, maxlen: int | None = None):
        self.maxlen = maxlen
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def add(self, trajectory: list[Experience]):
        self.buffer.extend(trajectory)
        if self.maxlen is not None and len(self.buffer) > self.maxlen:
            self.buffer = self.buffer[-self.maxlen :]

    def sample_batch(
        self, batch_size: int, unroll_steps: int, td_steps: int, gamma: float
    ):
        indices = [
            random.randint(0, len(self.buffer) - td_steps - unroll_steps - 1)
            for _ in range(batch_size)
        ]
        trajectories = [
            self.buffer[idx : idx + td_steps + unroll_steps + 1] for idx in indices
        ]

        states = []
        actions = []
        value_prefixes = []
        dones = []

        for trajectory in trajectories:
            is_done = False
            for i, transition in enumerate(trajectory):
                if not is_done:
                    is_done = bool(transition.done)
                    pass

        return states, actions, value_prefixes, dones
