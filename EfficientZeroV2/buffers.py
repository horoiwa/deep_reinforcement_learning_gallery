import random
import tensorflow as tf
from dataclasses import dataclass

import numpy as np


@dataclass
class Experience:
    observation: np.ndarray
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

    def sample_batch(self, batch_size: int, unroll_steps: int):
        indices = [
            random.randint(0, len(self.buffer) - unroll_steps - 1)
            for _ in range(batch_size)
        ]
        trajectories = [self.buffer[idx : idx + unroll_steps + 1] for idx in indices]

        observations = []
        actions = []
        rewards = []
        masks = []

        for trajectory in trajectories:
            _masks = 1 - np.cumsum([t.done for t in trajectory]).clip(0, 1)
            _observations = (
                np.concatenate([t.observation for t in trajectory], axis=0)
                * _masks[..., np.newaxis, np.newaxis, np.newaxis]
            )
            _actions = (np.array([t.action for t in trajectory])).reshape(
                -1, 1
            ) * _masks[..., np.newaxis]
            _rewards = (
                np.array([t.reward for t in trajectory]).reshape(-1, 1)
                * _masks[..., np.newaxis]
            )

            observations.append(_observations)
            actions.append(_actions)
            rewards.append(_rewards)
            masks.append(_masks)

        # ここでsegmentation fault
        observations = tf.convert_to_tensor(
            np.array(observations), dtype=tf.float32
        )  # (B, unroll_steps+1, 96, 96, 1)
        actions = tf.stack(actions, axis=0)  # (B, unroll_steps+1)
        rewards = tf.stack(rewards, axis=0)  # (B, unroll_steps+1)
        masks = tf.cast(tf.stack(masks, axis=0), tf.float32)  # (B, unroll_steps+1)

        init_obs = observations[:, 0, ...]  # (B, 96, 96, 1)
        init_action = actions[:, 0, ...]  # (B, 1)

        return (
            init_obs,
            init_action,
            observations[:, 1:, ...],
            tf.squeeze(actions[:, 1:, ...], axis=-1),
            tf.squeeze(rewards[:, 1:, ...], axis=-1),
            masks[:, 1:],
        )
