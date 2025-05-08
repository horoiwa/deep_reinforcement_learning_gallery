import random
import tensorflow as tf
from dataclasses import dataclass

import numpy as np


@dataclass
class Experience:
    observation: np.ndarray
    action: int
    reward: float
    done: float


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

        obs, actions, rewards, dones, masks = [], [], [], [], []
        for trajectory in trajectories:
            done = False
            _obs, _actions, _rewards, _dones, _masks = [], [], [], [], []
            for t in trajectory:
                if done:
                    # If the trajectory is done, we need to pad the observation and action
                    # to match the unroll steps
                    _obs.append(tf.zeros_like(t.observation, dtype=tf.float32))
                    _actions.append(0)
                    _rewards.append(0.0)
                    _dones.append(1.0)
                    _masks.append(0.0)
                else:
                    _obs.append(tf.convert_to_tensor(t.observation, dtype=tf.float32))
                    _actions.append(t.action)
                    _rewards.append(t.reward)
                    _dones.append(t.done)
                    _masks.append(1.0)

                if t.done:
                    done = True

            obs.append(tf.concat(_obs, axis=0))
            actions.append(np.array(_actions, dtype=np.int32))
            rewards.append(np.array(_rewards, dtype=np.float32))
            dones.append(np.array(_dones, dtype=np.float32))
            masks.append(np.array(_masks, dtype=np.float32))

        obs = tf.stack(obs)  # (B, unroll_steps+1, 96, 96, 1)

        observations = obs[:, :-1, ...]  # (B, unroll_steps, 96, 96, 1)
        next_observations = obs[:, 1:, ...]  # (B, unroll_steps, 96, 96, 1)
        actions = tf.stack(actions, axis=0)[:, :unroll_steps]  # (B, unroll_steps)
        rewards = tf.stack(rewards, axis=0)[:, :unroll_steps]  # (B, unroll_steps)
        dones = tf.stack(dones, axis=0)[:, :unroll_steps]  # (B, unroll_steps)
        masks = tf.stack(masks, axis=0)[:, :unroll_steps]  # (B, unroll_steps)

        return (
            observations,
            next_observations,
            actions,
            rewards,
            dones,
            masks,
        )
