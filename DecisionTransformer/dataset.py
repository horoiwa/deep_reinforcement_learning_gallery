import random
import time
from typing import List
import collections

import ray
import pickle
import lz4.frame as lz4f
import numpy as np
import tensorflow as tf
from dopamine.replay_memory.circular_replay_buffer import OutOfGraphReplayBuffer


@ray.remote(num_cpus=1, num_gpus=0)
class SequenceLoader:

    def __init__(self, pid, max_timestep: int, context_length=30):

        self.pid = pid

        self.context_length = context_length

        self.max_timestep = max_timestep

        # rtgs: returns to go or R
        self.rtgs, self.states, self.actions, self.dones, self.timesteps = [], [], [], [], []

        self.terminal_indices = None

        assert len(self.rtgs) == len(self.states) == len(self.actions) == len(self.timesteps)

    def __len__(self):
        return len(self.rtgs) - self.context_length

    def load(self, dataset_dir, samples_per_file, datafile_suffixes):

        for i in datafile_suffixes:
            tmp_buffer = OutOfGraphReplayBuffer(
                replay_capacity=samples_per_file,
                observation_shape=(84, 84),
                stack_size=4,
                batch_size=64,
                update_horizon=1,
                gamma=0.99)
            tmp_buffer.load(dataset_dir, suffix=f"{i}")

            indices = [idx for idx in range(0, tmp_buffer.cursor()) if tmp_buffer.is_valid_transition(idx)]
            transitions = tmp_buffer.sample_transition_batch(batch_size=len(indices), indices=indices)
            self.add_transitions(transitions)
            del tmp_buffer

        self.terminal_indices = [i for i, done in enumerate(self.dones) if done == 1]

        return "OK"

    def add_transitions(self, transitions):
        """
        Note:
            Decision transofomerはエピソード横断しないので余剰分はトリミング
        """
        states, actions, rewards, _, _, _, dones, indices = transitions
        terminal_indices = np.argwhere(dones == 1).flatten().tolist()

        for start_idx, terminal_idx in zip(terminal_indices[:-1], terminal_indices[1:]):

            start_idx = start_idx + 1

            #: 1エピソードが長すぎる場合は不毛なのでtruncate
            is_truncated = False
            if (terminal_idx - start_idx) >= self.max_timestep:
                is_truncated = True
                terminal_idx = start_idx + self.max_timestep - 100

            _rewards = [rewards[i] for i in range(start_idx, terminal_idx+1)]
            self.rtgs += [sum(_rewards) - sum(_rewards[:i+1]) for i in range(len(_rewards))]
            self.timesteps += [i for i in range(len(_rewards))]
            self.states += [lz4f.compress(pickle.dumps(states[i])) for i in range(start_idx, terminal_idx+1)]
            self.actions += [actions[i] for i in range(start_idx, terminal_idx+1)]

            _dones = [dones[i] for i in range(start_idx, terminal_idx+1)]
            if is_truncated:
                _dones[-1] = 1
            self.dones += _dones

        assert self.dones[-1] == 1
        assert max(rewards) <= 1.0, f"MAX {max(rewards)}"

    def sample_sequences(self, num_sample=64) -> List:

        sequences = []

        for _ in range(num_sample):
            start_idx = random.randint(0, len(self)-1)

            terminal_idx = next(filter(lambda v: v >= start_idx, self.terminal_indices))

            if terminal_idx - start_idx < self.context_length:
                start_idx, end_idx = terminal_idx - self.context_length, terminal_idx
            else:
                end_idx = start_idx + self.context_length

            #: states shape: (context_length, 84, 84, 4)
            rtgs = tf.cast(
                tf.stack([[self.rtgs[idx]] for idx in range(start_idx, end_idx)], axis=0),
                tf.float32)

            states = tf.cast(
                tf.stack([pickle.loads(lz4f.decompress(self.states[idx])) for idx in range(start_idx, end_idx)], axis=0),
                tf.float32)

            actions = tf.cast(
                tf.stack([[self.actions[idx]] for idx in range(start_idx, end_idx)], axis=0),
                tf.uint8)

            timesteps = tf.cast(
                tf.stack([[self.timesteps[start_idx]]], axis=0),
                tf.int32)

            seq = (rtgs, states, actions, timesteps)
            seq = lz4f.compress(pickle.dumps(seq))
            sequences.append(seq)

        return self.pid, sequences



class SequenceBuffer:

    def __init__(self, maxlen, batch_size):
        self.buffer = collections.deque(maxlen=maxlen)
        self.batch_size = batch_size
        self.dataset = None

    def __len__(self):
        return len(self.buffer)

    def add_sequences(self, sequences: list):
        for seq in sequences:
            self.buffer.append(seq)

    def sample_sequence(self):
        selected_idx = random.randint(0, len(self.buffer)-1)
        rtgs, states, actions, timesteps = pickle.loads(lz4f.decompress(self.buffer[selected_idx]))
        return (rtgs, states, actions, timesteps)

    def sample_minibatch(self):
        rtgs, states, actions, timesteps = zip(*[self.sample_sequence() for _ in range(self.batch_size)])

        rtgs = tf.stack(rtgs, axis=0)
        states = tf.stack(states, axis=0)
        actions = tf.stack(actions, axis=0)
        timesteps = tf.stack(timesteps, axis=0)

        return (rtgs, states, actions, timesteps)


def create_dataloaders(dataset_dir, max_timestep, num_data_files=50, samples_per_file=10_000,
                       context_length=30, batch_size=128, num_parallel_calls=1):

    assert num_data_files >= num_parallel_calls

    datafile_suffixes = [i for i in range(num_data_files)]

    loaders = []
    for pid, _suffixes in enumerate(np.array_split(datafile_suffixes, num_parallel_calls)):
        loader = SequenceLoader.remote(
            pid=pid, max_timestep=max_timestep, context_length=context_length)

        ret = loader.load.remote(dataset_dir, samples_per_file, _suffixes)
        print(pid, ray.get(ret))

        loaders.append(loader)

    return loaders
