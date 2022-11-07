import random

import pickle
import lz4.frame as lz4f
import numpy as np
import tensorflow as tf
from dopamine.replay_memory.circular_replay_buffer import OutOfGraphReplayBuffer


class SequenceReplayBuffer:

    def __init__(self, dataset_dir, num_data_files=50, samples_per_file=10000, context_length=30):

        self.context_length = context_length

        # rtgs: returns to go or R
        self.rtgs, self.states, self.actions, self.dones, self.timesteps = [], [], [], [], []

        self.load(dataset_dir, samples_per_file, num_data_files)

        self.terminal_indices = [i for i, done in enumerate(self.dones) if done == 1]

        assert len(self.rtgs) == len(self.states) == len(self.actions) == len(self.timesteps)

    def __len__(self):
        return len(self.rtgs) - self.context_length

    def load(self, dataset_dir, samples_per_file, num_data_files):

        for i in range(0, num_data_files):
            buffer = OutOfGraphReplayBuffer(
                replay_capacity=samples_per_file,
                observation_shape=(84, 84),
                stack_size=4,
                batch_size=64,
                update_horizon=1,
                gamma=0.99)
            buffer.load(dataset_dir, suffix=f"{i}")

            indices = [idx for idx in range(0, buffer.cursor()) if buffer.is_valid_transition(idx)]
            transitions = buffer.sample_transition_batch(batch_size=len(indices), indices=indices)
            self.add_transitions(transitions)

            del buffer

    def add_transitions(self, transitions):
        """
        Note:
            Decision transofomerはエピソード横断しないので余剰分はトリミング
        """
        states, actions, rewards, _, _, _, dones, _ = transitions

        _terminal_indices = np.argwhere(dones == 1).flatten().tolist()
        start_idx, terminal_indices = _terminal_indices[0]+1, _terminal_indices[1:]

        for terminal_idx in terminal_indices:

            _rewards = [rewards[i] for i in range(start_idx, terminal_idx+1)]
            self.rtgs += [sum(_rewards) - sum(_rewards[:i+1]) for i in range(len(_rewards))]
            self.timesteps += [i for i in range(len(_rewards))]

            self.states += [lz4f.compress(pickle.dumps(states[i])) for i in range(start_idx, terminal_idx+1)]
            self.actions += [actions[i] for i in range(start_idx, terminal_idx+1)]
            self.dones += [dones[i] for i in range(start_idx, terminal_idx+1)]

            start_idx = terminal_idx + 1

    def sample_sequence(self):

        while True:
            start_idx = random.randint(0, len(self))
            terminal_idx = next(filter(lambda v: v >= start_idx, self.terminal_indices))

            if terminal_idx - start_idx < self.context_length:
                start_idx, end_idx = terminal_idx - self.context_length, terminal_idx
            else:
                end_idx = start_idx + self.context_length

            #: states shape: (context_length, 84, 84, 4)
            sequence = {
                "rtgs": tf.cast(
                    tf.stack([[self.rtgs[idx]] for idx in range(start_idx, end_idx)], axis=0), tf.float32),
                "states": tf.cast(
                    tf.stack([pickle.loads(lz4f.decompress(self.states[idx])) for idx in range(start_idx, end_idx)], axis=0),
                    tf.float32),
                "actions": tf.cast(
                    tf.stack([[self.actions[idx]] for idx in range(start_idx, end_idx)], axis=0), tf.float32),
                "timesteps": tf.cast(
                    tf.stack([[self.timesteps[idx]] for idx in range(start_idx, end_idx)], axis=0), tf.float32)
            }

            yield sequence


def create_dataset(dataset_dir, context_length):

    buffer = SequenceReplayBuffer(dataset_dir=dataset_dir, num_data_files=1, context_length=context_length)

    example_seq = next(buffer.sample_sequence())

    dataset = tf.data.Dataset.from_generator(
        buffer.sample_sequence,
        output_types={k: v.dtype for k, v in example_seq.items()},
        output_shapes={k: v.shape for k, v in example_seq.items()}
    ).batch(48)

    return dataset


def speedtest(dataset):
    import time
    start = time.time()
    i = 0
    for _ in dataset:
        print(i, time.time() - start)
        start = time.time()
        i += 1
        if i == 500:
            break


if __name__ == "__main__":
    dataset_dir = "/mnt/disks/data/Breakout/1/replay_logs"
    context_length = 30
    dataset = create_dataset(dataset_dir, context_length)
    speedtest(dataset)
