from dataclasses import dataclass
import collections
import random

import numpy as np
import tensorflow as tf
import pickle
import lz4.frame as lz4f


@dataclass
class Experience:

    obs: tf.float32
    action: tf.float32
    reward: tf.float32
    is_first: tf.uint8
    is_done: tf.uint8


class SequenceReplayBuffer:

    def __init__(self, buffer_size, seq_len, batch_size, action_space):

        self.buffer_size = buffer_size

        self.L = seq_len

        self.buffer = collections.deque(maxlen=self.buffer_size+self.L)

        self.batch_size = batch_size

        self.action_space = action_space

    def __len__(self):
        return len(self.buffer)

    def create_tfdataset(self):

        example_seq = next(self.generate_sequence())

        dataset = tf.data.Dataset.from_generator(
            self.generate_sequence,
            output_types={k: v.dtype for k, v in example_seq.items()},
            output_shapes={k: v.shape for k, v in example_seq.items()},
        )

        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset.prefetch(5)

        self.dataset = iter(dataset)

    def generate_sequence(self):

        while True:

            selected_idx = random.randint(self.L, len(self.buffer)-self.L)

            sequence = [pickle.loads(lz4f.decompress(self.buffer[idx]))
                        for idx in range(selected_idx-self.L, selected_idx+self.L)]

            #: sequence shold not cross episode
            seqlen = len(sequence)
            for i in range(seqlen//2, seqlen):
                if sequence[i].is_done:
                    sequence = sequence[i+1-self.L:i+1]
                    assert sequence[-1].is_done, "Error #01"
                    break
            else:
                sequence = sequence[-self.L:]

            assert len(sequence) == self.L, "Error #02"
            assert np.all([not e.is_done for e in sequence[:-1]]), "Error #03"

            sequence = {
                "obs": tf.stack([e.obs for e in sequence]),   #: (self.L, 64, 64, 1)
                "action": tf.stack([e.action for e in sequence]),
                "reward": tf.stack([e.reward for e in sequence]),
                "is_first": tf.stack([e.is_first for e in sequence]),
                "is_done": tf.stack([e.is_done for e in sequence]),
            }

            yield sequence

    def add(self, obs: np.array, action_onehot: int, reward: int,
            is_first: bool, is_done: bool):

        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        action = tf.convert_to_tensor(action_onehot, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        is_first = tf.convert_to_tensor(is_first, dtype=tf.float32)
        is_done = tf.convert_to_tensor(is_done, dtype=tf.float32)

        exp = Experience(obs, action, reward, is_first, is_done)
        exp = lz4f.compress(pickle.dumps(exp))

        self.buffer.append(exp)

    def get_minibatch(self):

        if not hasattr(self, "dataset"):
            print("Create TfDataset")
            self.create_tfdataset()

        return next(self.dataset)


if __name__ == "__main__":
    replay_buffer = SequenceReplayBuffer(1024, 50, 50, action_space=4)

    for i in range(2000):
        obs = np.zeros((64, 64, 1))
        action = np.random.randint(0, 4)
        reward = 1.0
        done = True if i % 200 == 0 else False
        replay_buffer.add(obs, action, reward, False, done)

    for i in range(10):
        print(i)
        minibatch = replay_buffer.get_minibatch()
