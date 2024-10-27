from dataclasses import dataclass
import collections
import copy
import random
from typing import List

import numpy as np
import tensorflow as tf
import pickle
import lz4.frame as lz4f


@dataclass
class Experience:

    obs: tf.float32
    action: tf.float32
    reward: tf.float32
    next_obs: tf.float32
    done: tf.float32
    prev_z: tf.float32
    prev_h: tf.float32
    prev_a: tf.float32


class SequenceReplayBuffer:

    def __init__(self, buffer_size, seq_len, batch_size, action_space):

        self.buffer_size = buffer_size

        self.L = seq_len

        self.buffer = collections.deque(maxlen=self.buffer_size)

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

        dataset.prefetch(10)

        self.dataset = iter(dataset)

    def generate_sequence(self):

        while True:

            selected_idx = random.randint(0, len(self.buffer)-1)
            sequence = pickle.loads(lz4f.decompress(self.buffer[selected_idx]))

            #: prev_h, prev_zはsequenceの先頭だけでOK
            sequence = {
                "obs": tf.concat([e.obs for e in sequence], axis=0),
                "action": tf.concat([e.action for e in sequence], axis=0),
                "reward": tf.stack([[e.reward] for e in sequence], axis=0),
                "next_obs": tf.concat([e.next_obs for e in sequence], axis=0),
                "done": tf.stack([[e.done] for e in sequence], axis=0),
                "prev_z": sequence[0].prev_z,
                "prev_h": sequence[0].prev_h,
                "prev_a": sequence[0].prev_a,
            }

            yield sequence

    def add_sequences(self, sequences):

        for seq in sequences:
            self.buffer.append(seq)

    def get_minibatch(self):

        if not hasattr(self, "dataset"):
            print("Create TFDataset")
            self.create_tfdataset()

        minibatch = next(self.dataset)

        #: (batchsize, timesteps, ...) -> (timesteps, batchsize, ...)
        minibatch = {key: tf.einsum("ij... -> ji...", data)
                     for key, data in minibatch.items()}

        return minibatch


class EpisodeBuffer:

    def __init__(self, seqlen):

        self.buffer = []

        self.seqlen = seqlen

    def add(self, obs: np.array, action_onehot, reward, next_obs, done: bool,
            prev_z, prev_h, prev_a_onehot):
        """
        Note:
            RSSMの特性上、(st, at, rt, dt)ではなく,
            (rt-1, dt-1, st, at)を遷移の単位とした方が処理が楽

        Note:
            Assumed that transition information is sent from single env.
        """

        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        action = tf.convert_to_tensor(action_onehot, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_obs = tf.convert_to_tensor(next_obs, dtype=tf.float32)
        done = tf.convert_to_tensor(done, dtype=tf.float32)

        exp = Experience(
            obs, action, reward, next_obs, done,
            prev_z, prev_h, prev_a_onehot)

        self.buffer.append(exp)

    def get_sequences(self):

        assert self.buffer[-1].done

        sequences = []

        for t in range(0, len(self.buffer), self.seqlen):
            if (t + self.seqlen) > len(self.buffer):
                t = len(self.buffer) - self.seqlen
            seq = self.buffer[t:t+self.seqlen]
            seq = lz4f.compress(pickle.dumps(seq))
            sequences.append(seq)

        self.buffer = []

        return sequences


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
