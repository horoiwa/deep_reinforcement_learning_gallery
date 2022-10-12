from pathlib import Path
import random
import numpy as np
import shutil

import tensorflow as tf
from tqdm import tqdm
from dopamine.replay_memory.circular_replay_buffer import OutOfGraphReplayBuffer


class TmpOutOfGraphReplayBuffer(OutOfGraphReplayBuffer):

    def save_to_tfrecords(self, cache_dir, suffix, num_chunks=4) -> tf.data.Dataset:

        indices = [idx for idx in range(0, self.cursor()) if self.is_valid_transition(idx)]
        random.shuffle(indices)

        for i, _indices in enumerate(np.array_split(indices, num_chunks)):
            batch_size = len(_indices)

            #: state: (dtype, shape) == (np.uint8, (84,84,4))
            transitions = self.sample_transition_batch(batch_size=batch_size, indices=_indices)
            states, actions, rewards, next_states, _, _, dones, _ = transitions

            filepath = str(cache_dir / f"dataset_{suffix}_{i}.tfrecords")
            with tf.io.TFRecordWriter(filepath) as writer:
                for s, a, r, s2, d in tqdm(zip(states, actions, rewards, next_states, dones)):
                    record = tf.train.Example(
                        features=tf.train.Features(feature={
                            "state": tf.train.Feature(bytes_list=tf.train.BytesList(value=[s.tostring()])),
                            "action": tf.train.Feature(int64_list=tf.train.Int64List(value=[a])),
                            "reward": tf.train.Feature(float_list=tf.train.FloatList(value=[r])),
                            "next_state": tf.train.Feature(bytes_list=tf.train.BytesList(value=[s2.tostring()])),
                            "done": tf.train.Feature(float_list=tf.train.FloatList(value=[float(d)])),
                        }))
                    writer.write(record.SerializeToString())


def deserialize(serialized_transition):

    transition = tf.io.parse_example(
        serialized_transition,
        features={
            'state': tf.io.FixedLenFeature([], tf.string),
            'action': tf.io.FixedLenFeature([], tf.int64),
            'reward': tf.io.FixedLenFeature([], tf.float32),
            'next_state': tf.io.FixedLenFeature([], tf.string),
            'done': tf.io.FixedLenFeature([], tf.float32),
        }
    )

    a = transition["action"]
    r = transition["reward"]
    d = transition["done"]

    batch_size = a.shape[0]
    s = tf.reshape(
        tf.cast(tf.io.decode_raw(transition["state"], tf.uint8), tf.float32),
        (batch_size, 84, 84, 4))
    s2 = tf.reshape(
        tf.cast(tf.io.decode_raw(transition["next_state"], tf.uint8), tf.float32),
        (batch_size, 84, 84, 4))

    return s, a, r, s2, d


class OfflineBuffer:

    def __init__(self, dataset_dir, num_data_files=50, cache_dir="tmp/",
                 capacity_of_each_buffer=100000, batch_size=32):
        """
        DQN Replayデータセット(50M transition)は1M transitionごとに分割されていることに留意

        Args:
            dataset_dir (str): path to dqn-replay-dataset (50M遷移 = 1M * 50 files)
            num_data_files(int): 50分割されたデータセットの何番目まで使うか, set 50 for CQL paper
            capacity_of_each_buffer (int): 各データセットファイルからどれだけのサンプルを使うか. Max 1,000,000 = 1M transition.

        Note:
            #: Download dataset in advance
            mkdir dqn-replay-dataset && cd ./dqn-replay-dataset
            gsutil -m cp -R gs://atari-replay-datasets/dqn/BreakOut .
        """

        self.dataset_dir = Path(dataset_dir)

        self.cache_dir = Path(cache_dir)

        self.num_data_files = num_data_files

        self.capacity = capacity_of_each_buffer

        self.batch_size = batch_size

        self.dataset = self.load_dataset()

    def load_dataset(self):

        assert self.dataset_dir.exists()

        if not self.cache_dir.exists():
            print(f"==== Create tfrecords in {self.cache_dir} ====")
            self.cache_dir.mkdir()
            for i in range(0, self.num_data_files):

                tmp_buffer = TmpOutOfGraphReplayBuffer(
                    replay_capacity=self.capacity,
                    observation_shape=(84, 84),
                    stack_size=4,
                    batch_size=self.batch_size,
                    update_horizon=1,
                    gamma=0.99)

                tmp_buffer.load(self.dataset_dir, suffix=f"{i}")
                tmp_buffer.save_to_tfrecords(self.cache_dir, suffix=i)
        else:
            print(f"==== Use tfrecords in {self.cache_dir} ====")

        tfrecords_files = [str(pt) for pt in self.cache_dir.glob("*.tfrecords")]
        random.shuffle(tfrecords_files)

        dataset = tf.data.TFRecordDataset(tfrecords_files)

        dataset = (
            dataset.shuffle(1000, reshuffle_each_iteration=True)
                   .repeat()
                   .batch(self.batch_size, drop_remainder=True)
                   .map(deserialize, num_parallel_calls=tf.data.AUTOTUNE)
                   .prefetch(tf.data.AUTOTUNE)
        )

        return dataset

    def _sample_minibatch(self):
        for minibatch in self.dataset:
            yield minibatch

    def sample_minibatch(self):
        minibatch = next(self._sample_minibatch())
        return minibatch


if __name__ == '__main__':
    dataset_dir = "dqn-replay-dataset/Breakout/1/replay_logs"
    buffer = OfflineBuffer(dataset_dir=dataset_dir)
    buffer.sample_minibatch()
