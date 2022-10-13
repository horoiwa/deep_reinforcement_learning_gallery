from pathlib import Path
import random
import numpy as np
import shutil
import gc

import tensorflow as tf
from tqdm import tqdm
from dopamine.replay_memory.circular_replay_buffer import OutOfGraphReplayBuffer


class TmpOutOfGraphReplayBuffer(OutOfGraphReplayBuffer):

    def to_tfrecords(self, cache_dir, suffix, num_chunks=4) -> tf.data.Dataset:

        indices = [idx for idx in range(0, self.cursor()) if self.is_valid_transition(idx)]
        random.shuffle(indices)

        for i, _indices in enumerate(np.array_split(indices, num_chunks)):
            batch_size = len(_indices)

            #: state: (dtype, shape) == (np.uint8, (84,84,4))
            transitions = self.sample_transition_batch(batch_size=batch_size, indices=_indices)
            states, actions, rewards, next_states, _, _, dones, _ = transitions

            filepath = str(cache_dir / f"DQN_{suffix}_{i}.tfrecords")
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


def create_tfrecords(original_dataset_dir: str, dataset_dir: str, num_data_files: int, use_samples_per_file: int):
    """ Convert DQN replay dataset to tfrecords
    Args:
        original_dataset_dir (str): Path to original dqn-replay-dataset directory
        dataset_dir (str): Output directory
        num_data_files (int): 50分割されたファイルのうちいくつまで使うか(max=50)
        use_samples_per_file (int): １ファイルから何サンプルを使うか (max=1000000)

    Note:
        1. DQN Replayデータセット(50M)は1Mごとに50分割されていることに留意

        2. Download dataset in advance
          mkdir dqn-replay-dataset && cd ./dqn-replay-dataset
          gsutil -m cp -R gs://atari-replay-datasets/dqn/BreakOut .
    """

    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True)

    print(f"==== Create tfrecords from {original_dataset_dir} ====")
    print(f"==== Output: {dataset_dir} ====")

    for i in range(0, num_data_files):

        #: ここのbatch_sizeやgammaはダミーなので何でもいい
        tmp_buffer = TmpOutOfGraphReplayBuffer(
            replay_capacity=use_samples_per_file,
            observation_shape=(84, 84),
            stack_size=4,
            batch_size=64,
            update_horizon=1,
            gamma=0.99)

        tmp_buffer.load(original_dataset_dir, suffix=f"{i}")
        tmp_buffer.to_tfrecords(dataset_dir, suffix=i)

        del tmp_buffer
        gc.collect()


class OfflineBuffer:

    def __init__(self, dataset_dir: str, batch_size: int):
        """
        Args:
            dataset_dir (str): tfrecords dir
        """

        self.dataset_dir = Path(dataset_dir)

        self.batch_size = batch_size

        self.dataset = self.load_dataset()

    def load_dataset(self):

        assert self.dataset_dir.exists()

        tfrecords_files = [str(pt) for pt in self.dataset_dir.glob("*.tfrecords")]
        print(f"Detected tfrecords: {tfrecords_files}")

        random.shuffle(tfrecords_files)  #: shuffleが意味あるかは知らない

        dataset = tf.data.TFRecordDataset(tfrecords_files)

        dataset = (
            dataset.shuffle(1000, reshuffle_each_iteration=True)
                   .repeat()
                   .batch(self.batch_size, drop_remainder=True)
                   .map(deserialize)
                   .prefetch(4)
        )

        return dataset

    def _sample_minibatch(self):
        for minibatch in self.dataset:
            yield minibatch

    def sample_minibatch(self):
        minibatch = next(self._sample_minibatch())
        return minibatch
