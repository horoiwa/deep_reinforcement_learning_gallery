from pathlib import Path
import random
import numpy as np
import shutil
import gc

import tensorflow as tf
from tqdm import tqdm
from dopamine.replay_memory.circular_replay_buffer import OutOfGraphReplayBuffer


class TmpOutOfGraphReplayBuffer(OutOfGraphReplayBuffer):

    def to_tfrecords(self, cache_dir, suffix, num_chunks=25):

        indices = [idx for idx in range(0, self.cursor()) if self.is_valid_transition(idx)]
        random.shuffle(indices)
        print(len(indices))

        for i, _indices in enumerate(np.array_split(indices, num_chunks)):
            batch_size = len(_indices)

            #: state: (dtype, shape) == (np.uint8, (84,84,4))
            transitions = self.sample_transition_batch(batch_size=batch_size, indices=_indices)
            states, actions, rewards, next_states, next_actions, next_rewards, dones, indices = transitions
            dones = dones.astype(np.float32)
            print("debug", dones.sum())
            filepath = str(cache_dir / f"DQN_{suffix}_{i}.tfrecords")
            with tf.io.TFRecordWriter(filepath) as writer:
                for s, a, r, s2, d in tqdm(zip(states, actions, rewards, next_states, dones)):
                    record = tf.train.Example(
                        features=tf.train.Features(feature={
                            "state": tf.train.Feature(bytes_list=tf.train.BytesList(value=[s.tostring()])),
                            "action": tf.train.Feature(int64_list=tf.train.Int64List(value=[a])),
                            "reward": tf.train.Feature(float_list=tf.train.FloatList(value=[r])),
                            "next_state": tf.train.Feature(bytes_list=tf.train.BytesList(value=[s2.tostring()])),
                            "done": tf.train.Feature(float_list=tf.train.FloatList(value=[d])),
                        }))
                    writer.write(record.SerializeToString())


def deserialize(serialized_transition):

    transition = tf.io.parse_single_example(
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
    s = tf.reshape(
        tf.cast(tf.io.decode_raw(transition["state"], tf.uint8), tf.float32),
        (84, 84, 4))
    s2 = tf.reshape(
        tf.cast(tf.io.decode_raw(transition["next_state"], tf.uint8), tf.float32),
        (84, 84, 4))
    return s, a, r, s2, d


def create_tfrecords(original_dataset_dir: str, dataset_dir: str, num_data_files: int, use_samples_per_file: int, num_chunks=10):
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
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
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
        tmp_buffer.to_tfrecords(dataset_dir, suffix=i, num_chunks=num_chunks)

        del tmp_buffer
        gc.collect()


def load_dataset(dataset_dir: Path, batch_size: int):
    """ Replay buffer with TFrecords"""

    dataset_dir = Path(dataset_dir)
    assert dataset_dir.exists()

    tfrecords_files = [str(pt) for pt in dataset_dir.glob("*.tfrecords")]
    print(f"Detected tfrecords: {tfrecords_files}")

    random.shuffle(tfrecords_files)  #: shuffleが意味あるかは知らない

    dataset = (
        tf.data.TFRecordDataset(filenames=tfrecords_files, num_parallel_reads=tf.data.AUTOTUNE)
               .shuffle(256, reshuffle_each_iteration=True)
               .repeat()
               .map(deserialize, num_parallel_calls=tf.data.AUTOTUNE)
               .batch(batch_size, drop_remainder=False)
               .prefetch(tf.data.AUTOTUNE)
    )
    return dataset
