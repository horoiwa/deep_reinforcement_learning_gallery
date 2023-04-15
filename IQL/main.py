from pathlib import Path

import tensorflow as tf



class IQL:
    def __init__(self):
        pass

    def setup(self):
        pass


def load_dataset(dataset_path: str, batch_size: int):

    def deserialize(serialized_transition):

        transition = tf.io.parse_single_example(
            serialized_transition,
            features={
                'state': tf.io.FixedLenFeature([], tf.string),
                'action': tf.io.FixedLenFeature([], tf.string),
                'reward': tf.io.FixedLenFeature([], tf.float32),
                'next_state': tf.io.FixedLenFeature([], tf.string),
                'done': tf.io.FixedLenFeature([], tf.float32),
            }
        )

        a = tf.io.decode_raw(transition["action"], tf.float32)
        r = transition["reward"]
        d = transition["done"]
        s = tf.io.decode_raw(transition["state"], tf.float32)
        s2 = tf.io.decode_raw(transition["next_state"], tf.float32)

        return s, a, r, s2, d

    dataset = (
        tf.data.TFRecordDataset(filenames=dataset_path, num_parallel_reads=tf.data.AUTOTUNE)
               .shuffle(256, reshuffle_each_iteration=True)
               .repeat()
               .map(deserialize, num_parallel_calls=tf.data.AUTOTUNE)
               .batch(batch_size, drop_remainder=True)
               .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


def main():
    tf_dataset = load_dataset(dataset_path="bipedalwalker.tfrecord", batch_size=32)
    for minibatch in tf_dataset:
        s, a, r, s2, d = minibatch
        import pdb; pdb.set_trace()



if __name__ == '__main__':
    main()
