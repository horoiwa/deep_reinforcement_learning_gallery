import time
import random

import tensorflow as tf
import numpy as np
from PIL import Image


class Timer:

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        fin = time.time() - self.start
        print(self.name, fin)


def preprocess_frame(frame):
    """Breakout only"""
    image = Image.fromarray(frame)
    image = image.convert("L").crop((0, 34, 160, 200)).resize((84, 84))
    image_scaled = np.array(image) / 255.0
    return image_scaled.astype(np.float32)


def huber_loss(target_q, q, d=1.0):
    """
    See https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/losses.py#L1098-L1162
    """
    td_error = target_q - q
    is_smaller_than_d = tf.abs(td_error) < d
    squared_loss = 0.5 * tf.square(td_error)
    linear_loss = 0.5 * d ** 2 + d * (tf.abs(td_error) - d)
    loss = tf.where(is_smaller_than_d, squared_loss, linear_loss)
    return loss


class SumTree:
    """ See https://github.com/ray-project/ray/blob/master/rllib/execution/segment_tree.py
    """

    def __init__(self, capacity: int):
        #: 2のべき乗チェック
        assert capacity & (capacity - 1) == 0
        self.capacity = capacity
        self.values = [0 for _ in range(2 * capacity)]

    def __str__(self):
        return str(self.values[self.capacity:])

    def __setitem__(self, idx, val):
        idx = idx + self.capacity
        self.values[idx] = val

        current_idx = idx // 2
        while current_idx >= 1:
            idx_lchild = 2 * current_idx
            idx_rchild = 2 * current_idx + 1
            self.values[current_idx] = self.values[idx_lchild] + self.values[idx_rchild]
            current_idx //= 2

    def __getitem__(self, idx):
        idx = idx + self.capacity
        return self.values[idx]

    def sum(self):
        return self.values[1]

    def sample(self, z=None):
        z = random.uniform(0, self.sum()) if z is None else z
        assert 0 <= z <= self.sum()

        current_idx = 1
        while current_idx < self.capacity:

            idx_lchild = 2 * current_idx
            idx_rchild = 2 * current_idx + 1

            #: 左子ノードよりzが大きい場合は右子ノードへ
            if z > self.values[idx_lchild]:
                current_idx = idx_rchild
                z = z -self.values[idx_lchild]
            else:
                current_idx = idx_lchild

        #: 見かけ上のインデックスにもどす
        idx = current_idx - self.capacity
        return idx


if __name__ == "__main__":
    sumtree = SumTree(capacity=4)
    sumtree[0] = 4
    sumtree[1] = 1
    sumtree[2] = 2
    sumtree[3] = 3
    samples = [sumtree.sample() for _ in range(1000)]

    print(samples.count(0))
    print(samples.count(1))
    print(samples.count(2))
    print(samples.count(3))
