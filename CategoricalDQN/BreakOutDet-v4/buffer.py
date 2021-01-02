from dataclasses import dataclass
import ray

import numpy as np
import pickle
import zlib


@dataclass
class Experience:

    state: np.ndarray

    action: float

    reward: float

    next_state: np.ndarray

    done: bool


class ReplayBuffer:
    """わかりやすさのためにRAMを無駄遣いする実装なのでせめて圧縮する
    """

    def __init__(self, max_len):

        self.max_len = max_len

        self.buffer = []

        self.count = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, exp):

        exp = zlib.compress(pickle.dumps(exp))

        if self.count == self.max_len:
            self.count = 0

        try:
            self.buffer[self.count] = exp
        except IndexError:
            self.buffer.append(exp)

        self.count += 1

    def get_minibatch(self, batch_size):

        N = len(self.buffer)

        indices = np.random.choice(
            np.arange(N), replace=False, size=batch_size)

        selected_experiences = [
            pickle.loads(zlib.decompress(self.buffer[idx])) for idx in indices]

        states = np.vstack(
            [exp.state for exp in selected_experiences]).astype(np.float32)

        actions = np.vstack(
            [exp.action for exp in selected_experiences]).astype(np.float32)

        rewards = np.array(
            [exp.reward for exp in selected_experiences]).reshape(-1, 1)

        next_states = np.vstack(
            [exp.next_state for exp in selected_experiences]
            ).astype(np.float32)

        dones = np.array(
            [exp.done for exp in selected_experiences]).reshape(-1, 1)

        return (states, actions, rewards, next_states, dones)


if __name__ == "__main__":
    import pickle
    import time
    import sys
    import zlib
    import bz2
    import gzip


    N = 200
    buffer = []
    s = time.time()
    for i in range(N):
        exp = Experience(np.ones((84,84,4)), 1, 1, np.ones((84,84,4)), False)
        buffer.append(exp)
    else:
        print("raw object")
        print(time.time() - s)
        print()

    N = 200
    buffer = []
    s = time.time()
    for i in range(N):
        exp = Experience(np.ones((84,84,4)), 1, 1, np.ones((84,84,4)), False)
        exp = zlib.compress(pickle.dumps(exp))
        buffer.append(exp)
    else:
        print("zlib")
        print("compress:", time.time() - s)
        s = time.time()
        buffer = [pickle.loads(zlib.decompress(exp)) for exp in buffer]
        print("decompress", time.time() - s)
        exp = Experience(np.ones((84,84,4)), 1, 1, np.ones((84,84,4)), False)
        exp_bytes = pickle.dumps(exp)
        exp_compressed = zlib.compress(exp_bytes)
        print("圧縮率", sys.getsizeof(exp_compressed) / sys.getsizeof(exp_bytes))
        print()

    N = 200
    buffer = []
    s = time.time()
    for i in range(N):
        exp = Experience(np.ones((84,84,4)), 1, 1, np.ones((84,84,4)), False)
        exp = gzip.compress(pickle.dumps(exp))
        buffer.append(exp)
    else:
        print("gzip")
        print("compress:", time.time() - s)
        s = time.time()
        buffer = [pickle.loads(gzip.decompress(exp)) for exp in buffer]
        print("decompress", time.time() - s)
        exp = Experience(np.ones((84,84,4)), 1, 1, np.ones((84,84,4)), False)
        exp_bytes = pickle.dumps(exp)
        exp_compressed = gzip.compress(exp_bytes)
        print("圧縮率", sys.getsizeof(exp_compressed) / sys.getsizeof(exp_bytes))
        print()

    N = 200
    buffer = []
    s = time.time()
    for i in range(N):
        exp = Experience(np.ones((84,84,4)), 1, 1, np.ones((84,84,4)), False)
        exp = bz2.compress(pickle.dumps(exp))
        buffer.append(exp)
    else:
        print("bz2")
        print("compress:", time.time() - s)
        s = time.time()
        buffer = [pickle.loads(bz2.decompress(exp)) for exp in buffer]
        print("decompress", time.time() - s)
        exp = Experience(np.ones((84,84,4)), 1, 1, np.ones((84,84,4)), False)
        exp_bytes = pickle.dumps(exp)
        exp_compressed = bz2.compress(exp_bytes)
        print("圧縮率", sys.getsizeof(exp_compressed) / sys.getsizeof(exp_bytes))
        print()
