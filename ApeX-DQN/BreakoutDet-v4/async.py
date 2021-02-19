import time
import threading
import random
from queue import Queue

import numpy as np


class Buffer:

    def __init__(self):

        self.buffer = []
        self.temp = []

        self.in_queue = Queue()
        self.out_queue = Queue()

    def start(self):
        t1 = threading.Thread(target=self.push)
        t2 = threading.Thread(target=self.pull)

    def push(self, val):
        self.buffer.append(val)
        return val

    def pull(self, batch_size):
        while True:
            if len(out_queue) < self.maxoutlen:
                indices = np.random.choice(
                    np.arange(len(self.buffer)), size=batch_size)
                experiences = [self.buffer[idx] for idx in indices]

                time.sleep(1)
                out_queue.put((indices, experiences))


def main():

    buffer = Buffer()
    for val in range(1000):
        buffer.push(val)

    s = time.time()
    #: ミニバッチ作り続ける
    minibatch = buffer.pull(batch_size=32)
    print("1", time.time() - s)

    val = random.randint(0, 100)
    val = buffer.push(val)
    print(val)
    print("2", time.time() - s)
    #: データ追加し続ける
    for i in range(100000):
        val = random.randint(0, 100)
        val = buffer.push(val)

    minibatch = minibatch


def main2():
    q = Queue(maxsize=16)
    for i in range(100):
        print(i)
        q.put(i)
    print(q)

if __name__ == "__main__":
    main2()
