import ray
import time



@ray.remote
class ReplayBuffer:

    def __init__(self):

        self.buffer = []

    def task1(self):
        for i in range(10):
            time.sleep(1)
        return "task1"

    def task2(self):
        for i in range(10):
            time.sleep(1)
        return "task2"


def main():
    ray.init()
    rb = ReplayBuffer.remote()

    s = time.time()
    res1 = rb.task1.remote()
    res2 = rb.task2.remote()

    print(ray.get(res1), ray.get(res2))
    print(time.time() - s)


if __name__ == "__main__":
    main()
