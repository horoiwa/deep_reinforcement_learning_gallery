import time
import random

import ray


@ray.remote
def f(pid):

    time.sleep(random.randint(0, 5))

    return pid


def main():

    ray.init()

    results = [f.remote(i) for i in range(3)]

    while True:
        done_id, results = ray.wait(results, num_returns=1)
        print(done_id, results)
        res = ray.get(done_id)
        print(res)
        print()

        if not results:
            break


if __name__ == "__main__":
    main()
