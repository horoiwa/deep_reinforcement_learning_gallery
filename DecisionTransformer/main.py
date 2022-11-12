import time

import ray

from dataset import create_dataloaders
from networks import DecisionTransformer


class Timer:

    def __init__(self, tag: str):
        self.tag = tag

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        fin = time.time() - self.start
        print(self.tag, fin)


def train(env_id, dataset_dir, num_data_files=1, samples_per_file=10_000,
          context_length=30, batch_size=48, num_parallel_calls=1, resume_from=None):

    ray.init()

    model = DecisionTransformer()

    dataloaders = create_dataloaders(
        dataset_dir=dataset_dir, num_data_files=num_data_files,
        samples_per_file=samples_per_file, context_length=context_length,
        num_parallel_calls=num_parallel_calls, batch_size=batch_size)


    start = time.time()
    for i, _ in enumerate(dataset):
        print(time.time() - start)
        start = time.time()


def evaluate(env_id):
    pass


if __name__ == "__main__":
    env_id = "Breakout"
    dataset_dir = "/mnt/disks/data/Breakout/1/replay_logs"
    train(env_id="Breakout", dataset_dir=dataset_dir, num_data_files=4, num_parallel_calls=4)
