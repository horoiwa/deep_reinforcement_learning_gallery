import time

from dataset import create_dataset
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
          context_length=30, batch_size=48, resume_from=None):

    dataset = create_dataset(
        dataset_dir=dataset_dir, num_data_files=num_data_files,
        samples_per_file=samples_per_file, context_length=context_length,
        batch_size=batch_size)

    model = DecisionTransformer()

    dataset = iter(dataset)
    for i in range(3):
        with Timer(f"iter{i}"):
            _ = next(dataset)

    #n = 1 if resume_from is None else int(resume_from)
    #for minibatch in dataset:
    #    print(minibatch)
    #    if n % 10_000:
    #        DecisionTransformer.save_weights("checkpoints/")
    #        evaluate(env_id)
    #    n += 1


def evaluate(env_id):
    pass


if __name__ == "__main__":
    env_id = "Breakout"
    dataset_dir = "/mnt/disks/data/Breakout/1/replay_logs"
    train(env_id="Breakout", dataset_dir=dataset_dir)
