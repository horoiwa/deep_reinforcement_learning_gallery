from buffer import SequenceReplayBuffer
from models import DecisionTransformer


def train(env_id, resume_from=None):

    dataloader = get_dataloader()
    agent = DecisionTransformer()

    n = 1 if resume_from is None else int(resume_from)
    for minibatch in dataloader:

        if n % 10_000:
            DecisionTransformer.save_weights("checkpoints/")
            evaluate(env_id)

        n += 1


def evaluate(env_id):
    pass


if __name__ == "__main__":
    env_id = "Breakout"
    dataset_dir = "/mnt/disks/data/Breakout/1/replay_logs"
    create_dataset(
        src_dir=src_dataset_dir, out_dir=dataset_dir,
        num_data_files=1, samples_per_file=10_000)
    #train(env_id="Breakout")
