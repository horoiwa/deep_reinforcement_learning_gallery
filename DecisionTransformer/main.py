import time
import collections

from PIL import Image
import gym
import ray
import numpy as np

from dataset import create_dataloaders
from networks import DecisionTransformer


class DecisionTransformerAgent:

    def __init__(self, env_id, max_timestep, context_length):

        self.env_id = env_id

        self.action_space = gym.make(f"{self.env_id}Deterministic-v4").action_space.n

        self.model = DecisionTransformer(
            action_space=self.action_space,
            max_timestep=max_timestep,
            context_length=context_length)

    def preprocess(self, frame):

        img = Image.fromarray(frame).convert("L").resize((84, 84))
        img = np.array(img, dtype=np.float32)
        return img

    def update_network(self, rtgs, states, actions, timesteps):
        self.model(rtgs, states, actions, timesteps, training=True)

        pass


def train(env_id, dataset_dir, num_data_files,  num_parallel_calls=1,
          #samples_per_file=10_000,
          samples_per_file=5_000, max_timestep=3000,
          context_length=30, batch_size=48, resume_from=None):

    ray.init()

    dataloaders, max_timestep_dataset = create_dataloaders(
        dataset_dir=dataset_dir, num_data_files=num_data_files,
        samples_per_file=samples_per_file, context_length=context_length,
        num_parallel_calls=num_parallel_calls, batch_size=batch_size)

    print()
    print("Dataset maxtimestep:", max_timestep_dataset)
    print()

    assert max_timestep > max_timestep_dataset

    agent = DecisionTransformerAgent(
        env_id=env_id, context_length=context_length, max_timestep=max_timestep)

    jobs_wip = [loader.sample_minibatch.remote() for loader in dataloaders]

    n = 1
    while True:
        job_done, jobs_wip = ray.wait(jobs_wip, num_returns=1)
        pid, minibatch = ray.get(job_done)[0]
        jobs_wip.append(dataloaders[pid].sample_minibatch.remote())

        rtgs, states, actions, timesteps = minibatch
        loss = agent.update_network(rtgs, states, actions, timesteps)

        n += 1
        break


def evaluate(env_id):
    pass


if __name__ == "__main__":
    env_id = "Breakout"
    dataset_dir = "/mnt/disks/data/Breakout/1/replay_logs"
    train(env_id="Breakout", dataset_dir=dataset_dir, num_data_files=1, num_parallel_calls=1)
