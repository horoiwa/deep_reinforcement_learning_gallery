import time
from pathlib import Path
import random
import shutil
import collections
from typing import Optional

import gym
from gym.wrappers import RecordVideo
import ray
import numpy as np
import tensorflow as tf
from dopamine.discrete_domains.atari_lib import AtariPreprocessing

from dataset import create_dataloaders
from networks import DecisionTransformer


class DecisionTransformerAgent:

    def __init__(self, env_id, max_timestep, context_length, monitor_dir):

        self.env_id = env_id

        self.action_space = gym.make(f"{self.env_id}Deterministic-v4").action_space.n

        self.context_length = context_length

        self.model = DecisionTransformer(
            action_space=self.action_space,
            max_timestep=max_timestep,
            context_length=context_length)

        self.optimizer = tf.keras.optimizers.Adam(lr=6e-4, beta_1=-0.9, beta_2=0.95)

        self.monitor_dir = Path(monitor_dir)

    def save(self, save_dir="checkpoints/"):
        save_dir = Path(save_dir)
        self.model.save_weights(str(save_dir / "network"))

    def load(self, load_dir="checkpoints/"):
        load_dir = Path(load_dir)
        self.model.load_weights(str(load_dir / "network"))

    def update_network(self, rtgs, states, actions, timesteps):

        targets = tf.one_hot(
            tf.squeeze(actions, axis=-1),
            depth=self.action_space, on_value=1., off_value=0.)

        with tf.GradientTape() as tape:
            logits = self.model(rtgs, states, actions, timesteps, training=True)  # (B, context_length, 4)
            loss = tf.nn.softmax_cross_entropy_with_logits(targets, logits, axis=-1)  # (B, context_length)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, gnorm = tf.clip_by_global_norm(grads, 40.0)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

        return loss, gnorm

    def get_env(self, filename: Optional[str] = None):
        env = gym.make(f"{self.env_id}NoFrameskip-v4")
        if filename:
            env = RecordVideo(env, self.monitor_dir, name_prefix=filename)
        env = AtariPreprocessing(
            environment=env, frame_skip=4,
            terminal_on_life_loss=False, screen_size=84)
        return env

    def evaluate(self, target_rtg=90, filename=None):

        env = self.get_env(filename)

        frames = collections.deque(maxlen=4)
        for _ in range(4):
            frames.append(np.zeros((84, 84), dtype=np.float32))

        frame = env.reset()[:, :, 0]  # (84, 84)
        frames.append(frame)

        rtgs = [target_rtg]
        states = []
        actions = []

        done, rewards, steps = False, 0, 0
        while done:
            rtgs.append(max(target_rtg - rewards, 0))
            states.append(np.stack(frames, axis=2))  # (84, 84, 4)

            action = self.model.sample_action(
                rtgs=rtgs, states=states, actions=None, timestep=teps)

            next_frame, reward, done, _ = env.step(action)
            frames.append(next_frame[:, :, 0])

            actions.append(action)

            rewards += reward
            steps += 1

            if steps > 1600:
                break

        return rewards, steps


def train(env_id, dataset_dir, num_data_files,  num_parallel_calls,
          samples_per_file=10_000, max_timestep=3000,
          context_length=30, batch_size=48, resume_from=None):

    monitor_dir = Path(__file__).parent / "mp4"
    if monitor_dir.exists() and resume_from is None:
        shutil.rmtree(monitor_dir)

    agent = DecisionTransformerAgent(
        env_id=env_id, context_length=context_length,
        max_timestep=max_timestep, monitor_dir=monitor_dir)
    agent.evaluate(filename="sample")
    import pdb; pdb.set_trace()

    logdir = Path(__file__).parent / "log"
    if logdir.exists() and resume_from is None:
        shutil.rmtree(logdir)
    summary_writer = tf.summary.create_file_writer(str(logdir))

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
        env_id=env_id, context_length=context_length,
        max_timestep=max_timestep, monitor_dir=monitor_dir)

    jobs_wip = [loader.sample_minibatch.remote() for loader in dataloaders]

    n = 1
    while n < 1_000_000:

        job_done, jobs_wip = ray.wait(jobs_wip, num_returns=1)
        pid, minibatch = ray.get(job_done)[0]
        jobs_wip.append(dataloaders[pid].sample_minibatch.remote())

        rtgs, states, actions, timesteps = minibatch
        loss, gnorm = agent.update_network(rtgs, states, actions, timesteps)

        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=n)
            tf.summary.scalar("gnorm", gnorm, step=n)

        if n % 250 == 0:
            score, steps = agent.evaluate(filename=f"step{n}")
            with summary_writer.as_default():
                tf.summary.scalar("score", score, step=n)
                tf.summary.scalar("steps", steps, step=n)

        if n % 2500 == 0:
            agent.save()

        n += 1


def evaluate(env_id):
    pass


if __name__ == "__main__":
    env_id = "Breakout"
    dataset_dir = "/mnt/disks/data/Breakout/1/replay_logs"
    train(env_id="Breakout", dataset_dir=dataset_dir, num_data_files=1, num_parallel_calls=1)
