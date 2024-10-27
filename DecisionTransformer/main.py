import time
from pathlib import Path
import random
import shutil
import collections
import psutil
from typing import Optional

import gym
from gym.wrappers import RecordVideo
import ray
import numpy as np
import tensorflow as tf
from dopamine.discrete_domains.atari_lib import AtariPreprocessing

from dataset import create_dataloaders, SequenceBuffer
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

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=6e-4, beta_1=0.9, beta_2=0.95)

        self.monitor_dir = Path(monitor_dir)

    def save(self, save_dir="checkpoints/"):
        save_dir = Path(save_dir)
        self.model.save_weights(str(save_dir / "network"))

    def load(self, load_dir="checkpoints/"):
        load_dir = Path(load_dir)
        self.model.load_weights(str(load_dir / "network"))

    def update_network(self, rtgs, states, actions, timesteps):

        labels = tf.one_hot(
            tf.squeeze(actions, axis=-1),
            depth=self.action_space, on_value=1., off_value=0.)

        with tf.GradientTape() as tape:
            logits = self.model(rtgs, states, actions, timesteps, training=True)  # (B, context_length, 4)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels, logits, axis=-1)  # (B, context_length)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, gnorm = tf.clip_by_global_norm(grads, 40.0)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

        return loss, gnorm, tf.reduce_max(rtgs)

    def get_env(self, filename: Optional[str] = None):
        env = gym.make(f"{self.env_id}NoFrameskip-v4")
        if filename:
            env = RecordVideo(env, self.monitor_dir, name_prefix=filename)
        env = AtariPreprocessing(
            environment=env, frame_skip=4,
            terminal_on_life_loss=False, screen_size=84)
        return env

    def evaluate(self, target_rtg: float=70.0, filename=None):

        env = self.get_env(filename)

        frames = collections.deque(maxlen=4)
        for _ in range(4):
            frames.append(np.zeros((84, 84), dtype=np.float32))

        frame = env.reset()[:, :, 0]  # (84, 84)
        frames.append(frame)

        rtgs = collections.deque([], maxlen=self.context_length)
        states = collections.deque([], maxlen=self.context_length)
        actions = collections.deque([], maxlen=self.context_length-1)
        timesteps = collections.deque([], maxlen=self.context_length)

        done, sum_rewards, score = False, 0, 0
        for step in range(1600):
            timesteps.append(step)
            rtgs.append(max(target_rtg - sum_rewards, 0))
            states.append(np.stack(frames, axis=2).astype(np.float32))

            sampled_action, _ = self.model.sample_action(
                rtgs=rtgs, states=states,
                actions=list(actions), timestep=timesteps[0])

            next_frame, reward, done, _ = env.step(sampled_action)
            frames.append(next_frame[:, :, 0])

            actions.append(sampled_action)

            score += reward
            sum_rewards += np.clip(reward, 0., 1.)

        return score, step


@ray.remote(num_cpus=1, num_gpus=0)
class Tester(DecisionTransformerAgent):
    pass


def train(env_id, dataset_dir, num_data_files,  num_parallel_calls,
          samples_per_file=10_000, max_timestep=1800,
          context_length=30, batch_size=32, resume_from=None):

    monitor_dir = Path(__file__).parent / "mp4"
    if resume_from is None and monitor_dir.exists():
        shutil.rmtree(monitor_dir)

    logdir = Path(__file__).parent / "log"
    if resume_from is None and logdir.exists():
        shutil.rmtree(logdir)
    summary_writer = tf.summary.create_file_writer(str(logdir))

    savedir = Path("checkpoints/")
    if resume_from is None and savedir.exists():
        shutil.rmtree(savedir)

    ray.init(local_mode=False)

    agent = DecisionTransformerAgent(
        env_id=env_id, context_length=context_length,
        max_timestep=max_timestep, monitor_dir=monitor_dir)

    if resume_from is not None:
        agent.load(load_dir="checkpoints/")

    tester = Tester.remote(
        env_id=env_id, context_length=context_length,
        max_timestep=max_timestep, monitor_dir=monitor_dir)
    test_wip = tester.evaluate.remote(filename="test_0")

    loaders = create_dataloaders(
        dataset_dir=dataset_dir, max_timestep=max_timestep, num_data_files=num_data_files,
        samples_per_file=samples_per_file, context_length=context_length,
        num_parallel_calls=num_parallel_calls, batch_size=batch_size)

    buffer = SequenceBuffer(maxlen=100_000, batch_size=batch_size)

    jobs_wip = [loader.sample_sequences.remote() for loader in loaders]

    for _ in range(5):
        job_done, jobs_wip = ray.wait(jobs_wip, num_returns=1)
        pid, sequences = ray.get(job_done)[0]
        buffer.add_sequences(sequences)
        jobs_wip.append(loaders[pid].sample_sequences.remote())

    s = time.time()
    n = 1 if resume_from is None else int(resume_from) * 1000 + 1
    while n < 1_000_000:
        job_done, jobs_wip = ray.wait(jobs_wip, num_returns=1, timeout=0)
        if job_done:
            pid, sequences = ray.get(job_done)[0]
            jobs_wip.append(loaders[pid].sample_sequences.remote())
            buffer.add_sequences(sequences)

        rtgs, states, actions, timesteps = buffer.sample_minibatch()
        loss, gnorm, rmax = agent.update_network(rtgs, states, actions, timesteps)

        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=n)
            tf.summary.scalar("gnorm", gnorm, step=n)
            tf.summary.scalar("rtgs_max", rmax, step=n)

        if n % 2000 == 0:

            score, steps = ray.get(test_wip)
            agent.save(str(savedir))
            ray.get(tester.load.remote("checkpoints/"))
            test_wip = tester.evaluate.remote(filename=f"test_{n}")

            laptime = time.time() - s
            mem = psutil.virtual_memory().used / (1024 ** 3)
            with summary_writer.as_default():
                tf.summary.scalar("score", score, step=n)
                tf.summary.scalar("steps", steps, step=n)
                tf.summary.scalar("laptime", laptime, step=n)
                tf.summary.scalar("Mem", mem, step=n)
                tf.summary.scalar("buffer", len(buffer), step=n)
            s = time.time()

        n += 1


if __name__ == "__main__":
    env_id = "Breakout"
    dataset_dir = "/mnt/disks/data/Breakout/1/replay_logs"
    train(env_id="Breakout", dataset_dir=dataset_dir, num_data_files=48, num_parallel_calls=12, resume_from=None)
