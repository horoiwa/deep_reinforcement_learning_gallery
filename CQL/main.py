import collections
from pathlib import Path
import shutil
import time
import psutil

import tensorflow as tf
import gym
from gym.wrappers import RecordVideo
import numpy as np
from PIL import Image
from dopamine.discrete_domains.atari_lib import create_atari_environment

from buffer import load_dataset, create_tfrecords
from model import QuantileQNetwork


def preprocess(frame):

    img = Image.fromarray(frame).convert("L").resize((84, 84))
    img = np.array(img, dtype=np.float32)
    return img


class CQLAgent:

    def __init__(self, env_id, n_atoms=100,
                 gamma=0.99, kappa=1.0, cql_weight=1.0):

        self.env_id = env_id

        self.action_space = gym.make(self.env_id).action_space.n

        self.n_atoms = n_atoms

        self.quantiles = [1/(2*n_atoms) + i * 1 / n_atoms for i in range(self.n_atoms)]

        self.qnetwork = QuantileQNetwork(actions_space=self.action_space, n_atoms=self.n_atoms)

        self.target_qnetwork = QuantileQNetwork(actions_space=self.action_space, n_atoms=n_atoms)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005, epsilon=0.00031)

        self.gamma = gamma

        self.kappa = kappa

        self.cql_weight = cql_weight

        self.setup()

    def setup(self):

        frames = collections.deque(maxlen=4)

        env = gym.make(self.env_id)

        frame = preprocess(env.reset())

        for _ in range(4):
            frames.append(frame)

        state = np.stack(frames, axis=2)[np.newaxis, ...]

        self.qnetwork(state)
        self.target_qnetwork(state)

        self.sync_target_weights()

    def save(self, save_dir="checkpoints/"):
        save_dir = Path(save_dir)
        self.qnetwork.save_weights(str(save_dir / "qnetwork"))

    def load(self, load_dir="checkpoints/"):
        load_dir = Path(load_dir)
        self.qnetwork.load_weights(str(load_dir / "qnetwork"))
        self.target_qnetwork.load_weights(str(load_dir / "qnetwork"))

    def rollout(self):

        env = create_atari_environment(game_name=self.env_id, sticky_actions=False)

        rewards, steps = 0, 0

        frames = collections.deque(maxlen=4)
        for _ in range(4):
            frames.append(np.zeros((84, 84), dtype=np.float32))

        frame = env.reset()[:, :, 0]
        frames.append(frame)

        done = False
        while (not done) and (steps < 3000):
            state = np.stack(frames, axis=2)[np.newaxis, ...]
            action = self.qnetwork.sample_action(state)
            next_frame, reward, done, _ = env.step(action)
            frames.append(next_frame[:, :, 0])

            rewards += reward
            steps += 1

        return rewards, steps

    def record(self, monitor_dir, name_prefix):

        env = gym.make(f"{self.env_id}Deterministic-v4")
        env = RecordVideo(env, monitor_dir, name_prefix=name_prefix)

        frames = collections.deque(maxlen=4)
        for _ in range(4):
            frames.append(np.zeros((84, 84), dtype=np.float32))

        frame = preprocess(env.reset())
        frames.append(frame)

        done, steps = False, 0
        while (not done) and (steps < 3000):
            state = np.stack(frames, axis=2)[np.newaxis, ...]
            action = self.qnetwork.sample_action(state)
            next_frame, reward, done, _ = env.step(action)
            frames.append(preprocess(next_frame))
            steps += 1

    def update_network(self, minibatch):

        states, actions, rewards, next_states, dones = minibatch

        #  TQ = reward + γ * max_a[Q(s, a)]
        target_quantile_qvalues = self.make_target_distribution(rewards, next_states, dones)

        with tf.GradientTape() as tape:
            quantile_qvalues_all = self.qnetwork(states)  # (B, A, N_ATOMS)
            actions_onehot = tf.expand_dims(
                tf.one_hot(actions, self.action_space, on_value=1., off_value=0.), axis=2)  # (B, A, 1)
            quantile_qvalues = tf.reduce_sum(
                quantile_qvalues_all * actions_onehot, axis=1, keepdims=False)  # (B, N_ATOMS)

            #: Quantile huber loss
            td_loss = self.quantile_huberloss(target_quantile_qvalues, quantile_qvalues)  # (B, )

            #: CQL(H)
            Q_learned_all = tf.reduce_mean(quantile_qvalues_all, axis=2)  # (B, A)

            log_Z = tf.reduce_logsumexp(Q_learned_all, axis=1)  # (B, )

            Q_behavior = tf.reduce_mean(quantile_qvalues, axis=1)  # (B, )

            cql_loss = log_Z - Q_behavior

            loss = tf.reduce_mean(self.cql_weight * cql_loss + td_loss)

        variables = self.qnetwork.trainable_variables
        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

        Q_learned_max = tf.reduce_max(Q_learned_all, axis=1)
        Q_diff = (Q_behavior - Q_learned_max)

        #import pdb; pdb.set_trace()

        info = {"total_loss": loss.numpy(),
                "cql_loss": tf.reduce_mean(cql_loss).numpy(),
                "td_loss": tf.reduce_mean(td_loss).numpy(),
                "rewards": rewards.numpy().sum(),
                "q_behavior": tf.reduce_mean(Q_behavior).numpy(),
                "q_diff": tf.reduce_mean(Q_diff).numpy(),
                "dones": dones.numpy().sum()}

        return info

    def make_target_distribution(self, rewards, next_states, dones):

        next_quantile_qvalues_all = self.target_qnetwork(next_states)
        next_qvalues_all = tf.reduce_mean(next_quantile_qvalues_all, axis=2, keepdims=False)

        next_actions = tf.argmax(next_qvalues_all, axis=1)  # (B, )
        next_actions_onehot = tf.expand_dims(
            tf.one_hot(next_actions, self.action_space, on_value=1., off_value=0.), axis=2)  # (B, A, 1)

        next_quantile_qvalues = tf.reduce_sum(
            next_quantile_qvalues_all * next_actions_onehot, axis=1, keepdims=False)  # (B, N_ATOMS)

        #  TQ = reward + γ * max_a[Q(s, a)]
        target_quantile_qvalues = tf.expand_dims(rewards, axis=1) + self.gamma * (1. - tf.expand_dims(dones, axis=1)) * next_quantile_qvalues  # (B, N_ATOMS)

        return target_quantile_qvalues

    @tf.function
    def quantile_huberloss(self, target_quantile_values, quantile_values):
        target_quantile_values = tf.repeat(
            tf.expand_dims(target_quantile_values, axis=1), self.n_atoms, axis=1)  # (B, N_ATOMS, N_ATOMS)

        quantile_values = tf.repeat(
            tf.expand_dims(quantile_values, axis=2), self.n_atoms, axis=2)  # (B, N_ATOMS, N_ATOMS)

        errors = target_quantile_values - quantile_values

        is_smaller_than_kappa = tf.abs(errors) < self.kappa
        squared_loss = 0.5 * tf.square(errors)
        linear_loss = self.kappa * (tf.abs(errors) - 0.5 * self.kappa)
        huber_loss = tf.where(is_smaller_than_kappa, squared_loss, linear_loss)

        indicator = tf.stop_gradient(tf.where(errors < 0, 1., 0.))
        quantiles = tf.repeat(tf.expand_dims(self.quantiles, axis=1), self.n_atoms, axis=1)
        quantile_weights = tf.abs(quantiles - indicator)

        quantile_huber_loss = quantile_weights * huber_loss

        td_loss = tf.reduce_sum(tf.reduce_mean(quantile_huber_loss, axis=2), axis=1)

        return td_loss

    def sync_target_weights(self):
        self.target_qnetwork.set_weights(self.qnetwork.get_weights())


def train(n_iter=20000000,
          env_id="BreakoutDeterministic-v4",
          dataset_dir="/mnt/disks/data/tfrecords_dqn_replay_dataset/",
          batch_size=48,
          target_update_period=8000, resume_from=None):

    logdir = Path(__file__).parent / "log"
    if logdir.exists() and resume_from is None:
        shutil.rmtree(logdir)

    summary_writer = tf.summary.create_file_writer(str(logdir))

    agent = CQLAgent(env_id=env_id, cql_weight=1.0)
    dataset = load_dataset(dataset_dir=dataset_dir, batch_size=batch_size)

    if resume_from is not None:
        agent.load()
        n = int(resume_from * 1000)
    else:
        n = 1

    s = time.time()
    for minibatch in dataset:

        info = agent.update_network(minibatch)

        if n % 25 == 0:
            with summary_writer.as_default():
                tf.summary.scalar("loss", info["total_loss"], step=n)
                tf.summary.scalar("cql_loss", info["cql_loss"], step=n)
                tf.summary.scalar("td_loss", info["td_loss"], step=n)
                tf.summary.scalar("rewards", info["rewards"], step=n)
                tf.summary.scalar("q_behavior", info["q_behavior"], step=n)
                tf.summary.scalar("q_diff", info["q_diff"], step=n)
                tf.summary.scalar("dones", info["dones"], step=n)

        if n % target_update_period == 0:
            agent.sync_target_weights()

        if n % 2500 == 0:
            rewards, steps = agent.rollout()
            mem = psutil.virtual_memory().used / (1024 ** 3)

            with summary_writer.as_default():
                tf.summary.scalar("test_score", rewards, step=n)
                tf.summary.scalar("test_steps", steps, step=n)
                tf.summary.scalar("laptime", time.time() - s, step=n)
                tf.summary.scalar("Mem", mem, step=n)

            s = time.time()
            print(f"== test: {n} ===")
            print(f"score: {rewards}, step: {steps}")

        if n % 25000 == 0:
            agent.save()
            agent.reload_buffer()

        if n > n_iter:
            break

        n += 1


def test(env_id="BreakoutDeterministic-v4",
         dataset_dir="/mnt/disks/data/tfrecords_dqn_replay_dataset/"):

    monitor_dir = Path(__file__).parent / "mp4"
    if monitor_dir.exists():
        shutil.rmtree(monitor_dir)

    agent = CQLAgent(env_id=env_id)
    agent.load()

    for i in range(10):
        print(agent.rollout())

    for i in range(5):
        agent.record(monitor_dir=monitor_dir, name_prefix=f"test_{i}")

    print("Finished")


def check_buffer(env_id="BreakoutDeterministic-v0",
                 dataset_dir="/mnt/disks/data/tfrecords_dqn_replay_dataset/"):

    dataset = load_dataset(dataset_dir=dataset_dir, batch_size=32)

    print("=========START=======")
    s0 = time.time()
    s = time.time()

    for i, minibatch in enumerate(dataset):
        state, actions, rewards, next_state, dones = minibatch
        print(i, dones.numpy().sum(), rewards.numpy().sum(), dones.shape[0])

        if i % 100 == 0:
            print(time.time() - s)
            s = time.time()

        if i > 10000:
            break

    print("=========FINISHED=======")
    print(time.time() - s0)



if __name__ == '__main__':
    env_id = "Breakout"
    original_dataset_dir = "/mnt/disks/data/Breakout/1/replay_logs"
    dataset_dir = "/mnt/disks/data/tfrecords_dqn_replay_dataset/"

    #create_tfrecords(original_dataset_dir=original_dataset_dir, dataset_dir=dataset_dir, num_data_files=10, use_samples_per_file=50000, num_chunks=10)
    #check_buffer(dataset_dir=dataset_dir)

    train(env_id=env_id, resume_from=None)
    #test(env_id=env_id)
