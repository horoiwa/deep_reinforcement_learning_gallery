import collections
from pathlib import Path
import shutil
import time

import tensorflow as tf
import gym
from gym.wrappers import RecordVideo
import numpy as np
from PIL import Image

from buffer import OfflineBuffer, create_tfrecords
from model import QuantileQNetwork


def preprocess(frame):

    img = Image.fromarray(frame).convert("L").resize((84,84))
    img = np.array(img, dtype=np.float32)
    return img


class CQLAgent:

    def __init__(self, env_id, dataset_dir, n_atoms=200,
                 batch_size=32, gamma=0.99, kappa=1.0, cql_weight=1.0):

        self.env_id = env_id

        self.action_space = gym.make(self.env_id).action_space.n

        self.n_atoms = n_atoms

        self.quantiles = [1/(2*n_atoms) + i * 1 / n_atoms for i in range(self.n_atoms)]

        self.qnetwork = QuantileQNetwork(actions_space=self.action_space, n_atoms=self.n_atoms)

        self.target_qnetwork = QuantileQNetwork(actions_space=self.action_space, n_atoms=n_atoms)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005, epsilon=0.00031)

        self.replaybuffer = OfflineBuffer(dataset_dir=dataset_dir, batch_size=batch_size)

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

    def rollout(self, monitor_dir=None, name_prefix=None):

        rewards, steps = 0., 0.

        env = gym.make(self.env_id)
        if monitor_dir is not None:
            assert name_prefix is not None
            env = RecordVideo(env, monitor_dir, name_prefix=name_prefix)

        frames = collections.deque(maxlen=4)
        for _ in range(4):
            frames.append(np.zeros((84, 84), dtype=np.float32))

        frame = preprocess(env.reset())
        frames.append(frame)

        done = False
        while (not done) and (steps < 3000):
            state = np.stack(frames, axis=2)[np.newaxis, ...]
            action = self.qnetwork.sample_action(state)
            next_frame, reward, done, _ = env.step(action)
            frames.append(preprocess(next_frame))

            rewards += reward
            steps += 1

        return rewards, steps

    def update_network(self):

        states, actions, rewards, next_states, dones = self.replaybuffer.sample_minibatch()
        rewards = np.clip(rewards, 0., 1.)

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

        info = {"total_loss": loss.numpy(),
                "cql_loss": tf.reduce_mean(cql_loss).numpy(),
                "td_loss": tf.reduce_mean(td_loss).numpy()}

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
        target_quantile_qvalues = tf.expand_dims(rewards, axis=1)  #(B, 1)
        target_quantile_qvalues += self.gamma * (1. - tf.expand_dims(dones, axis=1)) * next_quantile_qvalues  # (B, N_ATOMS)

        return target_quantile_qvalues

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

        td_loss = tf.reduce_mean(tf.reduce_sum(quantile_huber_loss, axis=2), axis=1)

        return td_loss

    def sync_target_weights(self):
        self.target_qnetwork.set_weights(self.qnetwork.get_weights())


def train(n_iter=20000000,
          env_id="BreakoutDeterministic-v4",
          dataset_dir="/mnt/disks/data/tfrecords_dqn_replay_dataset/",
          target_update_period=8000, resume_from=None):

    logdir = Path(__file__).parent / "log"
    if logdir.exists() and resume_from is None:
        shutil.rmtree(logdir)

    summary_writer = tf.summary.create_file_writer(str(logdir))

    agent = CQLAgent(env_id=env_id, dataset_dir=dataset_dir)

    if resume_from is not None:
        agent.load()
        step_init = int(resume_from * 1000)
    else:
        step_init = 1

    s = time.time()
    for n in range(step_init, n_iter):

        info = agent.update_network()

        if n % 20 == 0:
            with summary_writer.as_default():
                tf.summary.scalar("loss", info["total_loss"], step=n)
                tf.summary.scalar("cql_loss", info["cql_loss"], step=n)
                tf.summary.scalar("td_loss", info["td_loss"], step=n)

        if n % target_update_period == 0:
            agent.sync_target_weights()

        if n % 5000 == 0:
            rewards, steps = agent.rollout()
            with summary_writer.as_default():
                tf.summary.scalar("test_score", rewards, step=n)
                tf.summary.scalar("test_steps", steps, step=n)
                tf.summary.scalar("laptime", time.time() - s, step=n)
            s = time.time()
            print(f"== test: {n} ===")
            print(f"score: {rewards}, step: {steps}")

        if n % 100000 == 0:
            agent.save()


def test(env_id="BreakoutDeterministic-v4",
         dataset_dir="/mnt/disks/data/tfrecords_dqn_replay_dataset/"):

    monitor_dir = Path(__file__).parent / "mp4"
    if monitor_dir.exists():
        shutil.rmtree(monitor_dir)

    agent = CQLAgent(env_id=env_id, dataset_dir=dataset_dir)

    agent.load()

    for i in range(10):
        agent.rollout(monitor_dir=monitor_dir, name_prefix=f"test_{i}")

    print("Finished")


def check(env_id="BreakoutDeterministic-v4",
          dataset_dir="/mnt/disks/data/tfrecords_dqn_replay_dataset/"):

    agent = CQLAgent(env_id=env_id, dataset_dir=dataset_dir)

    minibatch = agent.replaybuffer.sample_minibatch()
    state = minibatch[0][0].numpy()
    from PIL import Image
    for i in range(4):
        img = Image.fromarray(state[:, :, i])
        img.convert("L").save(f"tmp/s{i}.png")

    state = minibatch[3][0].numpy()
    for i in range(4):
        img = Image.fromarray(state[:, :, i])
        img.convert("L").save(f"tmp/ns{i}.png")


if __name__ == '__main__':
    original_dataset_dir = "./dqn-replay-dataset/Breakout/1/replay_logs"
    dataset_dir = "/mnt/disks/data/tfrecords_dqn_replay_dataset/"

    #create_tfrecords(original_dataset_dir=original_dataset_dir, dataset_dir=dataset_dir, num_data_files=5, use_samples_per_file=1000000)
    #check(dataset_dir=dataset_dir)
    train(resume_from=None)
    test()
