from pathlib import Path
import shutil
import os
import random

import numpy as np
import tensorflow as tf
import gym
from gym import wrappers

from networks import DualQNetwork, DiffusionPolicy


def load_dataset(dataset_path: str, batch_size: int):

    def deserialize(serialized_transition):

        transition = tf.io.parse_single_example(
            serialized_transition,
            features={
                'state': tf.io.FixedLenFeature([], tf.string),
                'action': tf.io.FixedLenFeature([], tf.string),
                'reward': tf.io.FixedLenFeature([], tf.float32),
                'next_state': tf.io.FixedLenFeature([], tf.string),
                'done': tf.io.FixedLenFeature([], tf.float32),
            }
        )

        a = tf.io.decode_raw(transition["action"], tf.float32)
        r = transition["reward"]
        d = transition["done"]
        s = tf.io.decode_raw(transition["state"], tf.float32)
        s2 = tf.io.decode_raw(transition["next_state"], tf.float32)

        return s, a, r, s2, d

    dataset = (
        tf.data.TFRecordDataset(filenames=dataset_path, num_parallel_reads=tf.data.AUTOTUNE)
               .shuffle(1024*1024, reshuffle_each_iteration=True)
               .repeat()
               .map(deserialize, num_parallel_calls=tf.data.AUTOTUNE)
               .batch(batch_size, drop_remainder=True)
               .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


class DiffusionQLAgent:

    def __init__(self, env_id: str):

        self.env_id = env_id
        self.action_space = gym.make(self.env_id).action_space.shape[0]

        self.soft_update_ratio = 0.005
        self.gamma = 0.99
        self.eta = 1.0

        self.qnet = DualQNetwork()
        self.target_qnet = DualQNetwork()
        self.policy = DiffusionPolicy(action_space=self.action_space)

        self.q_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.p_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

        self.setup()

    def setup(self):
        """ Initialize network weights """

        env = gym.make(self.env_id)

        dummy_state = env.reset()
        dummy_state = (dummy_state[np.newaxis, ...]).astype(np.float32)

        dummy_action = np.random.normal(0, 0.1, size=self.action_space)
        dummy_action = (dummy_action[np.newaxis, ...]).astype(np.float32)

        self.qnet(dummy_state, dummy_action)
        self.target_qnet(dummy_state, dummy_action)
        self.target_qnet.set_weights(self.qnet.get_weights())

    def save(self, save_dir="checkpoints/"):
        save_dir = Path(save_dir)

        self.policy.save_weights(str(save_dir / "policy"))
        self.qnet.save_weights(str(save_dir / "qnet"))
        self.valuenet.save_weights(str(save_dir / "valuenet"))

    def load(self, load_dir="checkpoints/"):
        load_dir = Path(load_dir)

        self.policy.load_weights(str(load_dir / "policy"))
        self.qnet.load_weights(str(load_dir / "qnet"))
        self.target_qnet.load_weights(str(load_dir / "qnet"))
        self.valuenet.load_weights(str(load_dir / "valuenet"))

    def update_q(self, states, actions, rewards, next_states, dones):

        rewards = tf.clip_by_value(tf.reshape(rewards, (-1, 1)), -1.0, 1.0)
        dones = tf.reshape(dones, (-1, 1))
        next_actions = self.policy.sample_actions(states=next_states)

        target_q1, target_q2 = self.target_qnet(states=next_states, actions=next_actions)
        target_q = tf.minimum(target_q1, target_q2)
        target = rewards + self.gamma * (1.0 - dones) * target_q

        with tf.GradientTape() as tape:
            q1, q2 = self.qnet(states=states, actions=actions)
            loss = tf.reduce_mean(
                tf.square(target - q1) + tf.square(target - q2)
            )

        variables = self.qnet.trainable_variables
        grads = tape.gradient(loss, variables)
        self.q_optimizer.apply_gradients(zip(grads, variables))

        return loss

    def update_policy(self, actions, states):

        th = random.random()

        with tf.GradientTape() as tape:

            bc_loss = self.policy.compute_bc_loss(states=states, actions=actions)
            sampled_actions = self.policy.sample_actions(states=states)
            q1, q2 = self.target_qnet(states=states, actions=sampled_actions)
            q1_loss = - tf.reduce_mean(q1) / tf.reduce_mean(tf.math.abs(q2))
            q2_loss = - tf.reduce_mean(q2) / tf.reduce_mean(tf.math.abs(q1))
            q_loss = tf.where(th > 0.5, q1_loss, q2_loss)

            loss = tf.reduce_mean(bc_loss + self.eta * q_loss)

        variables = self.policy.trainable_variables
        grads = tape.gradient(loss, variables)
        self.p_optimizer.apply_gradients(zip(grads, variables))

        return loss

    def sync_target_weight(self):

        tau = self.soft_update_ratio
        self.target_qnet.set_weights([
            tau * var + (1 - tau) * t_var for var, t_var
            in zip(self.qnet.get_weights(), self.target_qnet.get_weights())
        ])

    def test_play_(self, monitor_dir, tag):

        env = wrappers.RecordVideo(
            gym.make(self.env_id),
            video_folder=monitor_dir,
            step_trigger=lambda i: True,
            name_prefix=tag
        )

        state = env.reset()

        done = False
        episode_reward = 0
        while not done:
            action = self.policy.sample_actions(np.atleast_2d(state))
            action = action.numpy()[0]
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state


        return episode_reward


def main(env_id="BipedalWalker-v3"):

    os.environ["SDL_VIDEODRIVER"] = "dummy"   # Needed only for ubuntu

    LOGDIR = Path(__file__).parent / "log"
    if LOGDIR.exists():
        shutil.rmtree(LOGDIR)

    MONITOR_DIR = Path(__file__).parent / "mp4"
    if MONITOR_DIR.exists():
        shutil.rmtree(MONITOR_DIR)

    summary_writer = tf.summary.create_file_writer(str(LOGDIR))

    tf_dataset = load_dataset(dataset_path="bipedalwalker.tfrecord", batch_size=16)
    agent = DiffusionQLAgent(env_id)

    for n, minibatch in enumerate(tf_dataset):
        states, actions, rewards, next_states, dones = minibatch

        qloss = agent.update_q(states, actions, rewards, next_states, dones)
        ploss = agent.update_policy(actions, states)
        agent.sync_target_weight()
        agent.test_play()
        import pdb; pdb.set_trace()

        with summary_writer.as_default():
            tf.summary.scalar("loss_v", vloss, step=n)
            tf.summary.scalar("loss_p", ploss, step=n)
            tf.summary.scalar("loss_q", qloss, step=n)

        if n % 2000 == 0:
            score = agent.test_play(tag=f"{n}", monitor_dir=MONITOR_DIR)
            with summary_writer.as_default():
                tf.summary.scalar("test", score, step=n)

        if n % 10000 == 0:
            agent.save("checkpoints/")

        if n != 0 and n % 500000 == 0:
            break


if __name__ == '__main__':
    main()
