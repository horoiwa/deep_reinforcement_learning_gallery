from pathlib import Path
import shutil

import numpy as np
import tensorflow as tf
import gym
from gym import wrappers

from networks import DualQNetwork, GaussianPolicy, ValueNetwork


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


class IQLAgent:

    def __init__(self, env_id: str):

        self.env_id = env_id
        self.action_space = gym.make(self.env_id).action_space.shape[0]

        self.tau = 0.8

        self.temperature = 0.1
        self.soft_update_ratio = 0.005
        self.gamma = 0.99

        self.policy = GaussianPolicy(action_space=self.action_space)
        self.valuenet = ValueNetwork()
        self.qnet = DualQNetwork()
        self.target_qnet = DualQNetwork()

        self.p_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.v_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.q_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

        self.setup()

    def setup(self):
        """ Initialize network weights """

        env = gym.make(self.env_id)

        dummy_state = env.reset()
        dummy_state = (dummy_state[np.newaxis, ...]).astype(np.float32)

        dummy_action = np.random.normal(0, 0.1, size=self.action_space)
        dummy_action = (dummy_action[np.newaxis, ...]).astype(np.float32)

        self.policy(dummy_state)

        self.qnet(dummy_state, dummy_action)
        self.target_qnet(dummy_state, dummy_action)
        self.target_qnet.set_weights(self.qnet.get_weights())

        self.valuenet(dummy_state)

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

    def update_value(self, states, actions):
        """ Expectile Regression
        """
        q1, q2 = self.target_qnet(states, actions)
        target_values = tf.minimum(q1, q2)

        with tf.GradientTape() as tape:
            values = self.valuenet(states)
            error = (target_values - values)
            weights = tf.where(error > 0, self.tau, 1. - self.tau)
            loss = tf.reduce_mean(weights * tf.square(error))

        variables = self.valuenet.trainable_variables
        grads = tape.gradient(loss, variables)
        self.v_optimizer.apply_gradients(zip(grads, variables))

        return loss

    def update_policy(self, states, actions):
        """ Advantage weighted regression
        """
        q1, q2 = self.target_qnet(states, actions)
        Q = tf.minimum(q1, q2)
        V = self.valuenet(states)

        exp_Adv = tf.minimum(tf.exp(Q - V), 100.0)

        with tf.GradientTape() as tape:
            dists = self.policy(states)
            log_probs = tf.reshape(dists.log_prob(actions), (-1, 1))
            loss = tf.reduce_mean(-1 * (exp_Adv * log_probs))

        variables = self.policy.trainable_variables
        grads = tape.gradient(loss, variables)
        self.p_optimizer.apply_gradients(zip(grads, variables))

        return loss

    def update_q(self, states, actions, rewards, dones, next_states):

        rewards, dones = tf.reshape(rewards, (-1, 1)), tf.reshape(dones, (-1, 1))

        target_q = rewards + self.gamma * (1.0 - dones) * self.valuenet(next_states)

        with tf.GradientTape() as tape:
            q1, q2 = self.qnet(states, actions)
            loss = tf.reduce_mean(
                tf.square(target_q - q1) + tf.square(target_q - q2)
            )

        variables = self.qnet.trainable_variables
        grads = tape.gradient(loss, variables)
        self.q_optimizer.apply_gradients(zip(grads, variables))

        return loss

    def sync_target_weight(self):

        tau = self.soft_update_ratio
        self.target_qnet.set_weights([
            tau * var + (1 - tau) * t_var for var, t_var
            in zip(self.qnet.get_weights(), self.target_qnet.get_weights())
        ])

    def test_play(self, monitor_dir, tag):

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

        print(f"{tag}", episode_reward)

        return episode_reward


def main(env_id="BipedalWalker-v3"):

    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"   # Needed only for ubuntu

    LOGDIR = Path(__file__).parent / "log"
    if LOGDIR.exists():
        shutil.rmtree(LOGDIR)

    MONITOR_DIR = Path(__file__).parent / "mp4"
    if MONITOR_DIR.exists():
        shutil.rmtree(MONITOR_DIR)

    summary_writer = tf.summary.create_file_writer(str(LOGDIR))

    agent = IQLAgent(env_id)

    tf_dataset = load_dataset(dataset_path="bipedalwalker.tfrecord", batch_size=32)

    for n, minibatch in enumerate(tf_dataset):
        states, actions, rewards, next_states, dones = minibatch

        vloss = agent.update_value(states, actions)
        ploss = agent.update_policy(states, actions)
        qloss = agent.update_q(states, actions, rewards, dones, next_states)
        agent.sync_target_weight()

        with summary_writer.as_default():
            tf.summary.scalar("loss_v", vloss, step=n)
            tf.summary.scalar("loss_p", ploss, step=n)
            tf.summary.scalar("loss_q", qloss, step=n)

        if n % 5000 == 0:
            score = agent.test_play(tag=f"{n}", monitor_dir=MONITOR_DIR)
            with summary_writer.as_default():
                tf.summary.scalar("test", score, step=n)

    agent.save("checkpoints/")


if __name__ == '__main__':
    main()
