from pathlib import Path
import shutil

import gym
import numpy as np
import tensorflow as tf
import collections

from model import CategoricalQNet
from buffer import Experience, ReplayBuffer
from util import frame_preprocess


class CategoricalDQNAgent:

    def __init__(self, env_name="BreakoutDeterministic-v4",
                 n_atoms=51, Vmin=-10, Vmax=10, gamma=0.98,
                 n_frames=4, batch_size=32, lr=0.00025,
                 init_epsilon=0.95,
                 update_period=4,
                 target_update_period=10000):

        self.env_name = env_name

        self.n_atoms = n_atoms

        self.Vmin, self.Vmax = Vmin, Vmax

        self.Z = np.linspace(self.Vmin, self.Vmax, self.n_atoms)

        self.delta_z = (self.Vmax - self.Vmin) / self.n_atoms

        self.gamma = gamma

        self.n_frames = n_frames

        self.batch_size = batch_size

        self.init_epsilon = init_epsilon

        self.update_period = 4

        self.target_update_period = target_update_period

        env = gym.make(self.env_name)

        self.action_space = env.action_space.n

        self.qnet = CategoricalQNet(
            self.action_space, self.n_atoms, self.Z)

        self.target_qnet = CategoricalQNet(
            self.action_space, self.n_atoms, self.Z)

        self.optimizer = tf.keras.optimizers.Adam(lr=lr, epsilon=0.01/batch_size)

    def learn(self, n_episodes, buffer_size=800000, logdir="log"):

        logdir = Path(__file__).parent / logdir
        if logdir.exists():
            shutil.rmtree(logdir)
        self.summary_writer = tf.summary.create_file_writer(str(logdir))

        self.replay_buffer = ReplayBuffer(max_len=buffer_size)

        steps = 0
        for episode in range(n_episodes):
            env = gym.make(self.env_name)

            frames = collections.deque(maxlen=4)
            frame = frame_preprocess(env.reset())
            for _ in range(self.n_frames):
                frames.append(frame)

            #: ネットワーク重みの初期化
            state = np.stack(frames, axis=2)[np.newaxis, ...]
            self.qnet(state)
            self.target_qnet(state)
            self.target_qnet.set_weights(self.qnet.get_weights())

            episode_rewards = 0
            episode_steps = 0

            done = False
            lives = 5
            while not done:

                steps += 1
                episode_steps += 1

                epsilon = max(self.init_epsilon * (500000 - steps) / 500000, 0.1)

                state = np.stack(frames, axis=2)[np.newaxis, ...]
                action = self.qnet.sample_action(state, epsilon=epsilon)
                next_frame, reward, done, info = env.step(action)
                episode_rewards += reward
                frames.append(frame_preprocess(next_frame))
                next_state = np.stack(frames, axis=2)[np.newaxis, ...]

                if done:
                    exp = Experience(state, action, reward, next_state, done)
                    self.replay_buffer.push(exp)
                    break
                else:
                    if info["ale.lives"] != lives:
                        lives = info["ale.lives"]
                        exp = Experience(state, action, reward, next_state, True)
                    else:
                        exp = Experience(state, action, reward, next_state, done)

                    self.replay_buffer.push(exp)

                if (len(self.replay_buffer) > 20000) and (steps % self.update_period == 0):
                    loss = self.update_network()

                    with self.summary_writer.as_default():
                        tf.summary.scalar("loss", loss, step=steps)
                        tf.summary.scalar("epsilon", epsilon, step=steps)
                        tf.summary.scalar("buffer_size", len(self.replay_buffer), step=steps)

                #: Hard target update
                if steps % self.target_update_period == 0:
                    self.target_qnet.set_weights(self.qnet.get_weights())

            print(f"Episode: {episode}, score: {episode_rewards}, steps: {episode_steps}")

            if episode % 10 == 0:
                test_scores, test_steps = self.test_play(n_testplay=1)
                with self.summary_writer.as_default():
                    tf.summary.scalar("test_score", test_scores[0], step=steps)
                    tf.summary.scalar("test_step", test_steps[0], step=steps)

            if episode % 1000 == 0:
                self.qnet.save_weights("checkpoints/qnet")

    def update_network(self):

        #: ミニバッチの作成
        (states, actions, rewards,
         next_states, dones) = self.replay_buffer.get_minibatch(self.batch_size)

        next_actions, next_probs = self.target_qnet.sample_actions(next_states)

        #: 選択されたactionの確率分布だけ抽出する
        onehot_mask = self.create_mask(next_actions)
        next_dists = tf.reduce_sum(next_probs * onehot_mask, axis=1).numpy()

        #: 分布版ベルマンオペレータの適用
        target_dists = self.shift_and_projection(rewards, dones, next_dists)

        onehot_mask = self.create_mask(actions)
        with tf.GradientTape() as tape:
            probs = self.qnet(states)
            dists = tf.reduce_sum(probs * onehot_mask, axis=1)

            loss = tf.reduce_sum(
                -1 * target_dists * tf.math.log(dists), axis=1, keepdims=True)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, self.qnet.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.qnet.trainable_variables))

        return loss

    def shift_and_projection(self, rewards, dones, next_dists):

        target_dists = np.zeros((self.batch_size, self.n_atoms))

        Z = np.array([self.Z for _ in range(self.batch_size)])
        t_Z = self.gamma * Z + rewards

        """ TZ(t) = R(t) + γZ(t+1)
        """
        for atom_idx in range(self.n_atoms):

            indices = (t_Z[:, [atom_idx]] - self.Vmin) / self.delta_z

            lower_indices = np.floor(indices).astype(np.int8)
            lower_indices = np.clip(lower_indices, 0, self.n_atoms-1)

            upper_indices = np.ceil(indices).astype(np.int8)
            upper_indices = np.clip(upper_indices, 0, self.n_atoms-1)

            """ 1. はみだしへの対処
                lower_indices == upper_indices となる場合は,
                lower_prob + upper_prob = 2 になってしまうので
                この場合には均等に0.5を割り当てる
            """
            neq_mask = lower_indices != upper_indices
            eq_mask = lower_indices == upper_indices

            lower_probs = (1 - (indices - lower_indices)) * neq_mask
            lower_probs += 0.5 * np.ones((self.batch_size, 1)) * eq_mask

            upper_probs = 1 - (upper_indices - indices) * neq_mask
            upper_probs += 0.5 * np.ones((self.batch_size, 1)) * eq_mask

            next_dist = next_dists[:, [atom_idx]]

            tmp = [i for i in range(self.batch_size)]
            target_dists[tmp, lower_indices] += lower_probs * next_dist
            target_dists[tmp, upper_indices] += upper_probs * next_dist

        """ 2. doneへの対処
            doneのときは TZ(t) = R(t)
        """
        for batch_idx in range(self.batch_size):

            if not dones[batch_idx]:
                continue

            target_dists[batch_idx, :] = 0

            idx = (rewards[batch_idx] - self.Vmin) / self.delta_z

            lower_idx = np.clip(np.floor(idx).astype(np.int32), 0, self.n_atoms-1)
            upper_idx = np.clip(np.ceil(idx).astype(np.int32), 0, self.n_atoms-1)

            if lower_idx == upper_idx:
                target_dists[batch_idx, lower_idx] += 0.5
                target_dists[batch_idx, upper_idx] += 0.5
            else:
                target_dists[batch_idx, lower_idx] += idx - lower_idx
                target_dists[batch_idx, upper_idx] += upper_idx - idx

        return target_dists

    def create_mask(self, actions):

        mask = np.ones((self.batch_size, self.action_space, self.n_atoms))
        actions_onehot = tf.one_hot(
            tf.cast(actions, tf.int32), self.action_space, axis=1)

        for idx in range(self.batch_size):
            mask[idx, ...] = mask[idx, ...] * actions_onehot[idx, ...]

        return mask

    def test_play(self, n_testplay=1, monitor_dir=None, checkpoint_path=None):

        if checkpoint_path:
            self.qnet.load_weights(checkpoint_path)

        if monitor_dir:
            monitor_dir = Path(monitor_dir)
            if monitor_dir.exists():
                shutil.rmtree(monitor_dir)
            monitor_dir.mkdir()
            env = gym.wrappers.Monitor(
                gym.make(self.env_name), monitor_dir, force=True,
                video_callable=(lambda ep: True))
        else:
            env = gym.make(self.env_name)

        scores = []
        steps = []
        for _ in range(n_testplay):

            frames = collections.deque(maxlen=4)

            frame = frame_preprocess(env.reset())
            for _ in range(self.n_frames):
                frames.append(frame)

            done = False
            episode_steps = 0
            episode_rewards = 0

            while not done:
                state = np.stack(frames, axis=2)[np.newaxis, ...]
                action = self.qnet.sample_action(state, epsilon=0.1)
                next_frame, reward, done, info = env.step(action)
                frames.append(frame_preprocess(next_frame))

                episode_rewards += reward
                episode_steps += 1
                if episode_steps > 500 and episode_rewards < 3:
                    #: ゲーム開始(action: 0)しないまま停滞するケースへの対処
                    break

            scores.append(episode_rewards)
            steps.append(episode_steps)

        return scores, steps


def main():
    agent = CategoricalDQNAgent()
    agent.learn(n_episodes=7001)
    agent.test_play(n_testplay=3, monitor_dir="mp4")


if __name__ == '__main__':
    main()
