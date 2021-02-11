import collections
import shutil
from pathlib import Path

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import util
from buffers import create_replaybuffer
from models import create_network


class RainbowAgent:

    def __init__(self, env_name="BreakoutDeterministic-v4",
                 gamma=0.99,
                 batch_size=32,
                 lr=0.00025,
                 reward_clip=True,
                 update_period=4,
                 target_update_period=10000,
                 n_frames=4, alpha=0.6, beta=0.4, total_steps=2500000,
                 buffer_size=1000000,
                 Vmin=-10, Vmax=10, n_atoms=51,
                 use_noisy=False, use_priority=False, use_dueling=False,
                 use_multistep=False, use_categorical=False):

        self.use_noisy = use_noisy

        self.use_priority = use_priority

        self.use_dueling = use_dueling

        self.use_multistep = use_multistep

        self.use_categorical = use_categorical

        self.env_name = env_name

        self.gamma = gamma

        self.batch_size = batch_size

        self.n_frames = n_frames

        self.update_period = update_period

        self.target_update_period = target_update_period

        #: Categorical DQN
        self.n_atoms = n_atoms

        self.Vmin, self.Vmax = Vmin, Vmax

        self.delta_z = (self.Vmax - self.Vmin) / (self.n_atoms - 1)

        self.Z = np.linspace(self.Vmin, self.Vmax, self.n_atoms)

        #: Multistep-Q-learning
        if self.use_multistep:
            self.nstep_return = 3
        else:
            self.nstep_return = 1

        env = gym.make(self.env_name)

        self.action_space = env.action_space.n

        self.qnet = create_network(
            self.action_space, use_dueling, use_categorical, use_noisy,
            Vmin=self.Vmin, Vmax=self.Vmax, n_atoms=self.n_atoms)

        self.target_qnet = create_network(
            self.action_space, use_dueling, use_categorical, use_noisy,
            Vmin=self.Vmin, Vmax=self.Vmax, n_atoms=self.n_atoms)

        self.optimizer = Adam(lr=lr, epsilon=0.01 / self.batch_size)

        self.replay_buffer = create_replaybuffer(
                use_priority=self.use_priority,
                use_multistep=self.use_multistep,
                max_len=buffer_size,
                nstep_return=self.nstep_return, gamma=self.gamma,
                alpha=alpha, beta=beta, total_steps=total_steps,
                reward_clip=reward_clip)

        self.steps = 0

    @property
    def epsilon(self):
        if self.use_noisy:
            return 0
        else:
            return max(1.0 - 0.9 * self.steps / 1000000, 0.1)

    def learn(self, n_episodes, logdir="log"):

        logdir = Path(__file__).parent / logdir
        if logdir.exists():
            shutil.rmtree(logdir)
        self.summary_writer = tf.summary.create_file_writer(str(logdir))

        for episode in range(1, n_episodes+1):
            env = gym.make(self.env_name)

            frame = util.preprocess_frame(env.reset())
            frames = collections.deque(
                [frame] * self.n_frames, maxlen=self.n_frames)

            episode_rewards = 0
            episode_steps = 0
            done = False
            lives = 5
            while not done:

                self.steps, episode_steps = self.steps + 1, episode_steps + 1

                state = np.stack(frames, axis=2)[np.newaxis, ...]

                action = self.qnet.sample_action(state, self.epsilon)

                next_frame, reward, done, info = env.step(action)

                episode_rewards += reward

                frames.append(util.preprocess_frame(next_frame))

                next_state = np.stack(frames, axis=2)[np.newaxis, ...]

                if info["ale.lives"] != lives:
                    lives = info["ale.lives"]
                    transition = (state, action, reward, next_state, True)
                else:
                    transition = (state, action, reward, next_state, done)

                self.replay_buffer.push(transition)

                if len(self.replay_buffer) >= 50000:
                    if self.steps % self.update_period == 0:

                        if self.use_categorical:
                            loss = self.update_categorical_network()
                        else:
                            loss = self.update_network()

                        with self.summary_writer.as_default():
                            tf.summary.scalar(
                                "loss", loss, step=self.steps)
                            tf.summary.scalar(
                                "buffer_size", len(self.replay_buffer), step=self.steps)
                            tf.summary.scalar(
                                "epsilon", self.epsilon, step=self.steps)
                            tf.summary.scalar(
                                "train_score", episode_rewards, step=self.steps)
                            tf.summary.scalar(
                                "train_steps", episode_steps, step=self.steps)

                    if self.steps % self.target_update_period == 0:
                        self.target_qnet.set_weights(self.qnet.get_weights())

            print(f"Episode: {episode}, score: {episode_rewards}, steps: {episode_steps}")
            if episode % 20 == 0:
                test_scores, test_steps = self.test_play(n_testplay=1)
                with self.summary_writer.as_default():
                    tf.summary.scalar("test_score", test_scores[0], step=self.steps)
                    tf.summary.scalar("test_step", test_steps[0], step=self.steps)
                    for layer in self.qnet.layers[-3:]:
                        for var in layer.variables:
                            tf.summary.histogram(var.name, var, step=self.steps)

            if episode % 500 == 0:
                self.qnet.save_weights("checkpoints/qnet")

    def update_network(self):

        #: ミニバッチの作成
        if self.use_priority:
            indices, weights, (states, actions, rewards, next_states, dones) = self.replay_buffer.get_minibatch(self.batch_size, self.steps)
            weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.get_minibatch(self.batch_size)

        #: Double DQN
        next_actions, _ = self.qnet.sample_actions(next_states)
        _, next_qvalues = self.target_qnet.sample_actions(next_states)

        next_actions_onehot = tf.one_hot(next_actions, self.action_space)
        max_next_qvalues = tf.reduce_sum(
            next_qvalues * next_actions_onehot, axis=1, keepdims=True)

        target_q = rewards + self.gamma ** (self.nstep_return) * (1 - dones) * max_next_qvalues

        with tf.GradientTape() as tape:

            qvalues = self.qnet(states)
            actions_onehot = tf.one_hot(
                actions.flatten().astype(np.int32), self.action_space)
            q = tf.reduce_sum(
                qvalues * actions_onehot, axis=1, keepdims=True)
            td_loss = util.huber_loss(target_q, q)

            if self.use_priority:
                loss = tf.reduce_mean(weights * td_loss)
            else:
                loss = tf.reduce_mean(td_loss)

        grads = tape.gradient(loss, self.qnet.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.qnet.trainable_variables))

        #: update priority of experiences
        if self.use_priority:
            td_errors = td_loss.numpy().flatten()
            self.replay_buffer.update_priority(indices, td_errors)

        return loss

    def update_categorical_network(self):
        #: ミニバッチの作成
        if self.use_priority:
            indices, weights, (states, actions, rewards, next_states, dones) = self.replay_buffer.get_minibatch(self.batch_size, self.steps)
            weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.get_minibatch(self.batch_size)

        next_actions, _ = self.qnet.sample_actions(next_states)
        _, next_probs = self.target_qnet.sample_actions(next_states)

        #: 選択されたactionの確率分布だけ抽出する
        onehot_mask = self.create_mask(next_actions)
        next_dists = tf.reduce_sum(next_probs * onehot_mask, axis=1).numpy()

        #: 分布版ベルマンオペレータの適用
        target_dists = self.shift_and_projection(rewards, dones, next_dists)

        onehot_mask = self.create_mask(actions)
        with tf.GradientTape() as tape:
            probs = self.qnet(states)
            dists = tf.reduce_sum(probs * onehot_mask, axis=1)

            #: クリップしないとlogとったときに勾配爆発することがある
            dists = tf.clip_by_value(dists, 1e-6, 1.0)
            loss = tf.reduce_sum(
                -1 * target_dists * tf.math.log(dists), axis=1, keepdims=True)

            if self.use_priority:
                weighted_loss = weights * loss
                loss = tf.reduce_mean(weighted_loss)
            else:
                loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, self.qnet.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.qnet.trainable_variables))

        if self.use_priority:
            td_errors = weighted_loss.numpy().flatten()
            self.replay_buffer.update_priority(indices, td_errors)

        return loss

    def shift_and_projection(self, rewards, dones, next_dists):

        target_dists = np.zeros((self.batch_size, self.n_atoms))

        for j in range(self.n_atoms):

            tZ_j = np.minimum(
                self.Vmax, np.maximum(self.Vmin, rewards + self.gamma ** (self.nstep_return) * self.Z[j]))
            bj = (tZ_j - self.Vmin) / self.delta_z

            lower_bj = np.floor(bj).astype(np.int8)
            upper_bj = np.ceil(bj).astype(np.int8)

            eq_mask = lower_bj == upper_bj
            neq_mask = lower_bj != upper_bj

            lower_probs = 1 - (bj - lower_bj)
            upper_probs = 1 - (upper_bj - bj)

            next_dist = next_dists[:, [j]]
            indices = np.arange(self.batch_size).reshape(-1, 1)

            target_dists[indices[neq_mask], lower_bj[neq_mask]] += (lower_probs * next_dist)[neq_mask]
            target_dists[indices[neq_mask], upper_bj[neq_mask]] += (upper_probs * next_dist)[neq_mask]

            target_dists[indices[eq_mask], lower_bj[eq_mask]] += (0.5 * next_dist)[eq_mask]
            target_dists[indices[eq_mask], upper_bj[eq_mask]] += (0.5 * next_dist)[eq_mask]

        """ 2. doneへの対処
            doneのときは TZ(t) = R(t)
        """
        for batch_idx in range(self.batch_size):

            if not dones[batch_idx]:
                continue
            else:
                target_dists[batch_idx, :] = 0
                tZ = np.minimum(self.Vmax, np.maximum(self.Vmin, rewards[batch_idx]))
                bj = (tZ - self.Vmin) / self.delta_z

                lower_bj = np.floor(bj).astype(np.int32)
                upper_bj = np.ceil(bj).astype(np.int32)

                if lower_bj == upper_bj:
                    target_dists[batch_idx, lower_bj] += 1.0
                else:
                    target_dists[batch_idx, lower_bj] += 1 - (bj - lower_bj)
                    target_dists[batch_idx, upper_bj] += 1 - (upper_bj - bj)

        return target_dists

    def create_mask(self, actions):

        mask = np.ones((self.batch_size, self.action_space, self.n_atoms))
        actions_onehot = tf.one_hot(
            tf.cast(actions, tf.int32), self.action_space, axis=1)

        for idx in range(self.batch_size):
            mask[idx, ...] = mask[idx, ...] * actions_onehot[idx, ...]

        return mask

    def test_play(self, n_testplay=1, monitor_dir=None,
                  checkpoint_path=None):

        if checkpoint_path:
            env = gym.make(self.env_name)
            frame = util.preprocess_frame(env.reset())
            frames = collections.deque(
                [frame] * self.n_frames, maxlen=self.n_frames)

            state = np.stack(frames, axis=2)[np.newaxis, ...]
            self.qnet(state)
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

            frame = util.preprocess_frame(env.reset())
            frames = collections.deque(
                [frame] * self.n_frames, maxlen=self.n_frames)

            done = False
            episode_steps = 0
            episode_rewards = 0

            while not done:
                state = np.stack(frames, axis=2)[np.newaxis, ...]
                epsilon = 0 if self.use_noisy else 0.05
                action = self.qnet.sample_action(state, epsilon)
                next_frame, reward, done, _ = env.step(action)
                frames.append(util.preprocess_frame(next_frame))

                episode_rewards += reward
                episode_steps += 1
                if episode_steps > 500 and episode_rewards < 3:
                    #: ゲーム開始(action: 0)しないまま停滞するケースへの対処
                    break

            scores.append(episode_rewards)
            steps.append(episode_steps)

        return scores, steps


def main():
    agent = RainbowAgent(use_noisy=True, use_dueling=True,
                         use_priority=True, use_multistep=True,
                         use_categorical=True)
    agent.learn(n_episodes=5001)
    agent.test_play(n_testplay=5,
                    checkpoint_path="checkpoints/qnet",
                    monitor_dir="mp4")


if __name__ == '__main__':
    main()
