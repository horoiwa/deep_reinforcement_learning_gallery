from pathlib import Path
import shutil

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import collections

from model import QNetwork
from buffer import Experience, ReplayBuffer
from util import preprocess_frame


class DQNAgent:

    def __init__(self, env_name="BreakoutDeterministic-v4",
                 gamma=0.99,
                 batch_size=32,
                 lr=0.00025,
                 update_period=4,
                 target_update_period=10000,
                 n_frames=4):

        self.env_name = env_name

        self.gamma = gamma

        self.batch_size = batch_size

        self.epsilon_scheduler = (
            lambda steps: max(1.0 - 0.9 * steps / 1000000, 0.1))

        self.update_period = update_period

        self.target_update_period = target_update_period

        env = gym.make(self.env_name)

        self.action_space = env.action_space.n

        self.qnet = QNetwork(self.action_space)

        self.target_qnet = QNetwork(self.action_space)

        self.optimizer = Adam(lr=lr, epsilon=0.01/self.batch_size)

        self.n_frames = n_frames

        self.use_reward_clipping = True

        self.huber_loss = tf.keras.losses.Huber()

    def learn(self, n_episodes, buffer_size=1000000, logdir="log"):

        logdir = Path(__file__).parent / logdir
        if logdir.exists():
            shutil.rmtree(logdir)
        self.summary_writer = tf.summary.create_file_writer(str(logdir))

        self.replay_buffer = ReplayBuffer(max_len=buffer_size)

        steps = 0
        for episode in range(1, n_episodes+1):
            env = gym.make(self.env_name)

            frame = preprocess_frame(env.reset())
            frames = collections.deque(
                [frame] * self.n_frames, maxlen=self.n_frames)

            episode_rewards = 0
            episode_steps = 0
            done = False
            lives = 5

            while not done:

                steps, episode_steps = steps + 1, episode_steps + 1

                epsilon = self.epsilon_scheduler(steps)

                state = np.stack(frames, axis=2)[np.newaxis, ...]

                action = self.qnet.sample_action(state, epsilon=epsilon)

                next_frame, reward, done, info = env.step(action)

                episode_rewards += reward

                frames.append(preprocess_frame(next_frame))

                next_state = np.stack(frames, axis=2)[np.newaxis, ...]

                if info["ale.lives"] != lives:
                    lives = info["ale.lives"]
                    transition = (state, action, reward, next_state, True)
                else:
                    transition = (state, action, reward, next_state, done)

                self.replay_buffer.push(transition)

                if len(self.replay_buffer) > 50000:
                    if steps % self.update_period == 0:
                        loss = self.update_network()
                        with self.summary_writer.as_default():
                            tf.summary.scalar("loss", loss, step=steps)
                            tf.summary.scalar("epsilon", epsilon, step=steps)
                            tf.summary.scalar("buffer_size", len(self.replay_buffer), step=steps)
                            tf.summary.scalar("train_score", episode_rewards, step=steps)
                            tf.summary.scalar("train_steps", episode_steps, step=steps)

                    if steps % self.target_update_period == 0:
                        self.target_qnet.set_weights(self.qnet.get_weights())

                if done:
                    break

            print(f"Episode: {episode}, score: {episode_rewards}, steps: {episode_steps}")
            if episode % 20 == 0:
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

        if self.use_reward_clipping:
            rewards = np.clip(rewards, -1, 1)

        #: Double DQN
        next_actions, _ = self.qnet.sample_actions(next_states)
        _, next_qvalues = self.target_qnet.sample_actions(next_states)

        next_actions_onehot = tf.one_hot(next_actions, self.action_space)
        max_next_qvalues = tf.reduce_sum(
            next_qvalues * next_actions_onehot, axis=1, keepdims=True)

        target_q = rewards + self.gamma * (1 - dones) * max_next_qvalues

        with tf.GradientTape() as tape:

            qvalues = self.qnet(states)
            actions_onehot = tf.one_hot(
                actions.flatten().astype(np.int32), self.action_space)
            q = tf.reduce_sum(
                qvalues * actions_onehot, axis=1, keepdims=True)
            loss = self.huber_loss(target_q, q)

        grads = tape.gradient(loss, self.qnet.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.qnet.trainable_variables))

        return loss

    def test_play(self, n_testplay=1, monitor_dir=None,
                  checkpoint_path=None):

        if checkpoint_path:
            env = gym.make(self.env_name)
            frame = preprocess_frame(env.reset())
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

            frame = preprocess_frame(env.reset())
            frames = collections.deque(
                [frame] * self.n_frames, maxlen=self.n_frames)

            done = False
            episode_steps = 0
            episode_rewards = 0

            while not done:
                state = np.stack(frames, axis=2)[np.newaxis, ...]
                action = self.qnet.sample_action(state, epsilon=0.05)
                next_frame, reward, done, _ = env.step(action)
                frames.append(preprocess_frame(next_frame))

                episode_rewards += reward
                episode_steps += 1
                if episode_steps > 500 and episode_rewards < 3:
                    #: ゲーム開始(action: 0)しないまま停滞するケースへの対処
                    break

            scores.append(episode_rewards)
            steps.append(episode_steps)

        return scores, steps


def main():
    agent = DQNAgent()
    agent.learn(n_episodes=5001)
    agent.qnet.save_weights("checkpoints/qnet_fin")
    agent.test_play(n_testplay=5,
                    checkpoint_path="checkpoints/qnet",
                    monitor_dir="mp4")


if __name__ == '__main__':
    main()
