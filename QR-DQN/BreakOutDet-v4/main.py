from pathlib import Path
import shutil

import gym
import numpy as np
import tensorflow as tf
import collections

from model import QuantileQNetwork
from buffer import Experience, ReplayBuffer
from util import frame_preprocess


class QRDQNAgent:

    def __init__(self, env_name="BreakoutDeterministic-v4",
                 N=50, gamma=0.98,
                 n_frames=4, batch_size=32,
                 buffer_size=1000000,
                 update_period=8,
                 target_update_period=10000):

        self.env_name = env_name

        self.gamma = gamma

        self.N = N

        self.quantiles = [1/(2*N) + i * 1 / N for i in range(N)]

        self.k = 1.0

        self.n_frames = n_frames

        self.batch_size = batch_size

        self.update_period = update_period

        self.target_update_period = target_update_period

        self.action_space = gym.make(self.env_name).action_space.n

        self.qnet = QuantileQNetwork(actions_space=self.action_space, N=N)

        self.target_qnet = QuantileQNetwork(actions_space=self.action_space, N=N)

        self._define_network()

        self.replay_buffer = ReplayBuffer(max_len=buffer_size)

        self.optimizer = tf.keras.optimizers.Adam(lr=0.00025, epsilon=0.01/32)

        self.steps = 0

    def _define_network(self):
        """ initialize network weights
        """
        env = gym.make(self.env_name)
        frames = collections.deque(maxlen=4)
        frame = frame_preprocess(env.reset())
        for _ in range(self.n_frames):
            frames.append(frame)

        state = np.stack(frames, axis=2)[np.newaxis, ...]
        self.qnet(state)
        self.target_qnet(state)
        self.target_qnet.set_weights(self.qnet.get_weights())

    @property
    def epsilon(self):
        if self.steps <= 1000000:
            return max(0.99 * (1000000 - self.steps) / 1000000, 0.1)
        elif self.steps <= 2000000:
            return 0.05 + 0.05 * (2000000 - self.steps) / 2000000
        else:
            return 0.05

    def learn(self, n_episodes, logdir="log"):

        logdir = Path(__file__).parent / logdir
        if logdir.exists():
            shutil.rmtree(logdir)
        self.summary_writer = tf.summary.create_file_writer(str(logdir))

        for episode in range(1, n_episodes+1):
            env = gym.make(self.env_name)

            frames = collections.deque(maxlen=4)
            frame = frame_preprocess(env.reset())
            for _ in range(self.n_frames):
                frames.append(frame)

            episode_rewards = 0
            episode_steps = 0
            done = False
            lives = 5
            while not done:
                self.steps += 1
                episode_steps += 1
                state = np.stack(frames, axis=2)[np.newaxis, ...]
                action = self.qnet.sample_action(state, epsilon=self.epsilon)
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
                        #: life loss as episode ends
                        exp = Experience(state, action, reward, next_state, True)
                    else:
                        exp = Experience(state, action, reward, next_state, done)

                    self.replay_buffer.push(exp)

                if (len(self.replay_buffer) > 20000) and (self.steps % self.update_period == 0):
                #if (len(self.replay_buffer) > 500) and (self.steps % self.update_period == 0):
                    loss = self.update_network()

                    with self.summary_writer.as_default():
                        tf.summary.scalar("loss", loss, step=self.steps)
                        tf.summary.scalar("epsilon", self.epsilon, step=self.steps)
                        tf.summary.scalar("buffer_size", len(self.replay_buffer), step=self.steps)
                        tf.summary.scalar("train_score", episode_rewards, step=self.steps)
                        tf.summary.scalar("train_steps", episode_steps, step=self.steps)

                #: Target update
                if self.steps % self.target_update_period == 0:
                    self.target_qnet.set_weights(self.qnet.get_weights())

            print(f"Episode: {episode}, score: {episode_rewards}, steps: {episode_steps}")

            if episode % 20 == 0:
                test_scores, test_steps = self.test_play(n_testplay=1)
                with self.summary_writer.as_default():
                    tf.summary.scalar("test_score", test_scores[0], step=self.steps)
                    tf.summary.scalar("test_step", test_steps[0], step=self.steps)

            if episode % 500 == 0:
                self.qnet.save_weights("checkpoints/qnet")
                print("Model Saved")

    def update_network(self):
        (states, actions, rewards,
         next_states, dones) = self.replay_buffer.get_minibatch(self.batch_size)

        next_actions, next_quantile_values_all = self.target_qnet.sample_actions(next_states)

        assert self.batch_size == states.shape[0] == len(next_actions)

        next_actions_onehot = tf.one_hot(next_actions.numpy().flatten(), self.action_space)
        next_actions_mask = tf.expand_dims(next_actions_onehot, axis=2)
        next_quantile_values = tf.reduce_sum(
            next_quantile_values_all * next_actions_mask, axis=1, keepdims=True)

        target_quantile_values = np.zeros_like(next_quantile_values)
        for i in range(self.batch_size):
            target_quantile_values[i, ...] = rewards[i] + self.gamma * (1 - dones[i]) * next_quantile_values[i, ...]
        target_quantile_values = tf.repeat(target_quantile_values, self.N, axis=1)

        with tf.GradientTape() as tape:
            quantile_values_all = self.qnet(states)
            actions_onehot = tf.one_hot(
                actions.flatten().astype(np.int32), self.action_space)
            actions_mask = tf.expand_dims(actions_onehot, axis=2)
            quantile_values = tf.reduce_sum(
                quantile_values_all * actions_mask, axis=1, keepdims=True)
            quantile_values = tf.repeat(
                tf.transpose(quantile_values, [0, 2, 1]), self.N, axis=2)

            #: (batchsize, N, N)
            td_errors = target_quantile_values - quantile_values

            #: huberloss(k=1.0)
            is_smaller_than_k = tf.abs(td_errors) < self.k
            squared_loss = 0.5 * tf.square(td_errors)
            linear_loss = self.k * (tf.abs(td_errors) - 0.5 * self.k)
            huberloss = tf.where(is_smaller_than_k, squared_loss, linear_loss)

            #: quantile huberloss
            indicator = tf.stop_gradient(tf.where(td_errors < 0, 1., 0.))
            quantiles = tf.repeat(tf.expand_dims(self.quantiles, axis=1), self.N, axis=1)
            quantile_weights = tf.abs(quantiles - indicator)
            quantile_huberloss = quantile_weights * huberloss

            loss = tf.reduce_mean(quantile_huberloss, axis=2)
            loss = tf.reduce_sum(loss, axis=1)
            loss = tf.reduce_mean(loss)

        variables = self.qnet.trainable_variables
        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(
            zip(grads, variables))

        return loss

    def test_play(self, n_testplay=1, monitor_dir=None,
                  checkpoint_path=None):

        if checkpoint_path:
            env = gym.make(self.env_name)
            frames = collections.deque(maxlen=4)
            frame = frame_preprocess(env.reset())
            for _ in range(self.n_frames):
                frames.append(frame)
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

            frames = collections.deque(maxlen=4)

            frame = frame_preprocess(env.reset())
            for _ in range(self.n_frames):
                frames.append(frame)

            done = False
            episode_steps = 0
            episode_rewards = 0

            while not done:
                state = np.stack(frames, axis=2)[np.newaxis, ...]
                action = self.qnet.sample_action(state, epsilon=0.01)
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
    agent = QRDQNAgent()
    agent.learn(n_episodes=6001)
    agent.test_play(n_testplay=10,
                    checkpoint_path="checkpoints/qnet",
                    monitor_dir="mp4")


if __name__ == '__main__':
    main()
