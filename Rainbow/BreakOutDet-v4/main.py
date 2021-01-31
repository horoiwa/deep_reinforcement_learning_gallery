import collections
import shutil
from pathlib import Path

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import util
from buffer import NstepPrioritizedReplayBuffer
from model import create_network


class RainbowAgent:

    def __init__(self, env_name="BreakoutDeterministic-v4",
                 gamma=0.99,
                 batch_size=32,
                 lr=0.00025,
                 reward_clip=True,
                 update_period=4,
                 target_update_period=10000,
                 n_frames=4, alpha=0.6, beta=0.4, total_steps=2500000,
                 nstep_return=3, buffer_size=1000000,
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

        if use_categorical:

            self.n_atoms = 51

            self.Vmin, self.Vmax = -10, 10

            self.delta_z = (self.Vmax - self.Vmin) / (self.n_atoms - 1)

            self.Z = np.linspace(self.Vmin, self.Vmax, self.n_atoms)
        else:
            self.n_atoms = None

            self.Z = None

        env = gym.make(self.env_name)

        self.action_space = env.action_space.n

        self.qnet = create_network(
            self.action_space, use_dueling, use_categorical, use_noisy,
            n_atoms=self.n_atoms, Z=self.Z)

        self.target_qnet = create_network(
            self.action_space, use_dueling, use_categorical, use_noisy,
            n_atoms=self.n_atoms, Z=self.Z)

        self.optimizer = Adam(lr=lr, epsilon=0.01/self.batch_size)

        self.nstep_return = nstep_return

        self.replay_buffer = NstepPrioritizedReplayBuffer(
            max_len=buffer_size, gamma=self.gamma,
            alpha=alpha, beta=beta, nstep_return=nstep_return,
            reward_clip=reward_clip)

        self.steps = 0

    def __post_init__(self):
        env = gym.make(self.env_name)
        frame = env.step(np.random.choice(np.arange(env.action_space.n)))
        frame = util.preprocess_frame(frame)
        state = np.array([frame] * self.n_frames)
        self.qnet(state)
        self.target_qnet(state)
        self.target_qnet.set_weights(self.qnet.get_weights)

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

                #if len(self.replay_buffer) >= 50000:
                if len(self.replay_buffer) >= 500:
                    if self.steps % self.update_period == 0:
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

                if done:
                    break

            print(f"Episode: {episode}, score: {episode_rewards}, steps: {episode_steps}")
            if episode % 20 == 0:
                test_scores, test_steps = self.test_play(n_testplay=1)
                with self.summary_writer.as_default():
                    tf.summary.scalar("test_score", test_scores[0], step=self.steps)
                    tf.summary.scalar("test_step", test_steps[0], step=self.steps)

            if episode % 500 == 0:
                self.qnet.save_weights("checkpoints/qnet")

    def update_network(self):

        #: ミニバッチの作成
        indices, weights, (states, actions, rewards, next_states, dones) = self.replay_buffer.get_minibatch(self.batch_size, self.steps)

        #: Double DQN
        next_actions, _ = self.qnet.sample_actions(next_states)
        _, next_qvalues = self.target_qnet.sample_actions(next_states)

        next_actions_onehot = tf.one_hot(next_actions, self.action_space)
        max_next_qvalues = tf.reduce_sum(
            next_qvalues * next_actions_onehot, axis=1, keepdims=True)

        target_q = rewards + self.gamma ** (self.nstep_return) * (1 - dones) * max_next_qvalues
        import pdb; pdb.set_trace()
        with tf.GradientTape() as tape:

            qvalues = self.qnet(states)
            actions_onehot = tf.one_hot(
                actions.flatten().astype(np.int32), self.action_space)
            q = tf.reduce_sum(
                qvalues * actions_onehot, axis=1, keepdims=True)
            td_errors = target_q - q
            weighted_td_errors = weights * td_errors
            loss = util.huber_loss(weighted_td_errors)

        grads = tape.gradient(loss, self.qnet.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.qnet.trainable_variables))

        #: update priority of experiences
        td_errors = td_errors.numpy().flatten()
        self.replay_buffer.update_priority(indices, td_errors)

        return loss

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
    agent = RainbowAgent(use_noisy=False, use_dueling=False,
                         use_priority=False, use_multistep=False,
                         use_categorical=True)
    agent.learn(n_episodes=5001)
    agent.qnet.save_weights("checkpoints/qnet_fin")
    agent.test_play(n_testplay=5,
                    checkpoint_path="checkpoints/qnet",
                    monitor_dir="mp4")


if __name__ == '__main__':
    main()
