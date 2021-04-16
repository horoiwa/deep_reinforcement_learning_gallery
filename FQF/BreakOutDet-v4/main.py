from pathlib import Path
import shutil

import gym
import numpy as np
import tensorflow as tf
import collections

from models import FQFNetwork
from buffer import Experience, ReplayBuffer
from util import frame_preprocess


class FQFAgent:

    def __init__(self, env_name,
                 num_quantiles=24, fqf_factor=0.000001, ent_coef=0.001,
                 state_embedding_dim=3136, quantile_embedding_dim=64,
                 gamma=0.99, n_frames=4, batch_size=32,
                 buffer_size=1000000,
                 update_period=8,
                 target_update_period=10000):

        self.env_name = env_name

        self.num_quantiles = num_quantiles

        self.state_embedding_dim = state_embedding_dim

        self.quantile_embedding_dim = quantile_embedding_dim

        self.k = 1.0

        self.ent_coef = ent_coef

        self.n_frames = n_frames

        self.action_space = gym.make(self.env_name).action_space.n

        self.fqf_network = FQFNetwork(
            action_space=self.action_space,
            num_quantiles=self.num_quantiles,
            state_embedding_dim=self.state_embedding_dim,
            quantile_embedding_dim=self.quantile_embedding_dim)

        self.target_fqf_network = FQFNetwork(
            action_space=self.action_space,
            num_quantiles=self.num_quantiles,
            state_embedding_dim=self.state_embedding_dim,
            quantile_embedding_dim=self.quantile_embedding_dim)

        self._define_network()

        self.optimizer = tf.keras.optimizers.Adam(
            lr=0.00015, epsilon=0.01/32)

        #: fpl; fraction proposal layer
        self.optimizer_fpl = tf.keras.optimizers.Adam(
            learning_rate=0.00005 * fqf_factor,
            epsilon=0.0003125)

        self.gamma = gamma

        self.replay_buffer = ReplayBuffer(max_len=buffer_size)

        self.batch_size = batch_size

        self.update_period = update_period

        self.target_update_period = target_update_period

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
        self.fqf_network(state)
        self.target_fqf_network(state)
        self.target_fqf_network.set_weights(self.fqf_network.get_weights())

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
                action = self.fqf_network.sample_action(state, epsilon=self.epsilon)
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
                        #: life loss as episode ends
                        lives = info["ale.lives"]
                        exp = Experience(state, action, reward, next_state, True)
                    else:
                        exp = Experience(state, action, reward, next_state, done)

                    self.replay_buffer.push(exp)

                if (len(self.replay_buffer) > 20000) and (self.steps % self.update_period == 0):
                #if (len(self.replay_buffer) > 300) and (self.steps % self.update_period == 0):
                    loss, loss_tau, entropy = self.update_network()

                    with self.summary_writer.as_default():
                        tf.summary.scalar("loss", loss, step=self.steps)
                        tf.summary.scalar("epsilon", self.epsilon, step=self.steps)
                        tf.summary.scalar("buffer_size", len(self.replay_buffer), step=self.steps)
                        tf.summary.scalar("train_score", episode_rewards, step=self.steps)
                        tf.summary.scalar("train_steps", episode_steps, step=self.steps)

                #: Target update
                if self.steps % self.target_update_period == 0:
                    self.target_fqf_network.set_weights(
                        self.fqf_network.get_weights())

            print(f"Episode: {episode}, score: {episode_rewards}, steps: {episode_steps}")

            if episode % 20 == 0:
                test_scores, test_steps = self.test_play(n_testplay=1)
                with self.summary_writer.as_default():
                    tf.summary.scalar("test_score", test_scores[0], step=self.steps)
                    tf.summary.scalar("test_step", test_steps[0], step=self.steps)

            if episode % 500 == 0:
                self.fqf_network.save_weights("checkpoints/fqfnet")
                print("Model Saved")

    def update_network(self):

        (states, actions, rewards,
         next_states, dones) = self.replay_buffer.get_minibatch(self.batch_size)

        rewards = rewards.reshape((self.batch_size, 1, 1))
        dones = dones.reshape((self.batch_size, 1, 1))

        with tf.GradientTape() as tape:
            #: Compute F(τ^)
            state_embedded = self.fqf_network.state_embedding_layer(states)

            taus, taus_hat, taus_hat_probs = self.fqf_network.propose_fractions(state_embedded)
            tf.stop_gradient(taus_hat)

            quantiles = self.fqf_network.quantile_function(
                state_embedded, taus_hat)
            actions_onehot = tf.one_hot(
                actions.flatten().astype(np.int32), self.action_space)
            actions_mask = tf.expand_dims(actions_onehot, axis=2)
            quantiles = tf.reduce_sum(
                quantiles * actions_mask, axis=1, keepdims=True)

            #: Compute target F(τ^), use same taus proposed by online network
            next_actions, target_quantiles = self.target_fqf_network.greedy_action_on_given_taus(
                next_states, taus_hat, taus_hat_probs)

            next_actions_onehot = tf.one_hot(next_actions.numpy().flatten(), self.action_space)
            next_actions_mask = tf.expand_dims(next_actions_onehot, axis=2)
            target_quantiles = tf.reduce_sum(
                target_quantiles * next_actions_mask, axis=1, keepdims=True)

            #: TF(τ^)
            target_quantiles = rewards + self.gamma * (1-dones) * target_quantiles
            target_quantiles = tf.stop_gradient(target_quantiles)

            #: Compute Quantile regression loss
            target_quantiles = tf.repeat(
                target_quantiles, self.num_quantiles, axis=1)
            quantiles = tf.repeat(
                tf.transpose(quantiles, [0, 2, 1]), self.num_quantiles, axis=2)

            #: huberloss
            bellman_errors = target_quantiles - quantiles
            is_smaller_than_k = tf.abs(bellman_errors) < self.k
            squared_loss = 0.5 * tf.square(bellman_errors)
            linear_loss = self.k * (tf.abs(bellman_errors) - 0.5 * self.k)

            huberloss = tf.where(is_smaller_than_k, squared_loss, linear_loss)

            #: quantile loss
            indicator = tf.stop_gradient(tf.where(bellman_errors < 0, 1., 0.))
            _taus_hat = tf.repeat(
                tf.expand_dims(taus_hat, axis=2), self.num_quantiles, axis=2)

            quantile_factors = tf.abs(_taus_hat - indicator)
            quantile_huberloss = quantile_factors * huberloss

            loss = tf.reduce_mean(quantile_huberloss, axis=2),
            loss = tf.reduce_sum(loss, axis=1)
            loss = tf.reduce_mean(loss)

        state_embedding_vars = self.fqf_network.state_embedding_layer.trainable_variables
        quantile_function_vars = self.fqf_network.quantile_function.trainable_variables

        variables = state_embedding_vars + quantile_function_vars
        grads = tape.gradient(loss, variables)

        with tf.GradientTape() as tape2:
            taus_all = self.fqf_network.fraction_proposal_layer(state_embedded)
            taus = taus_all[:, 1:-1]

            quantiles = self.fqf_network.quantile_function(
                state_embedded, taus)
            taus_hat = (taus_all[:, 1:] + taus_all[:, :-1]) / 2.
            quantiles_hat = self.fqf_network.quantile_function(
                state_embedded, taus_hat)
            dw_dtau = 2 * quantiles - quantiles_hat[:, :, 1:] - quantiles_hat[:, :, :-1]
            dw_dtau = tf.reduce_sum(dw_dtau * actions_mask, axis=1)

            loss_fp = tf.reduce_mean(tf.square(dw_dtau), axis=1)
            loss_fp += self.ent_coef * entropy

        fp_variables = self.fqf_network.fraction_proposal_layer.trainable_variables
        grads_fp = tape2.gradient(
            loss_fp, fp_variables, output_gradients=dw_dtau)

        self.optimizer.apply_gradients(zip(grads, variables))
        self.optimizer_fpl.apply_gradients(zip(grads_fp, fp_variables))

        return loss, loss_fp, entropy

    def test_play(self, n_testplay=1, monitor_dir=None,
                  checkpoint_path=None):

        if checkpoint_path:
            env = gym.make(self.env_name)
            frames = collections.deque(maxlen=4)
            frame = frame_preprocess(env.reset())
            for _ in range(self.n_frames):
                frames.append(frame)
            state = np.stack(frames, axis=2)[np.newaxis, ...]
            self.fqf_network(state)
            self.fqf_network.load_weights(checkpoint_path)

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
                action = self.fqf_network.sample_action(state, epsilon=0.01)
                next_frame, reward, done, _ = env.step(action)
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
    agent = FQFAgent(env_name="BreakoutDeterministic-v4")
    agent.learn(n_episodes=6001)
    agent.test_play(n_testplay=10,
                    checkpoint_path="checkpoints/fqfnet",
                    monitor_dir="mp4")


def debug():
    agent = FQFAgent(env_name="BreakoutDeterministic-v4")
    env = gym.make("BreakoutDeterministic-v4")
    frames = collections.deque(maxlen=4)
    frame = frame_preprocess(env.reset())
    for _ in range(4):
        frames.append(frame)
    state = np.stack(frames, axis=2)[np.newaxis, ...]

    agent.fqf_network.load_weights("checkpoints/fqfnet")
    for _ in range(64):
        state = np.stack(frames, axis=2)[np.newaxis, ...]
        action = agent.fqf_network.sample_action(state, epsilon=0.1)
        next_frame, reward, done, info = env.step(action)
        frames.append(frame_preprocess(next_frame))
        next_state = np.stack(frames, axis=2)[np.newaxis, ...]
        exp = Experience(state, action, reward, next_state, done)
        agent.replay_buffer.push(exp)

    loss = agent.update_network()
    fqf = agent.fqf_network
    state_embedded = fqf.state_embedding_layer(state)


if __name__ == '__main__':
    # main()
    debug()
