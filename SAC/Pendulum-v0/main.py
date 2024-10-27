from pathlib import Path
import shutil

import numpy as np
import tensorflow as tf
import gym
from gym import wrappers

from models import GaussianPolicy, DualQNetwork
from buffer import ReplayBuffer, Experience


class SAC:

    MAX_EXPERIENCES = 100000

    MIN_EXPERIENCES = 512

    UPDATE_PERIOD = 4

    GAMMA = 0.99

    TAU = 0.005

    BATCH_SIZE = 256

    def __init__(self, env_id, action_space, action_bound):

        self.env_id = env_id

        self.action_space = action_space

        self.action_bound = action_bound

        self.env = gym.make(self.env_id)

        self.replay_buffer = ReplayBuffer(max_len=self.MAX_EXPERIENCES)

        self.policy = GaussianPolicy(action_space=self.action_space,
                                     action_bound=self.action_bound)

        self.duqlqnet = DualQNetwork()

        self.target_dualqnet = DualQNetwork()

        self.log_alpha = tf.Variable(0.)  #: alpha=1

        self.alpha_optimizer = tf.keras.optimizers.Adam(3e-4)

        self.target_entropy = -0.5 * self.action_space

        self.global_steps = 0

        self._initialize_weights()

    def _initialize_weights(self):
        """1度callすることでネットワークの重みを初期化
        """

        env = gym.make(self.env_id)

        dummy_state = env.reset()
        dummy_state = (dummy_state[np.newaxis, ...]).astype(np.float32)

        dummy_action = np.random.normal(0, 0.1, size=self.action_space)
        dummy_action = (dummy_action[np.newaxis, ...]).astype(np.float32)

        self.policy(dummy_state)

        self.duqlqnet(dummy_state, dummy_action)
        self.target_dualqnet(dummy_state, dummy_action)
        self.target_dualqnet.set_weights(self.duqlqnet.get_weights())

    def play_episode(self):

        episode_reward = 0

        local_steps = 0

        done = False

        state = self.env.reset()

        while not done:

            action, _ = self.policy.sample_action(np.atleast_2d(state))

            action = action.numpy()[0]

            next_state, reward, done, _ = self.env.step(action)

            exp = Experience(state, action, reward, next_state, done)

            self.replay_buffer.push(exp)

            state = next_state

            episode_reward += reward

            local_steps += 1

            self.global_steps += 1

            if (len(self.replay_buffer) >= self.MIN_EXPERIENCES
               and self.global_steps % self.UPDATE_PERIOD == 0):

                self.update_networks()

        return episode_reward, local_steps, tf.exp(self.log_alpha)

    def update_networks(self):

        (states, actions, rewards,
         next_states, dones) = self.replay_buffer.get_minibatch(self.BATCH_SIZE)

        alpha = tf.math.exp(self.log_alpha)

        #: Update Q-function
        next_actions, next_logprobs = self.policy.sample_action(next_states)

        target_q1, target_q2 = self.target_dualqnet(next_states, next_actions)

        target = rewards + (1 - dones) * self.GAMMA * (
            tf.minimum(target_q1, target_q2) + -1 * alpha * next_logprobs
            )

        with tf.GradientTape() as tape:
            q1, q2 = self.duqlqnet(states, actions)
            loss_1 = tf.reduce_mean(tf.square(target - q1))
            loss_2 = tf.reduce_mean(tf.square(target - q2))
            loss = 0.5 * loss_1 + 0.5 * loss_2

        variables = self.duqlqnet.trainable_variables
        grads = tape.gradient(loss, variables)
        self.duqlqnet.optimizer.apply_gradients(zip(grads, variables))

        #: Update policy
        with tf.GradientTape() as tape:
            selected_actions, logprobs = self.policy.sample_action(states)
            q1, q2 = self.duqlqnet(states, selected_actions)
            q_min = tf.minimum(q1, q2)
            loss = -1 * tf.reduce_mean(q_min + -1 * alpha * logprobs)

        variables = self.policy.trainable_variables
        grads = tape.gradient(loss, variables)
        self.policy.optimizer.apply_gradients(zip(grads, variables))

        #: Adjust alpha
        entropy_diff = -1 * logprobs - self.target_entropy
        with tf.GradientTape() as tape:
            tape.watch(self.log_alpha)
            selected_actions, logprobs = self.policy.sample_action(states)
            alpha_loss = tf.reduce_mean(tf.exp(self.log_alpha) * entropy_diff)

        grad = tape.gradient(alpha_loss, self.log_alpha)
        self.alpha_optimizer.apply_gradients([(grad, self.log_alpha)])

        #: Soft target update
        self.target_dualqnet.set_weights(
           (1 - self.TAU) * np.array(self.target_dualqnet.get_weights())
           + self.TAU * np.array(self.duqlqnet.get_weights())
           )

    def save_model(self):

        self.policy.save_weights("checkpoints/actor")

        self.duqlqnet.save_weights("checkpoints/critic")

    def load_model(self):

        self.policy.load_weights("checkpoints/actor")

        self.duqlqnet.load_weights("checkpoints/critic")

        self.target_dualqnet.load_weights("checkpoints/critic")

    def testplay(self, n=1, monitordir=None):

        if monitordir:
            env = wrappers.Monitor(gym.make(self.env_id),
                                   monitordir, force=True,
                                   video_callable=(lambda ep: True))
        else:
            env = gym.make(self.env_id)

        total_rewards = []

        for _ in range(n):

            state = env.reset()

            done = False

            total_reward = 0

            while not done:

                action, _ = self.policy.sample_action(np.atleast_2d(state))

                action = action.numpy()[0]

                next_state, reward, done, _ = env.step(action)

                total_reward += reward

                if done:
                    break
                else:
                    state = next_state

            total_rewards.append(total_reward)
            print()
            print(total_reward)
            print()

        return total_rewards


def main(n_episodes, n_testplay=1, logging_steps=5):

    LOGDIR = Path(__file__).parent / "log"
    if LOGDIR.exists():
        shutil.rmtree(LOGDIR)

    summary_writer = tf.summary.create_file_writer(str(LOGDIR))

    agent = SAC(env_id="Pendulum-v0", action_space=1, action_bound=2)

    episode_rewards = []

    for n in range(n_episodes):

        episode_reward, episode_steps, alpha = agent.play_episode()

        episode_rewards.append(episode_reward)

        with summary_writer.as_default():
            tf.summary.scalar("episode_reward", episode_reward, step=n)
            tf.summary.scalar("episode_steps", episode_steps, step=n)
            tf.summary.scalar("alpha", alpha, step=n)

        if n % logging_steps == 0:
            print(f"Episode {n}: {episode_reward}, {episode_steps} steps")

    agent.save_model()

    if n_testplay:
        MONITOR_DIR = Path(__file__).parent / "mp4"
        if MONITOR_DIR.exists():
            shutil.rmtree(MONITOR_DIR)

        agent = SAC(env_id="Pendulum-v0", action_space=1, action_bound=2)
        agent.load_model()
        agent.testplay(n=n_testplay, monitordir=MONITOR_DIR)


if __name__ == '__main__':
    main(n_episodes=150, n_testplay=4)
