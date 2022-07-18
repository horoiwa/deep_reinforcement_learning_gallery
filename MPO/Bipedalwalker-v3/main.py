from pathlib import Path
import shutil
from dataclasses import dataclass
from collections import namedtuple

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import gym
from gym import wrappers
import numpy as np

from buffer import ReplayBuffer
from networks import MultiVariateGaussianPolicyNetwork, GaussianPolicyNetwork, QNetwork


Transition = namedtuple('Transition', ["state", "action", "reward", "next_state", "done"])


class MPOAgent:

    def __init__(self, env_id: str, logdir: Path):

        self.env_id = env_id

        self.summary_writer = tf.summary.create_file_writer(str(logdir)) if logdir else None

        self.action_space = gym.make(self.env_id).action_space.shape[0]

        self.replay_buffer = ReplayBuffer(maxlen=50000)

        self.policy = GaussianPolicyNetwork(action_space=self.action_space)
        self.target_policy = GaussianPolicyNetwork(action_space=self.action_space)

        self.critic = QNetwork()
        self.target_critic = QNetwork()

        self.log_temperature = tf.Variable(1.)
        self.temperature_optimizer = tf.keras.optimizers.Adam(lr=0.0005)

        self.log_alpha_mu = tf.Variable(1.)
        self.log_alpha_sigma = tf.Variable(1.)

        self.eps = 0.1

        self.eps_mu = 0.1
        self.eps_sigma = 0.01

        self.policy_optimizer = tf.keras.optimizers.Adam(lr=0.0005)
        self.critic_optimizer = tf.keras.optimizers.Adam(lr=0.0005)
        self.alpha_optimizer = tf.keras.optimizers.Adam(lr=0.0005)

        self.batch_size = 128

        self.n_samples = 10

        self.update_period = 4

        self.gamma = 0.99

        self.target_policy_update_period = 400

        self.target_critic_update_period = 400

        self.global_steps = 0

        self.episode_count = 0

        self.setup()

    def setup(self):
        """ Initialize network weights """

        env = gym.make(self.env_id)

        dummy_state = env.reset()
        dummy_state = (dummy_state[np.newaxis, ...]).astype(np.float32)

        dummy_action = np.random.normal(0, 0.1, size=self.action_space)
        dummy_action = (dummy_action[np.newaxis, ...]).astype(np.float32)

        self.policy(dummy_state)
        self.target_policy(dummy_state)

        self.critic(dummy_state, dummy_action)
        self.target_critic(dummy_state, dummy_action)

        self.target_policy.set_weights(self.policy.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    def save(self, save_dir):
        save_dir = Path(save_dir)

        self.policy.save_weights(str(save_dir / "policy"))
        self.critic.save_weights(str(save_dir / "critic"))

    def load(self, load_dir=None):
        load_dir = Path(load_dir)

        self.policy.load_weights(str(load_dir / "policy"))
        self.target_policy.load_weights(str(load_dir / "policy"))

        self.critic.load_weights(str(load_dir / "critic"))
        self.target_critic.load_weights(str(load_dir / "critic"))

    def rollout(self):

        episode_rewards, episode_steps = 0, 0

        done = False

        env = gym.make(self.env_id)

        state = env.reset()

        while not done:

            action = self.policy.sample_action(np.atleast_2d(state))

            action = action.numpy()[0]

            try:
                next_state, reward, done, _ = env.step(action)
            except Exception as err:
                print(err)
                import pdb; pdb.set_trace()

            #: Bipedalwalkerの転倒ペナルティ-100は大きすぎるためclip
            transition = Transition(state, action, np.clip(reward, 0., 1.), next_state, done)

            self.replay_buffer.add(transition)

            state = next_state

            episode_rewards += reward

            episode_steps += 1

            self.global_steps += 1

            if (len(self.replay_buffer) >= 5000 and self.global_steps % self.update_period == 0):
                self.update_networks()

            if self.global_steps % self.target_critic_update_period == 0:
                self.target_critic.set_weights(self.critic.get_weights())

            if self.global_steps % self.target_policy_update_period == 0:
                self.target_policy.set_weights(self.policy.get_weights())

        self.episode_count += 1
        with self.summary_writer.as_default():
            tf.summary.scalar("episode_reward_stp", episode_rewards, step=self.global_steps)
            tf.summary.scalar("episode_steps_stp", episode_steps, step=self.global_steps)
            tf.summary.scalar("episode_reward", episode_rewards, step=self.episode_count)
            tf.summary.scalar("episode_steps", episode_steps, step=self.episode_count)

        return episode_rewards, episode_steps

    def update_networks(self):

        (states, actions, rewards,
         next_states, dones) = self.replay_buffer.get_minibatch(batch_size=self.batch_size)

        B, M = self.batch_size, self.n_samples

        # [B, obs_dim] -> [B, obs_dim * M] -> [B * M, obs_dim]
        next_states_tiled = tf.reshape(
            tf.tile(next_states, multiples=(1, M)), shape=(B * M, -1)
            )

        target_mu, target_sigma = self.target_policy(next_states_tiled)

        # For MultivariateGaussianPolicy
        #target_dist = tfd.MultivariateNormalFullCovariance(loc=target_mu, covariance_matrix=target_sigma)

        # For IndependentGaussianPolicy
        target_dist = tfd.Independent(
            tfd.Normal(loc=target_mu, scale=target_sigma),
            reinterpreted_batch_ndims=1)

        sampled_actions = target_dist.sample()                             # [B * M,  action_dim]
        sampled_actions = tf.clip_by_value(sampled_actions, -1.0, 1.0)

        # Update Q-network:
        sampled_qvalues = tf.reshape(
            self.target_critic(next_states_tiled, sampled_actions),
            shape=(B, M, -1)
            )
        mean_qvalues = tf.reduce_mean(sampled_qvalues, axis=1)
        TQ = rewards + self.gamma * (1.0 - dones) * mean_qvalues

        with tf.GradientTape() as tape1:
            Q = self.critic(states, actions)
            loss_critic = tf.reduce_mean(tf.square(TQ - Q))

        variables = self.critic.trainable_variables
        grads = tape1.gradient(loss_critic, variables)
        grads, _ = tf.clip_by_global_norm(grads, 40.)
        self.critic_optimizer.apply_gradients(zip(grads, variables))

        # E-step:
        # Obtain η* by minimising g(η)
        with tf.GradientTape() as tape2:
            temperature = tf.math.softplus(self.log_temperature)
            q_logmeanexp = tf.math.log(
                tf.reduce_mean(tf.math.exp(sampled_qvalues / temperature), axis=1) + 1e-6
                )
            loss_temperature = temperature * (self.eps + tf.reduce_mean(q_logmeanexp, axis=0))

        grad = tape2.gradient(loss_temperature, self.log_temperature)
        self.temperature_optimizer.apply_gradients([(grad, self.log_temperature)])

        # Obtain sample-based variational distribution q(a|s)
        temperature = tf.math.softplus(self.log_temperature)

        # M-step: Optimize the lower bound J with respect to θ
        weights = tf.squeeze(
            tf.math.softmax(sampled_qvalues / temperature, axis=1), axis=2)    # [B, M, 1]

        with tf.GradientTape(persistent=True) as tape3:

            online_mu, online_sigma = self.policy(next_states_tiled)

            # For MultivariateGaussianPolicy
            #online_dist = tfd.MultivariateNormalFullCovariance(loc=online_mu, covariance_matrix=online_sigma)

            # For IndependentGaussianPolicy
            online_dist = tfd.Independent(
                tfd.Normal(loc=online_mu, scale=online_sigma),
                reinterpreted_batch_ndims=1)

            log_probs = tf.reshape(
                online_dist.log_prob(sampled_actions) + 1e-6,
                shape=(B, M))                                                    # [B * M, ] -> [B, M]

            cross_entropy_qp = tf.reduce_sum(weights * log_probs, axis=1)      # [B, M] -> [B,]

            # For MultivariateGaussianPolicy
            # online_dist_fixedmu = tfd.MultivariateNormalFullCovariance(loc=target_mu, covariance_matrix=online_sigma)
            # online_dist_fixedsigma = tfd.MultivariateNormalFullCovariance(loc=online_mu, covariance_matrix=target_sigma)

            # For IndependentGaussianPolicy
            online_dist_fixedmu = tfd.Independent(
                tfd.Normal(loc=target_mu, scale=online_sigma),
                reinterpreted_batch_ndims=1)
            online_dist_fixedsigma = tfd.Independent(
                tfd.Normal(loc=online_mu, scale=target_sigma),
                reinterpreted_batch_ndims=1)

            kl_mu = tf.reshape(
                target_dist.kl_divergence(online_dist_fixedsigma), shape=(B, M))  # [B * M, ] -> [B, M]

            kl_sigma = tf.reshape(
                target_dist.kl_divergence(online_dist_fixedmu), shape=(B, M))     # [B * M, ] -> [B, M]

            alpha_mu = tf.math.softplus(self.log_alpha_mu)
            alpha_sigma = tf.math.softplus(self.log_alpha_sigma)

            loss_policy = - cross_entropy_qp                                      # [B,]
            loss_policy += tf.stop_gradient(alpha_mu) * tf.reduce_mean(kl_mu, axis=1)
            loss_policy += tf.stop_gradient(alpha_sigma) * tf.reduce_mean(kl_sigma, axis=1)

            loss_policy = tf.reduce_mean(loss_policy)                            # [B,] -> [1]

            loss_alpha_mu = tf.reduce_mean(
                alpha_mu * tf.stop_gradient(self.eps_mu - tf.reduce_mean(kl_mu, axis=1))
                )

            loss_alpha_sigma = tf.reduce_mean(
                alpha_sigma * tf.stop_gradient(self.eps_sigma - tf.reduce_mean(kl_sigma, axis=1))
                )

            loss_alpha = loss_alpha_mu + loss_alpha_sigma

        variables = self.policy.trainable_variables
        grads = tape3.gradient(loss_policy, variables)
        grads, _ = tf.clip_by_global_norm(grads, 40.)
        self.policy_optimizer.apply_gradients(zip(grads, variables))

        variables = [self.log_alpha_mu, self.log_alpha_sigma]
        grads = tape3.gradient(loss_alpha, variables)
        grads, _ = tf.clip_by_global_norm(grads, 40.)
        self.alpha_optimizer.apply_gradients(zip(grads, variables))

        del tape3

        with self.summary_writer.as_default():
            tf.summary.scalar("loss_policy", loss_policy, step=self.global_steps)
            tf.summary.scalar("loss_critic", loss_critic, step=self.global_steps)
            tf.summary.scalar("sigma", tf.reduce_mean(online_sigma), step=self.global_steps)
            tf.summary.scalar("kl_mu", tf.reduce_mean(kl_mu), step=self.global_steps)
            tf.summary.scalar("kl_sigma", tf.reduce_mean(kl_sigma), step=self.global_steps)
            tf.summary.scalar("temperature", temperature, step=self.global_steps)
            tf.summary.scalar("alpha_mu", alpha_mu, step=self.global_steps)
            tf.summary.scalar("alpha_sigma", alpha_sigma, step=self.global_steps)
            tf.summary.scalar("replay_buffer", len(self.replay_buffer), step=self.global_steps)

    def testplay(self, name, monitor_dir):

        total_rewards = []

        env = wrappers.RecordVideo(
            gym.make(self.env_id),
            video_folder=monitor_dir,
            step_trigger=lambda i: True,
            name_prefix=name
        )

        state = env.reset()

        done = False

        total_reward = 0

        while not done:

            action = self.policy.sample_action(np.atleast_2d(state))

            action = action.numpy()[0]

            next_state, reward, done, _ = env.step(action)

            total_reward += reward

            state = next_state

        total_rewards.append(total_reward)

        print(f"{name}", total_reward)


def train(env_id="BipedalWalker-v3", n_episodes=1500):
    """
    Note:
        if you failed to "pip install gym[box2d]", try "pip install box2d"
    """
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"   # Needed only for ubuntu

    LOGDIR = Path(__file__).parent / "log"
    if LOGDIR.exists():
        shutil.rmtree(LOGDIR)

    MONITOR_DIR = Path(__file__).parent / "mp4"
    if MONITOR_DIR.exists():
        shutil.rmtree(MONITOR_DIR)

    agent = MPOAgent(env_id=env_id, logdir=LOGDIR)

    for n in range(n_episodes):

        rewards, steps = agent.rollout()

        print(f"Episode {n}: {rewards}, {steps} steps")

        if n % 100 == 0:
            agent.testplay(name=f"ep_{n}", monitor_dir=MONITOR_DIR)
            agent.save("checkpoints/")

    agent.save("checkpoints/")
    print("Training finshed")


def test(env_id="BipedalWalker-v3", n_testplay=5):
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"   # Needed only for ubuntu

    MONITOR_DIR = Path(__file__).parent / "mp4"

    agent = MPOAgent(env_id=env_id, logdir=None)
    agent.load("checkpoints/")
    for i in range(1, n_testplay+1):
        agent.testplay(name=f"test_{i}", monitor_dir=MONITOR_DIR)


if __name__ == '__main__':
    """
    tensorflow 2.4.0
    tensorflow-probability 0.11.1
    gym[box2d] 0.24.1
    """
    train()
    test()
