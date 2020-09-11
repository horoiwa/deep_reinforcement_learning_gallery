import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pathlib import Path

import gym
from gym import wrappers
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from env import VecEnv
from models import PolicyNetwork, ValueNetwork


def env_func():
    return gym.make("Pendulum-v0")


class PPOAgent:

    TRAJECTORY_SIZE = 64

    V_BATCH_SIZE = 64

    OBS_SPACE = 3

    GAMMA = 0.99

    GAE_LAMBDA = 0.95

    CLIPRANGE = 0.2

    def __init__(self, env_id, action_space, n_envs):

        self.env_id = env_id

        self.n_envs = n_envs

        self.vecenv = VecEnv(env_id=self.env_id, n_envs=self.n_envs)

        self.policy = PolicyNetwork(action_space=action_space)

        self.value = ValueNetwork()

    def run(self, n_epochs, trajectory_size):

        history = {"steps": [], "scores": []}

        states = self.vecenv.reset()

        hiscore = 0

        for epoch in range(n_epochs):

            for _ in range(trajectory_size):

                actions = self.policy.sample_action(states)

                next_states = self.vecenv.step(actions)

                states = next_states

            trajectories = self.vecenv.get_trajectories()

            trajectories = self.compute_advantage(trajectories)

            states, actions, advantages, vtargs = self.create_minibatch(trajectories)

            self.update_policy(states, actions, advantages)

            self.update_value(states, vtargs)

            global_steps = (epoch+1) * trajectory_size * self.n_envs
            test_scores = np.array(self.play(n=3))
            history["steos"].append(global_steps)
            history["scores"].append(test_scores.mean())

            ma_score = sum(history["scores"][-10:]) / 10
            if epoch > 10 and ma_score > hiscore:
                self.save_model()
                print("Model Saved")

            print(f"Epoch {epoch}, {global_steps//1000}K, {test_score}")

        return history

    def compute_advantage(self, trajectories):
        """
            compute Generalized Advantage Estimation (GAE, 2016)
        """
        for trajectory in trajectories:

            trajectory["v_pred"] = self.value(trajectory["s"]).numpy()

            trajectory["v_pred_next"] = self.value(trajectory["s2"]).numpy()

            is_nonterminals = 1 - trajectory["done"]

            deltas = trajectory["r"] + self.GAMMA * is_nonterminals * trajectory["v_pred_next"] - trajectory["v_pred"]

            advantages = np.zeros_like(deltas, dtype=np.float32)

            lastgae = 0
            for i in reversed(range(len(deltas))):
                lastgae = deltas[i] + self.GAMMA * self.GAE_LAMBDA * is_nonterminals[i] * lastgae
                advantages[i] = lastgae

            trajectory["v_target"] = advantages + trajectory["v_pred"]

            trajectory["advantage"] = advantages

        return trajectories

    def update_policy(self, states, actions, advantages):

        old_means, old_stdevs = self.policy(states)

        old_logprob = self.compute_logprob(old_means, old_stdevs, actions)

        with tf.GradientTape() as tape:

            new_means, new_stdevs = self.policy(states)

            new_logprob = self.compute_logprob(new_means, new_stdevs, actions)

            ratio = tf.exp(new_logprob - old_logprob)
            ratio_clipped = tf.clip_by_value(ratio,
                                             1 - self.CLIPRANGE,
                                             1 + self.CLIPRANGE)
            loss_unclipped = ratio * advantages
            loss_clipped = ratio_clipped * advantages
            loss = tf.minimum(loss_unclipped, loss_clipped)

            #: 最大化したいので-1倍する
            loss *= -1

        variables = self.policy.trainable_variables
        grads = tape.gradient(loss, variables)
        self.policy.optimizer.apply_gradients(zip(grads, variables))

    def update_value(self, states, v_targs):
        return None

    @tf.function
    def compute_logprob(self, means, stdevs, actions):
        """ガウス分布の確率密度関数よりlogp(x)を計算
            logp(x) = -0.5 log(2π) - log(std)  -0.5 * ((x - mean) / std )^2
        """
        logprob = - 0.5 * np.log(2*np.pi)
        logprob += - tf.math.log(stdevs)
        logprob += - 0.5 * tf.square((actions - means) / stdevs)
        logprob = tf.reduce_sum(logprob, axis=1, keepdims=True)
        return logprob

    def create_minibatch(self, trajectories):

        states = np.vstack([traj["s"] for traj in trajectories])
        actions = np.vstack([traj["a"] for traj in trajectories])

        advantages = np.vstack([traj["advantage"] for traj in trajectories])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        v_targs = np.vstack([traj["v_target"] for traj in trajectories])

        return states, actions, advantages, v_targs

    def save_model(self):

        self.policy.save_weights("checkpoints/policy")

        self.value.save_weights("checkpoints/value")

    def load_model(self):

        self.policy.load_weights("checkpoints/policy")

        self.value.load_weights("checkpoints/value")

    def play(self, n=1, monitordir=None):

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

                action = self.policy.sample_action(state)

                next_state, reward, done, _ = env.step(action[0])

                total_reward += reward

                if done:
                    break
                else:
                    state = next_state

            total_rewards.append(total_reward)

        return total_rewards


def main():

    MONITOR_DIR = Path(__file__).parent / "log"

    agent = PPOAgent(env_id="Pendulum-v0", action_space=1, n_envs=4)

    history = agent.run(n_epochs=10, trajectory_size=256)

    plt.plot(history["steps"], history["scores"])
    plt.xlabel("steps")
    plt.ylabel("Total rewards")
    plt.savefig(MONITOR_DIR / "testplay.png")


def testplay():

    MONITOR_DIR = Path(__file__).parent / "log"

    agent = PPOAgent(env_id="Pendulum-v0", action_space=1, n_envs=4)
    #agent.load_model()
    agent.play(n=1, monitordir=MONITOR_DIR)


if __name__ == "__main__":
    #main()
    testplay()
