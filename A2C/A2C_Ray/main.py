from pathlib import Path
import shutil

import ray
import gym
from gym import wrappers
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
from tqdm import tqdm

from model import PolicyWithValue


@ray.remote(num_cpus=1)
class Agent:

    def __init__(self, agent_id, env_name):

        self.agent_id = agent_id

        self.env = gym.make(env_name)

        self.state = None

        self.trajectory = {"s": [], "a": [], "r": [],
                           "s2": [], "dones": []}

    def reset_env(self):
        self.state = self.env.reset()
        return self.state

    def step(self, action):

        state = self.state

        next_state, reward, done, _ = self.env.step(action)

        self.trajectory["s"].append(state)
        self.trajectory["a"].append(action)
        self.trajectory["r"].append(reward)
        self.trajectory["s2"].append(next_state)
        self.trajectory["dones"].append(done)

        if done:
            self.state = self.env.reset()
        else:
            self.state = next_state

        return self.state

    def collect_trajectory(self):
        """蓄積したtrajectoryの回収
        """
        trajectory = self.trajectory
        self.trajectory = {"s": [], "a": [], "r": [],
                           "s2": [], "dones": []}

        return trajectory


def learn(num_agents=5, env_name="CartPole-v1", gamma=0.98, entropy_coef=0.01,
          trajectory_length=8, num_updates=10000, lr=1e-4):

    ray.init(local_mode=False)

    env = gym.make(env_name)
    action_space = env.action_space.n

    policy = PolicyWithValue(action_space=action_space)
    optimizer = tf.keras.optimizers.Adam(lr=lr)

    agents = [Agent.remote(agent_id=i, env_name=env_name)
              for i in range(num_agents)]

    logdir = Path(__file__).parent / "log"
    if logdir.exists():
        shutil.rmtree(logdir)
    summary_writer = tf.summary.create_file_writer(str(logdir))

    states = ray.get([agent.reset_env.remote() for agent in agents])
    states = np.array(states)

    for n in tqdm(range(num_updates)):

        for _ in range(trajectory_length):
            #: 各プロセスごとにNstepのrolloutを実行
            actions = policy.sample_actions(states)
            states = ray.get(
                [agent.step.remote(action) for action, agent in zip(actions, agents)])

        #: 蓄積されたtrjectoryを回収
        trajectories = ray.get(
            [agent.collect_trajectory.remote() for agent in agents])

        #: mixed n-step return の計算
        for trajectory in trajectories:
            trajectory["R"] = [0] * trajectory_length
            value, _ = policy(np.atleast_2d(trajectory["s2"][-1]))
            R = value[0][0].numpy()
            for i in reversed(range(trajectory_length)):
                R = trajectory["r"][i] + gamma * (1 - trajectory["dones"][i]) * R
                trajectory["R"][i] = R

        #: trajectoriesをまとめる
        (states, actions, next_states, rewards,
         dones, discounted_returns) = [], [], [], [], [], []

        for trajectory in trajectories:
            states += trajectory["s"]
            actions += trajectory["a"]
            next_states += trajectory["s2"]
            rewards += trajectory["r"]
            dones += trajectory["dones"]
            discounted_returns += trajectory["R"]

        states = np.array(states, dtype=np.float32)
        selected_actions = np.array(actions, dtype=np.int32)
        next_states = np.array(next_states, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.int8)
        discounted_returns = np.array(discounted_returns, dtype=np.float32).reshape(-1, 1)

        with tf.GradientTape() as tape:

            values, action_probs = policy(states)

            #: tf.one_hotはint型しか受け付けないことに注意
            selected_actions_onehot = tf.one_hot(
                selected_actions, action_space, dtype=tf.float32)

            log_probs = selected_actions_onehot * tf.math.log(action_probs + 1e-5)

            selected_action_log_probs = tf.reduce_sum(log_probs, axis=1, keepdims=True)

            #: value_loss = 打ち切り累積報酬和 - V(s1)
            advantages = discounted_returns - values
            value_loss = tf.reduce_mean(advantages ** 2)

            #: policy_loss = logπ(a | s) * Advantage
            policy_loss = selected_action_log_probs * tf.stop_gradient(advantages)
            policy_loss = tf.reduce_mean(policy_loss)

            #: entropy = Σ-π(a|s)logπ(a|s)
            entropy = -1 * tf.reduce_sum(
                action_probs * tf.math.log(action_probs + 1e-5),
                axis=1, keepdims=True)
            entropy = tf.reduce_mean(entropy)

            #: policylossとエントロピーは最大化する
            loss = -1 * policy_loss + 0.5 * value_loss + -1 * entropy_coef * entropy

        grads = tape.gradient(loss, policy.trainable_variables)
        optimizer.apply_gradients(
            zip(grads, policy.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar("loss", policy_loss, step=n)
            tf.summary.scalar("policy_loss", policy_loss, step=n)
            tf.summary.scalar("value_loss", value_loss, step=n)
            tf.summary.scalar("entropy", entropy, step=n)

        if n % 50 == 0:
            episode_rewards = test_play(policy, env_name)
            with summary_writer.as_default():
                tf.summary.scalar("test_reward", episode_rewards, step=n)

    ray.shutdown()

    print("TestPlay")
    monitordir = Path(__file__).parent / "mp4"
    if monitordir.exists():
        shutil.rmtree(monitordir)

    for i in range(3):
        total_rewards = test_play(policy, env_name, monitordir)
        print(f"Test {i}:", total_rewards)


def test_play(policy, env_name, monitordir=None):
    if monitordir:
        env = wrappers.Monitor(gym.make(env_name),
                               monitordir, force=True,
                               video_callable=(lambda ep: True))
    else:
        env = gym.make(env_name)

    state = env.reset()

    total_rewards = 0

    while True:

        action = policy.sample_actions(state)[0]

        next_state, reward, done, _ = env.step(action)

        total_rewards += reward

        if done:
            break
        else:
            state = next_state

    return total_rewards


if __name__ == '__main__':
    learn()
