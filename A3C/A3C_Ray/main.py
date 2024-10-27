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

    def __init__(self, agent_id, env_name,
                 gamma=0.98, entropy_coef=0.01, batch_size=8):

        self.agent_id = agent_id

        self.gamma = gamma

        self.ent_coef = entropy_coef

        self.batch_size = batch_size

        self.env = gym.make(env_name)

        self.action_space = self.env.action_space.n

        self.state = self.env.reset()

        self.policy = PolicyWithValue(action_space=self.action_space)

        #: initialize weights
        self.policy.call(np.atleast_2d(self.state).astype(np.float32))

    def rollout_and_compute_grads(self, weights):
        """
            0. グローバルネットワークの重みと同期
            1. ミニバッチ数のサンプルを収集(rollout)
            2. ミニバッチからロスを算出して勾配を計算
        """
        self.policy.set_weights(weights)

        trajectory = self._rollout()

        with tf.GradientTape() as tape:

            values, action_probs = self.policy(trajectory["s"])

            #: tf.one_hotはint型しか受け付けないことに注意
            selected_actions = tf.convert_to_tensor(
                trajectory["a"].flatten(), dtype=tf.int32)
            selected_actions_onehot = tf.one_hot(
                selected_actions, self.action_space, dtype=tf.float32)

            log_probs = selected_actions_onehot * tf.math.log(action_probs + 1e-5)

            selected_action_log_probs = tf.reduce_sum(log_probs, axis=1, keepdims=True)

            #: value_loss = 打ち切り累積報酬和 - V(s1)
            advantages = (trajectory["R"] - values)
            value_loss = tf.reduce_mean(advantages ** 2)
            mean_advantage = tf.reduce_mean(advantages)

            #: policy_loss = logπ(a | s) * Advantage
            policy_loss = selected_action_log_probs * tf.stop_gradient(advantages)
            policy_loss = tf.reduce_mean(policy_loss)

            #: entropy = Σ-π(a|s)logπ(a|s)
            entropy = -1 * tf.reduce_sum(
                action_probs * tf.math.log(action_probs + 1e-5),
                axis=1, keepdims=True)
            entropy = tf.reduce_mean(entropy)

            #: policylossとエントロピーは最大化する
            loss = -1 * policy_loss + 0.5 * value_loss + -1 * self.ent_coef * entropy

        grads = tape.gradient(
            loss, self.policy.trainable_variables)

        info = {"id": self.agent_id,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy, "advantage": mean_advantage}

        return (grads, info)

    def _rollout(self):

        trajectory = {}

        trajectory["s"] = np.zeros(
            (self.batch_size, self.env.observation_space.shape[0])
            ).astype(np.float32)

        trajectory["a"] = np.zeros(
            (self.batch_size, 1)).astype(np.float32)

        trajectory["r"] = np.zeros(
            (self.batch_size, 1)).astype(np.float32)

        trajectory["s2"] = np.zeros(
            (self.batch_size, self.env.observation_space.shape[0])
            ).astype(np.float32)

        trajectory["dones"] = np.zeros(
            (self.batch_size, 1)).astype(np.float32)

        for i in range(self.batch_size):

            action = self.policy.sample_action(self.state)

            next_state, reward, done, _ = self.env.step(action)

            trajectory["s"][i] = self.state
            trajectory["a"][i] = action
            trajectory["r"][i] = reward
            trajectory["s2"][i] = next_state
            trajectory["dones"][i] = done

            if done:
                self.state = self.env.reset()
            else:
                self.state = next_state

        #: multistep-advantageの計算

        trajectory["R"] = np.zeros(
            (self.batch_size, 1)).astype(np.float32)

        values, _ = self.policy(trajectory["s2"])
        R = values[-1]
        for i in reversed(range(self.batch_size)):
            R = trajectory["r"][i] + self.gamma * (1 - trajectory["dones"][i]) * R
            trajectory["R"][i] = R

        return trajectory


def learn(num_agents=5, env_name="CartPole-v1", num_updates=50000, lr=1e-4):

    print("ray version:", ray.__version__)
    ray.init(local_mode=False)

    env = gym.make(env_name)
    action_space = env.action_space.n

    #: tensorflow2では一度callしないとパラメータが初期化されない
    global_policy = PolicyWithValue(action_space=action_space)
    global_policy(np.atleast_2d(env.reset()))

    optimizer = tf.keras.optimizers.Adam(lr=lr)

    agents = [Agent.remote(agent_id=i, env_name=env_name)
              for i in range(num_agents)]

    logdir = Path(__file__).parent / "log"
    if logdir.exists():
        shutil.rmtree(logdir)
    summary_writer = tf.summary.create_file_writer(str(logdir))

    weights = global_policy.get_weights()
    work_in_progresses = [agent.rollout_and_compute_grads.remote(weights)
                          for agent in agents]

    for n in tqdm(range(num_updates)):

        #: 完了jobをひとつ取り出し
        finished_job, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
        grads, info = ray.get(finished_job)[0]
        agent_id = info["id"]

        #: 勾配適用
        optimizer.apply_gradients(
            zip(grads, global_policy.trainable_variables))

        #: jobを追加
        weights = global_policy.get_weights()
        work_in_progresses.extend(
            [agents[agent_id].rollout_and_compute_grads.remote(weights)]
            )

        with summary_writer.as_default():
            tf.summary.scalar("policy_loss", info["policy_loss"], step=n)
            tf.summary.scalar("value_loss", info["value_loss"], step=n)
            tf.summary.scalar("entropy", info["entropy"], step=n)
            tf.summary.scalar("advantage", info["advantage"], step=n)

        if n % 100 == 0:
            episode_rewards = test_play(global_policy, env_name)
            with summary_writer.as_default():
                tf.summary.scalar("test_reward", episode_rewards, step=n)
    else:
        ray.shutdown()

    print("Start TestPlay")
    monitordir = Path(__file__).parent / "mp4"
    if monitordir.exists():
        shutil.rmtree(monitordir)

    for i in range(3):
        total_rewards = test_play(global_policy, env_name, monitordir)
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

        action = policy.sample_action(state)

        #: Debug only
        #_, action_prob = policy(np.atleast_2d(state))
        #print(action_prob.numpy(), action)

        next_state, reward, done, _ = env.step(action)

        total_rewards += reward

        if done:
            break
        else:
            state = next_state

    return total_rewards


if __name__ == '__main__':
    learn()
