import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dataclasses import dataclass
from pathlib import Path
import threading

import numpy as np
import pandas as pd
import tensorflow as tf
import gym
import matplotlib.pyplot as plt

from models import ActorCriticNet


@dataclass
class Step:

    state: np.ndarray

    action: int

    reward: float

    next_state: np.ndarray

    done: bool


@dataclass
class GlobalCounter:

    n: int = 0


class A3CAgent:

    MAX_TRAJECTORY = 3

    def __init__(self, agent_id, env,
                 global_counter, action_space,
                 global_ACNet,
                 gamma, global_history, global_steps_fin):

        self.agent_id = agent_id

        self.env = env

        self.global_counter = global_counter

        self.action_space = action_space

        self.global_ACNet = global_ACNet

        self.local_ACNet = ActorCriticNet(self.action_space)

        self.gamma = gamma

        self.global_history = global_history

        self.global_steps_fin = global_steps_fin

        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)

    def play(self, coord):

        self.total_reward = 0

        self.state = self.env.reset()

        try:
            while not coord.should_stop():

                trajectory = self.play_n_steps(N=self.MAX_TRAJECTORY)

                with tf.GradientTape() as tape:

                    total_loss = self.compute_loss(trajectory)

                grads = tape.gradient(
                    total_loss, self.local_ACNet.trainable_variables)

                self.optimizer.apply_gradients(
                    zip(grads, self.global_ACNet.trainable_variables))

                self.local_ACNet.set_weights(self.global_ACNet.get_weights())

                if self.global_counter.n >= self.global_steps_fin:
                    coord.request_stop()

        except tf.errors.CancelledError:
            return

    def play_n_steps(self, N):

        trajectory = []

        for _ in range(N):

            self.global_counter.n += 1

            action = self.local_ACNet.sample_action(self.state)

            next_state, reward, done, info = self.env.step(action)

            step = Step(self.state, action, reward, next_state, done)

            trajectory.append(step)

            if done:
                print(f"Global step {self.global_counter.n}")
                print(f"Total Reward: {self.total_reward}")
                print(f"Agent: {self.agent_id}")
                print()

                self.global_history.append(self.total_reward)

                self.total_reward = 0

                self.state = self.env.reset()

                break

            else:
                self.total_reward += reward
                self.state = next_state

        return trajectory

    def compute_loss(self, trajectory):

        if trajectory[-1].done:
            R = 0
        else:
            R, _ = self.local_ACNet(
                tf.convert_to_tensor(np.atleast_2d(trajectory[-1].next_state),
                                     dtype=tf.float32))

        discounted_rewards = []
        for step in reversed(trajectory):
            R = step.reward + self.gamma * R
            discounted_rewards.append(R)
        discounted_rewards.reverse()

        states = tf.convert_to_tensor(
            np.vstack([step.state for step in trajectory]), dtype=tf.float32)

        values, logits = self.local_ACNet(states)

        advantages = tf.convert_to_tensor(
            np.array(discounted_rewards)[:, None], dtype=tf.float32) - values

        value_loss = advantages ** 2

        actions_onehot = tf.one_hot([step.action for step in trajectory],
                                    self.action_space, dtype=tf.float32)

        action_probs = tf.nn.softmax(logits)

        entropy = tf.reduce_sum(
            action_probs * tf.math.log(action_probs + 1e-20), axis=1)

        policy_loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
            labels=actions_onehot, logits=logits)

        policy_loss *= tf.stop_gradient(advantages)

        policy_loss -= 0.01 * entropy

        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))

        return total_loss


def main():

    ACTION_SPACE = 2

    NUM_AGENTS = 4

    N_STEPS = 3000

    MONITOR_DIR = Path(__file__).parent / "history"
    if not MONITOR_DIR.exists():
        MONITOR_DIR.mkdir()

    with tf.device("/cpu:0"):

        global_counter = GlobalCounter()

        global_history = []

        global_ACNet = ActorCriticNet(ACTION_SPACE)

        global_ACNet.build(input_shape=(None, 4))

        agents = []

        for agent_id in range(NUM_AGENTS):

            agent = A3CAgent(agent_id=f"agent_{agent_id}",
                             env=gym.envs.make("CartPole-v1"),
                             global_counter=global_counter,
                             action_space=ACTION_SPACE,
                             global_ACNet=global_ACNet,
                             gamma=0.99,
                             global_history=global_history,
                             global_steps_fin=N_STEPS)

            agents.append(agent)

    coord = tf.train.Coordinator()
    agent_threads = []
    for agent in agents:
        target_func = (lambda: agent.play(coord))
        thread = threading.Thread(target=target_func)
        thread.start()
        agent_threads.append(thread)

    coord.join(agent_threads, stop_grace_period_secs=300)

    print(global_history)

    plt.plot(range(len(global_history)), global_history)
    plt.plot([0, 350], [195, 195], "--", color="darkred")
    plt.xlabel("episodes")
    plt.ylabel("Total Reward")
    plt.savefig(MONITOR_DIR / "a3c_cartpole-v1.png")

    df = pd.DataFrame()
    df["Total Reward"] = global_history
    df.to_csv(MONITOR_DIR / "a3c_cartpole-v1.csv", index=None)


if __name__ == "__main__":
    main()
