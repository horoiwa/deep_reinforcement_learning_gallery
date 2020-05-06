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

    def run(self, coord):

        self.sync_with_globalparameters()

        self.total_reward = 0

        self.state = self.env.reset()

        try:
            while not coord.should_stop():

                trajectory = self.play_n_steps(N=self.MAX_TRAJECTORY)

                self.update_globalnets(trajectory)

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

            if done:
                step = Step(self.state, action, -10, next_state, done)
            else:
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

                self.sync_with_globalnetworks()

                break

            else:
                self.total_reward += reward
                self.state = next_state

        return trajectory

    def update_globalnets(self, trajectory):

        if trajectory[-1].done:
            R = 0
        else:
            R = self.value_network.predict(trajectory[-1].next_state)

        discounted_rewards = []
        advantages = []

        for step in reversed(trajectory):
            R = step.reward + self.gamma * R
            adv = R - self.value_network.predict(step.state)
            discounted_rewards.append(R)
            advantages.append(adv)

        discounted_rewards.reverse()
        discounted_rewards = np.array(discounted_rewards)

        advantages.reverse()
        advantages = np.array(advantages)

        states = np.array([step.state for step in trajectory])
        actions = np.array([step.action for step in trajectory])

        policy_grads = self.policy_network.compute_grads(
            states, actions, advantages)

        value_grads = self.value_network.compute_grads(
            states, discounted_rewards)

        global_policy_variables = self.global_policy_network.trainable_variables
        global_value_variables = self.global_value_network.trainable_variables

        self.global_policy_network.optimizer.apply_gradients(
            zip(policy_grads, global_policy_variables))

        self.global_value_network.optimizer.apply_gradients(
            zip(value_grads, global_value_variables))

    def sync_with_globalparameters(self):

        global_variables = self.global_ACNet.trainable_variables

        local_variables = self.local_ACNet.trainable_variables

        for v1, v2 in zip(local_variables, global_variables):
            v1.assign(v2.numpy())


def main():

    ACTION_SPACE = 2

    NUM_AGENTS = 4

    N_STEPS = 10000

    MONITOR_DIR = Path(__file__).parent / "history"
    if not MONITOR_DIR.exists():
        MONITOR_DIR.mkdir()

    with tf.device("/cpu:0"):

        global_counter = GlobalCounter()

        global_history = []

        global_ACNet = ActorCriticNet(ACTION_SPACE)

        agents = []

        for agent_id in range(NUM_AGENTS):

            agent = A3CAgent(agent_id=f"agent_{agent_id}",
                             env=gym.envs.make("CartPole-v1"),
                             global_counter=global_counter,
                             action_space=ACTION_SPACE,
                             global_ACNet=ActorCriticNet,
                             gamma=0.99,
                             global_history=global_history,
                             global_steps_fin=N_STEPS)

            agents.append(agent)

    coord = tf.train.Coordinator()
    agent_threads = []
    for agent in agents:
        target_func = (lambda: agent.run(coord))
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
