import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dataclasses import dataclass
import threading

import tensorflow as tf
import gym

from models import create_networks


@dataclass
class Step:

    state: list

    action: int

    reward: float

    next_state: list

    done: bool


@dataclass
class GlobalCounter:

    n: int = 0


class A3CAgent:

    MAX_TRAJECTORY = 5

    def __init__(self, agent_id, env,
                 global_counter, action_space,
                 global_value_network, global_policy_network,
                 gamma, total_rewards, global_steps_fin):

        self.agent_id = agent_id

        self.env = env

        self.global_counter = global_counter

        self.action_space = action_space

        self.global_value_network = global_value_network

        self.global_policy_network = global_policy_network

        self.value_network, self.policy_network = create_networks(action_space)

        self.gamma = gamma

        self.total_rewards = total_rewards

        self.global_steps_fin = global_steps_fin

    def run(self, coord):

        self.state = self.env.reset()

        try:
            while not coord.should_stop():

                self.copy_globalnets()

                trajectory = self.play_n_steps()

                self.update_globalnets(trajectory)

                if self.global_counter.n >= self.global_steps_fin:
                    coord.request_stop()

        except tf.errors.CancelledError:
            return

    def play_n_steps(self):
        trajectory = []
        for _ in range(self.MAX_TRAJECTORY):

            self.global_counter.n += 1

        return trajectory

    def update_globalnets(self, trajectory):
        pass

    def copy_globalnets(self):
        global_valuenet_vars = self.global_value_network.trainable_variables
        global_policynet_vars = self.global_policy_network.trainable_variables

        local_valuenet_vars = self.value_network.trainable_variables
        local_policynet_vars = self.policy_network.trainable_variables

        for v1, v2 in zip(local_valuenet_vars, global_valuenet_vars):
            v1.assign(v2.numpy())

        for v1, v2 in zip(local_policynet_vars, global_policynet_vars):
            v1.assign(v2.numpy())


def main():

    ACTION_SPACE = 2

    NUM_AGENTS = 1

    N_STEPS = 100

    with tf.device("/cpu:0"):

        global_counter = GlobalCounter()

        total_rewards = []

        global_value_network, global_policy_network = create_networks(ACTION_SPACE)

        agents = []
        for agent_id in range(NUM_AGENTS):

            agent = A3CAgent(agent_id=f"agent_{agent_id}",
                             env=gym.envs.make("CartPole-v1"),
                             global_counter=global_counter,
                             action_space=ACTION_SPACE,
                             global_value_network=global_value_network,
                             global_policy_network=global_policy_network,
                             gamma=0.99, total_rewards=total_rewards,
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

    print(total_rewards)


if __name__ == "__main__":
    main()
