import collections
from dataclasses import dataclass
import pickle
import random

import ray
import gym
import numpy as np
import tensorflow as tf
import lz4.frame as lz4f

import util
from mcts import AtariMCTS
from networks import DynamicsNetwork, PVNetwork, RepresentationNetwork


@dataclass
class Sample:

    observation: tf.Tensor
    actions: list
    target_policies: list
    target_rewards: list
    nstep_returns: list
    last_observations: list
    dones: list


#@ray.remote(num_cpus=1, num_gpus=0)
class Actor:

    def __init__(self, pid, env_id, n_frames,
                 num_mcts_simulations, unroll_steps,
                 gamma, V_max, V_min, td_steps,
                 dirichlet_alpha, initial_randomize=True):

        self.pid = pid

        self.env_id = env_id

        self.unroll_steps = unroll_steps

        self.num_mcts_simulations = num_mcts_simulations

        self.action_space = gym.make(env_id).action_space.n

        self.n_frames = n_frames

        self.gamma = gamma

        self.dirichlet_alpha = dirichlet_alpha

        self.V_min, self.V_max = V_min, V_max

        self.td_steps = td_steps

        self.preprocess_func = util.get_preprocess_func(self.env_id)

        self.repr_network = RepresentationNetwork(
            action_space=self.action_space)

        self.pv_network = PVNetwork(action_space=self.action_space,
                                    V_min=V_min, V_max=V_max)

        self.dynamics_network = DynamicsNetwork(action_space=self.action_space,
                                                V_min=V_min, V_max=V_max)

        self.initial_randomize = initial_randomize

        self.setup()

        self.reset_env()

    def setup(self):
        """ initialize weights """

        env = gym.make(self.env_id)
        frame = self.preprocess_func(env.reset())

        frame_history = [frame] * self.n_frames
        action_history = [0] * self.n_frames

        state, obs = self.repr_network.predict(frame_history, action_history)
        policy, value = self.pv_network(state)
        next_state, reward = self.dynamics_network.predict(state, action=0)

    def reset_env(self):

        self.env = gym.make(self.env_id)

        self.lives = self.env.ale.lives()

        frame = self.preprocess_func(self.env.reset())

        self.frame_history = collections.deque(
            [frame] * self.n_frames, maxlen=self.n_frames)

        self.action_history = collections.deque(
            [0] * self.n_frames, maxlen=self.n_frames)

        self.mcts = AtariMCTS(
            action_space=self.action_space,
            pv_network=self.pv_network,
            dynamics_network=self.dynamics_network,
            gamma=self.gamma,
            dirichlet_alpha=self.dirichlet_alpha)

        self.episode_rewards = 0

        self.episode_steps = 0

        if self.initial_randomize:
            """
            Randomize first few frames to stable early learning
            This code is tuned for and only valid for BreakoutDeterminitic-v4
            """

            for i in range(random.choice(range(10))):

                if i == 0:
                    action = 1  # FIRE
                    frame, reward, done, info = self.env.step(action)
                else:
                    action = random.choice(range(self.action_space))
                    frame, reward, done, info = self.env.step(action)

                self.frame_history.append(self.preprocess_func(frame))
                self.action_history.append(action)

                self.episode_rewards += reward
                self.episode_steps += 1

    def sync_weights_and_rollout(self, current_weights, T):

        #: 最新のネットワークに同期
        if current_weights:
            self.sync_weights(current_weights)

        with util.Timer("Actor:"):

            if self.lives == 0:
                self.reset_env()

            #: 1episodeのrollout
            game_history = self.rollout(T)

        samples, priorities = self.make_samples(game_history)

        return self.pid, samples, priorities

    def sync_weights(self, weights):

        self.repr_network.set_weights(weights[0])

        self.pv_network.set_weights(weights[1])

        self.dynamics_network.set_weights(weights[2])

    def rollout(self, T):

        observations, actions, rewards, mcts_policies, root_values = [], [], [], [], []

        done = False

        while not done:

            hidden_state, obs = self.repr_network.predict(
                self.frame_history, self.action_history)

            mcts_policy, action, root_value = self.mcts.search(
                hidden_state, self.num_mcts_simulations, T)

            frame, reward, done, info = self.env.step(action)

            #: Life loss as episode end
            if info["ale.lives"] != self.lives:
                done = True
                self.lives = info["ale.lives"]

            self.frame_history.append(self.preprocess_func(frame))

            self.action_history.append(action)

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            mcts_policies.append(mcts_policy)
            root_values.append(root_value)

            self.episode_rewards += reward

            self.episode_steps += 1

            #: Handle freeze agent
            if ((self.episode_steps > 500 and self.episode_rewards < 3) or
                (self.episode_steps > 1500 and self.episode_rewards < 10)):
                self.lives = 0
                break

        game_history = (observations, actions, rewards,
                        mcts_policies, root_values, done)

        return game_history

    def make_samples(self, game_history):
        """
            zは割引が必要であることに注意
            またbootstrapping return
            absorbing states.
        """

        samples = []

        observations, actions, rewards, mcts_policies, root_values = game_history

        episode_len = len(observations)

        #: States past the end of games are treated as absorbing states.
        rewards += [0] * self.td_steps

        #: NOOP
        actions += [0] * self.td_steps

        #: Uniform policy
        mcts_policies += [
            np.array([1. / self.action_space] * self.action_space, dtype=np.float32)
            for _ in range(self.td_steps)]

        #: n-step bootstrapping value
        nstep_returns = []

        priorities = []

        for idx in range(episode_len):

            bootstrap_idx = idx + self.td_steps

            nstep_return = sum([r * self.gamma ** i for i, r
                                in enumerate(rewards[idx:bootstrap_idx])])

            residual_value = root_values[bootstrap_idx] if bootstrap_idx < episode_len else 0

            #: value = r_0 + γr_1 + ... r_n + v(n+1)
            value = nstep_return + residual_value

            priorities.append(abs(value - root_values[idx]))

            nstep_returns.append(nstep_return)

        nstep_returns += [0] * self.td_steps

        for idx in range(episode_len):

            bootstrap_idx = idx + self.td_steps

            #: shape == (unroll_steps, ...)
            _actions = np.array(actions[idx:idx+self.unroll_steps])
            _nstep_returns = np.array(nstep_returns[idx:idx+self.unroll_steps], dtype=np.float32)
            target_policies = np.vstack(mcts_policies[idx:idx+self.unroll_steps])
            target_rewards = np.array(rewards[idx:idx+self.unroll_steps], dtype=np.float32)
            last_observations = [observations[i] if i < episode_len else observations[-1]
                                 for i in range(bootstrap_idx, bootstrap_idx+self.unroll_steps)]
            dones = [False if i < episode_len else True
                     for i in range(bootstrap_idx, bootstrap_idx+self.unroll_steps)]

            sample = Sample(
                observation=observations[idx],
                actions=_actions,
                target_policies=target_policies,
                target_rewards=target_rewards,
                nstep_returns=_nstep_returns,
                last_observations=last_observations,
                dones=dones)

            samples.append(sample)

        #: Compress for memory efficiency
        samples = [lz4f.compress(pickle.dumps(sample)) for sample in samples]

        return samples, priorities

    def testplay(self, current_weights=None, T=0.25):

        #: 最新のネットワークに同期
        if current_weights:
            self.sync_weights(current_weights)

        env = gym.make(self.env_id)

        frame = self.preprocess_func(env.reset())

        frame_history = collections.deque(
            [frame] * self.n_frames, maxlen=self.n_frames)

        action_history = collections.deque(
            [0] * self.n_frames, maxlen=self.n_frames)

        mcts = AtariMCTS(
            action_space=self.action_space,
            pv_network=self.pv_network,
            dynamics_network=self.dynamics_network,
            gamma=self.gamma,
            dirichlet_alpha=self.dirichlet_alpha)

        episode_rewards, episode_steps = 0, 0

        done = False

        while not done:

            hidden_state, obs = self.repr_network.predict(frame_history, action_history)

            mcts_policy, action, root_value = mcts.search(
                hidden_state, self.num_mcts_simulations, T=T)

            frame, reward, done, info = env.step(action)

            episode_rewards += reward

            episode_steps += 1

            frame_history.append(self.preprocess_func(frame))

            action_history.append(action)

            #: handle freeze
            if episode_steps > 500 and episode_rewards < 3:
                break
            elif episode_steps > 1500 and episode_rewards < 10:
                break

        return episode_rewards, episode_steps


if __name__ == "__main__":

    actor = Actor(pid=0, env_id="BreakoutDeterministic-v4", n_frames=4,
                  unroll_steps=3, td_steps=3,
                  num_mcts_simulations=10,
                  V_min=-30, V_max=30, gamma=0.997,
                  dirichlet_alpha=0.25)

    rewards, steps = actor.testplay(current_weights=None)
    print(rewards, steps)

    for _ in range(6):
        actor.sync_weights_and_rollout(current_weights=None, T=1.0)
