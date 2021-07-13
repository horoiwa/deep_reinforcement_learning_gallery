import collections
from dataclasses import dataclass
import pickle

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


class Actor:

    def __init__(self, env_id, n_frames,
                 num_mcts_simulations, unroll_steps,
                 gamma, V_max, V_min, td_steps,
                 dirichlet_alpha):

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

        self._build_network()

    def _build_network(self):

        env = gym.make(self.env_id)
        frame = self.preprocess_func(env.reset())

        frame_history = [frame] * self.n_frames
        action_history = [0] * self.n_frames

        state, obs = self.repr_network.predict(frame_history, action_history)
        policy, value = self.pv_network(state)
        next_state, reward = self.dynamics_network.predict(state, action=0)

    def sync_weights_and_rollout(self, current_weights, T):

        #: 最新のネットワークに同期
        self.sync_weights(current_weights)

        #: 1episodeのrollout
        game_history = self.rollout(T)

        samples, priorities = self.make_samples(game_history)

        return samples, priorities

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
            target_rewards = np.array(rewards[idx:idx+self.unroll_steps])
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

    def sync_weights(self, weights):

        self.repr_network.set_weights(weights[0])

        self.pv_network.set_weights(weights[1])

        self.dynamics_network.set_weights(weights[2])

    def rollout(self, T):

        env = gym.make(self.env_id)

        observations, actions, rewards, mcts_policies, root_values = [], [], [], [], []

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

        done = False

        while not done:

            with tf.device("/cpu:0"):
                hidden_state, obs = self.repr_network.predict(frame_history, action_history)

            mcts_policy, root_value = mcts.search(
                hidden_state, self.num_mcts_simulations, T)

            action = np.random.choice(
                range(self.action_space), p=mcts_policy)

            #mcts_policy, root_value = np.array([0.1, 0.1, 0.3, 0.5], dtype=np.float32), 1.0
            #action = np.random.choice(range(self.action_space), p=mcts_policy)

            frame, reward, done, info = env.step(action)

            frame_history.append(self.preprocess_func(frame))
            action_history.append(action)

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            mcts_policies.append(mcts_policy)
            root_values.append(root_value)

        game_history = (observations, actions, rewards, mcts_policies, root_values)

        return game_history


class Tester(Actor):

    def __init__(self, env_id, n_frames,
                 num_mcts_simulations, unroll_steps,
                 gamma, V_max, V_min, td_steps,
                 dirichlet_alpha):

        super().__init__(env_id, n_frames, num_mcts_simulations, unroll_steps,
                         gamma, V_max, V_min, td_steps, dirichlet_alpha)

    def play(self, current_weights):

        #: 最新のネットワークに同期
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

        total_rewards, steps = 0, 0

        done = False

        while not done:

            with tf.device("/cpu:0"):
                hidden_state, obs = self.repr_network.predict(frame_history, action_history)

            mcts_policy, root_value = mcts.search(
                hidden_state, self.num_mcts_simulations, T=0.25)

            action = np.random.choice(
                range(self.action_space), p=mcts_policy)

            frame, reward, done, info = env.step(action)

            total_rewards += reward

            step += 1

            frame_history.append(self.preprocess_func(frame))
            action_history.append(action)

        return total_rewards, steps
