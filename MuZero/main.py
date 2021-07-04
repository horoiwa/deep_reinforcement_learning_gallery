import collections
from dataclasses import dataclass
import pickle

import gym
import numpy as np
import tensorflow as tf
import lz4.frame as lz4f

import util
from buffer import PrioritizedReplay
from mcts import AtariMCTS
from networks import DynamicsNetwork, PVNetwork, RepresentationNetwork


@dataclass
class Sample:

    observation: np.ndarray
    actions: list
    mcts_policies: list
    rewards: list
    values: float


class Learner:

    def __init__(self, env_id, n_frames: int,
                 V_min: int, V_max: int, gamma: float):

        self.env_id = env_id

        self.n_frames = n_frames

        self.V_min, self.V_max = V_min, V_max

        self.gamma = gamma

        self.action_space = gym.make(env_id).action_space.n

        self.repr_network = RepresentationNetwork(
            action_space=self.action_space)

        self.pv_network = PVNetwork(action_space=self.action_space,
                                    V_min=V_min, V_max=V_max)

        self.dynamics_network = DynamicsNetwork(action_space=self.action_space,
                                                V_min=V_min, V_max=V_max)

        self.preprocess_func = util.get_preprocess_func(self.env_id)

    def build_network(self):
        """ initialize network parameter """

        env = gym.make(self.env_id)
        frame = self.preprocess_func(env.reset())

        frame_history = [frame] * self.n_frames
        action_history = [0] * self.n_frames

        state, obs = self.repr_network.predict(frame_history, action_history)
        policy, value = self.pv_network.predict(state)
        next_state, reward = self.dynamics_network.predict(state, action=0)

        weights = (self.repr_network.get_weights(),
                   self.pv_network.get_weights(),
                   self.dynamics_network.get_weights())

        return weights

    def update_network(self, minibatchs):

        indices_all, priorities_all, losses = [], [], []

        for (indices, weights, samples) in minibatchs:

            samples = [pickle.loads(lz4f.decompress(sample)) for sample in samples]

            priorities, loss = self._update(weights, samples)

            indices_all += indices
            priorities_all += priorities
            losses.append(loss)

        current_weights = self.q_network.get_weights()
        loss_mean = np.array(losses).mean()

        return current_weights, indices_all, priorities_all, loss_mean

    def _update(self, weights, samples):

        #: Network inputs
        observations = tf.concat([s.observation for s in samples], axis=0)  #: (batchsize, ...)
        actions = tf.stack([s.actions for s in samples], axis=1)            #: (unroll_steps, batchsize)

        #: Targets
        mcts_policies = tf.stack([s.mcts_policies for s in samples], axis=1)  #: (unroll_steps, batch_size, action_space)
        rewards_scalar = tf.stack([s.rewards for s in samples], axis=1)       #: (unroll_steps, batch_size)
        values_scalar = tf.stack([s.values for s in samples], axis=1)          #: (unroll_steps, batch_size)
        import pdb; pdb.set_trace()


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
        self._sync_weights(current_weights)

        #: 1episodeのrollout
        game_history = self._rollout(T)

        samples, priorities = self._make_samples(game_history)

        return samples, priorities

    def _make_samples(self, game_history):
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
            np.array([1. / self.action_space] * self.action_space)
            for _ in range(self.td_steps)]

        values = []
        for idx in range(episode_len):

            bootstrap_idx = idx + self.td_steps

            #: remaining value
            if bootstrap_idx < episode_len:
                value = root_values[bootstrap_idx]
            else:
                value = 0

            value += sum([r * self.gamma ** i for i, r in
                          enumerate(rewards[idx:bootstrap_idx])])
            values.append(value)

        for idx in range(episode_len):

            #: shape == (unroll_steps, ...)
            _rewards = np.array(rewards[idx:idx+self.unroll_steps])
            _actions = np.array(actions[idx:idx+self.unroll_steps])
            _mcts_policies = np.vstack(mcts_policies[idx:idx+self.unroll_steps])
            _values = np.array(values[idx:idx+self.unroll_steps])

            sample = Sample(
                observation=observations[idx],
                actions=_actions,
                mcts_policies=_mcts_policies,
                rewards=_rewards,
                values=_values)

            samples.append(sample)

        priorities = [abs(sample.value - root_value)
                      for sample, root_value in zip(samples, root_values)]

        #: Compress for memory efficiency
        samples = [lz4f.compress(pickle.dumps(sample)) for sample in samples]

        return samples, priorities

    def _sync_weights(self, weights):

        self.repr_network.set_weights(weights[0])

        self.pv_network.set_weights(weights[1])

        self.dynamics_network.set_weights(weights[2])

    def _rollout(self, T):

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

            debug = True
            if not debug:

                mcts_policy, root_value = mcts.search(
                    hidden_state, self.num_mcts_simulations, T)

                action = np.random.choice(
                    range(self.action_space), p=mcts_policy)

            else:
                mcts_policy, root_value = np.array([0.1, 0.1, 0.3, 0.5]), 1.0
                action = np.random.choice(range(self.action_space), p=mcts_policy)

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


def main(env_id="BreakoutDeterministic-v4",
         n_episodes=10000, unroll_steps=5,
         n_frames=8, gamma=0.997, td_steps=10,
         V_min=-30, V_max=30, dirichlet_alpha=0.25,
         buffer_size=2**21, num_mcts_simulations=10,
         batchsize=256):
    """

    Args:
        n_frames (int): num of stacked RGB frames. Defaults to 8. (original 32)
        gamma (float): discount factor. Defaults to 0.997.
        V_min, V_max (int):
            assumed range of rescaled rewards,
            -30 ~ 30 corresponds to roughly score -1000 ~ 1000
            (original -300 ~ 300)

    Changes from original paper:
        - Use Grey scaled frame instead of RGB frame
        - Reduce the number of residual blocks for compuational efficiency.
    """

    learner = Learner(env_id=env_id, n_frames=n_frames,
                      V_min=V_min, V_max=V_max, gamma=gamma)

    current_weights = learner.build_network()

    buffer = PrioritizedReplay(capacity=buffer_size)

    actor = Actor(env_id=env_id, n_frames=n_frames, unroll_steps=unroll_steps,
                  num_mcts_simulations=num_mcts_simulations, td_steps=td_steps,
                  V_min=V_min, V_max=V_max, gamma=gamma,
                  dirichlet_alpha=0.25)

    n = 0

    for _ in range(2):
        samples, priorities = actor.sync_weights_and_rollout(current_weights, T=1.0)
        buffer.add_samples(priorities, samples)
        n += 1

    while n <= n_episodes:

        samples, priorities = actor.sync_weights_and_rollout(current_weights, T=1.0)
        buffer.add_samples(priorities, samples)
        n += 1

        minibatchs = [buffer.sample_minibatch(batchsize=batchsize)]

        learner.update_network(minibatchs)




if __name__ == '__main__':
    main()
