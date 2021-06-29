import collections

import gym
import numpy as np

import util
from buffer import EpisodeBuffer, PrioritizedReplay
from mcts import AtariMCTS
from networks import DynamicsNetwork, PVNetwork, RepresentationNetwork


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

        state = self.repr_network.predict(frame_history, action_history)
        policy, value = self.pv_network.predict(state)
        next_state, reward = self.dynamics_network.predict(state, action=0)

        weights = (self.repr_network.get_weights(),
                   self.pv_network.get_weights(),
                   self.dynamics_network.get_weights())

        return weights


class Actor:

    def __init__(self, env_id, n_frames,
                 n_mcts_simulation,
                 gamma, V_max, V_min,
                 dirichlet_alpha):

        self.env_id = env_id

        self.n_mcts_simulation = n_mcts_simulation

        self.action_space = gym.make(env_id).action_space.n

        self.n_frames = n_frames

        self.gamma = gamma

        self.dirichlet_alpha = dirichlet_alpha

        self.V_min, self.V_max = V_min, V_max

        self.preprocess_func = util.get_preprocess_func(self.env_id)

        self.buffer = EpisodeBuffer()

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

        state = self.repr_network.predict(frame_history, action_history)
        policy, value = self.pv_network(state)
        next_state, reward = self.dynamics_network.predict(state, action=0)

    def sync_weights_and_rollout(self, current_weights, temperature):

        #: 最新のネットワークに同期
        self._sync_weights(current_weights)

        #: 1episodeのrollout
        self._rollout(temperature)

    def _sync_weights(self, weights):

        self.repr_network.set_weights(weights[0])

        self.pv_network.set_weights(weights[1])

        self.dynamics_network.set_weights(weights[2])

    def _rollout(self, temperature):

        env = gym.make(self.env_id)

        frame = self.preprocess_func(env.reset())

        frame_history = collections.deque(
            [frame] * self.n_frames, maxlen=self.n_frames)

        action_history = collections.deque(
            [0] * self.n_frames, maxlen=self.n_frames)

        mcts = AtariMCTS(
            n_mcts_simulation=self.n_mcts_simulation,
            action_space=self.action_space,
            pv_network=self.pv_network,
            dynamics_network=self.dynamics_network,
            gamma=self.gamma,
            dirichlet_alpha=self.dirichlet_alpha)

        done = False

        while not done:

            latent_state = self.repr_network.predict(frame_history, action_history)

            mcts_policy = mcts.search(latent_state, self.n_mcts_simulation)

            action = np.random.choice(range(self.action_space), p=mcts_policy)

            frame, reward, done, info = env.step(action)

            frame_history.append(self.preprocess_func(frame))

            action_history.append(action)


def main(env_id="BreakoutDeterministic-v4",
         n_episodes=10000,
         n_frames=8, gamma=0.997,
         V_min=-30, V_max=30, dirichlet_alpha=0.25,
         buffer_size=2**21, n_mcts_simulation=10):
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

    actor = Actor(env_id=env_id, n_frames=n_frames,
                  n_mcts_simulation=n_mcts_simulation,
                  V_min=V_min, V_max=V_max, gamma=gamma,
                  dirichlet_alpha=0.25)

    n = 0

    for _ in range(20):
        samples = actor.sync_weights_and_rollout(
            current_weights, temperature=1.0)

    while n <= n_episodes:
        break



if __name__ == '__main__':
    main()
