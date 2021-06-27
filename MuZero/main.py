import math

import gym
import numpy as np

import util
from networks import RepresentationNetwork, DynamicsNetwork, PVNetwork


class Learner:

    def __init__(self, env_id, n_frames: int,
                 V_min: int, V_max: int, gamma: float):

        self.env_id = env_id

        self.n_frames = n_frames

        self.V_min, self.V_max = V_min, V_max

        self.n_supprts = V_max - V_min + 1

        self.gamma = gamma

        self.action_space = gym.make(env_id).action_space.n

        self.repr_network = RepresentationNetwork(
            action_space=self.action_space)

        self.pv_network = PVNetwork(action_space=self.action_space,
                                    n_supports=self.n_supprts)

        self.dynamics_network = DynamicsNetwork(action_space=self.action_space,
                                                n_supports=self.n_supprts)

        self.preprocess_func = util.get_preprocess_func(self.env_id)

    def build_network(self):
        """ initialize network parameter """

        env = gym.make(self.env_id)
        frame = self.preprocess_func(env.reset())

        frame_history = [frame] * self.n_frames
        action_history = [0] * self.n_frames

        state = self.repr_network.predict(frame_history, action_history)
        policy, value = self.pv_network(state)
        next_state, reward = self.dynamics_network(state)

        weights = (self.repr_network.get_weights(),
                   self.pv_network.get_weights(),
                   self.dynamics_network.get_weights())

        return weights


class Actor:

    def __init__(self, env_id, n_frames=32, gamma=0.997):

        self.env_id = env_id

        self.action_space = gym.make(env_id).action_space.n

        self.n_frames = n_frames

        self.gamma = gamma

        self.preprocess_func = util.get_preprocess_func(self.env_id)

        self._build_network()

    def _build_network(self):
        env = gym.make(self.env_id)
        frame = self.preprocess_func(env.reset())
        frames = [frame] * self.n_frames
        observations = np.concatenate(frames, axis=2)[np.newaxis, ...]

        return None


def main(env_id="BreakoutDeterministic-v4",
         n_frames=8, gamma=0.997,
         V_min=-30, V_max=30):
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

    #actor = Actor(env_id=env_id, n_frames=n_frames, gamma=gamma)


if __name__ == '__main__':
    main()
