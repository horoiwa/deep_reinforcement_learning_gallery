import gym
import numpy as np

import util


class Learner:

    def __init__(self, env_id):

        self.env_id = env_id

        self.action_space = gym.make(env_id).action_space.n

        self.preprocess_func = util.get_preprocess_func(self.env_id)

    def build_network(self):
        env = gym.make(self.env_id)
        frame = self.preprocess_func(env.reset())
        frames = [frame] * self.n_frames
        observations = np.concatenate(frames, axis=2)[np.newaxis, ...]

        return None


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
         n_frames=32, gamma=0.997):

    learner = Learner(env_id=env_id)
    actor = Actor(env_id=env_id, n_frames=n_frames, gamma=gamma)


if __name__ == '__main__':
    main()
