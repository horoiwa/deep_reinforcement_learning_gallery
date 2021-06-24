import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.python.keras.backend import dtype


class RepresentationNetwork(tf.keras.Model):

    def __init__(self, action_space):
        super(RepresentationNetwork, self).__init__()
        self.action_space = action_space

    def call(self, observations, training=False):
        """
            observations: <batch_size, 96, 96, 3*n_frames + n_frames> for atari.
              first (3 * frames) planes are for RGB * n_frames
              and (+ frames) planes are for hitorical actions
        """

        state = None
        return state

    def predict(self, frames: list, action_history: list):

        h, w, length = frames[0].shape[0], frames[0].shape[1], len(frames)

        frames = np.concatenate(frames, axis=2)

        actions = np.ones((h, w, length), dtype=np.float32)
        action_history = np.array(action_history, dtype=np.float32)
        actions = actions * action_history / (self.action_space - 1)

        observations = np.concatenate([frames, actions], axis=2)
        observations = observations[np.newaxis, ...]

        state = self(observations)

        return state


class PVNetwork(tf.keras.Model):

    def __init__(self, action_space):
        super(DynamicsNetwork, self).__init__()


class DynamicsNetwork(tf.keras.Model):

    def __init__(self, action_space):
        super(DynamicsNetwork, self).__init__()



if __name__ == '__main__':
    import gym
    import util

    n_frames = 8
    env_name = "BreakoutDeterministic-v4"
    f = util.get_preprocess_func(env_name)

    env = gym.make(env_name)
    action_space = env.action_space.n

    frame = f(env.reset())

    frames = [frame] * n_frames
    actions_history = [0, 1, 2, 3, 0, 1, 2, 3]

    repr_function = RepresentationNetwork(action_space=action_space)
    repr_function.predict(frames, actions_history)
