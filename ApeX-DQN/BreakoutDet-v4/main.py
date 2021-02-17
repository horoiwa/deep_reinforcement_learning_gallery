import collections

import ray
import gym
import numpy as np
import tensorflow as tf

from model import DuelingQNetwork
from buffer import GlobalPrioritizedReplayBuffer, LocalReplayBuffer


def preprocess_frame(frame):
    """ frame preprocessing only for Breakout """
    image = tf.cast(tf.convert_to_tensor(frame), tf.float32)
    image_gray = tf.image.rgb_to_grayscale(image)
    image_crop = tf.image.crop_to_bounding_box(image_gray, 34, 0, 160, 160)
    image_resize = tf.image.resize(image_crop, [84, 84])
    image_scaled = tf.divide(image_resize, 255)

    frame = image_scaled.numpy()[:, :, 0]

    return frame


class Actor:

    def __init__(self, pid, env_name, epsilon, alpha, buffer_size, n_frames,
                 gamma, nstep, reward_clip, compress):

        self.pid = pid

        self.env = gym.make(env_name)

        self.epsilon = epsilon

        self.gamma = gamma

        self.alpha = alpha

        self.n_frames = n_frames

        self.action_space = self.env.action_space.n

        self.frames = collections.deque(maxlen=n_frames)

        self.nstep = nstep

        self.buffer_size = buffer_size

        self.local_buffer = LocalReplayBuffer(
            reward_clip=reward_clip, nstep=nstep,
            gamma=gamma, compress=compress)

        self.local_qnet = DuelingQNetwork(action_space=self.action_space)

        self.define_network()

    def define_network(self):

        frame = preprocess_frame(self.env.reset())
        for _ in range(self.n_frames):
            self.frames.append(frame)

        state = np.stack(self.frames, axis=2)[np.newaxis, ...]
        self.local_qnet(state)

    def set_weights_and_rollout(self, current_weights):

        self.local_qnet.set_weights(current_weights)

        state = np.stack(self.frames, axis=2)[np.newaxis, ...]

        while True:

            state = np.stack(self.frames, axis=2)[np.newaxis, ...]

            action = self.local_qnet.sample_action(state, self.epsilon)

            next_frame, reward, done, _ = self.env.step(action)

            self.frames.append(preprocess_frame(next_frame))

            next_state = np.stack(self.frames, axis=2)[np.newaxis, ...]

            transition = (state, action, reward, next_state, done)

            self.local_buffer.push(transition)

            if done:
                frame = preprocess_frame(self.env.reset())
                for _ in range(self.n_frames):
                    self.frames.append(frame)

            if len(self.local_buffer) == self.buffer_size:
                minibatch, experiences = self.local_buffer.pull()
                states, actions, rewards, next_states, dones = minibatch

                next_actions, next_qvalues = self.local_qnet.sample_actions(next_states)

                next_actions_onehot = tf.one_hot(next_actions, self.action_space)
                max_next_qvalues = tf.reduce_sum(
                    next_qvalues * next_actions_onehot, axis=1, keepdims=True)

                TQ = rewards + self.gamma ** (self.nstep) * (1 - dones) * max_next_qvalues

                qvalues = self.local_qnet(states)
                actions_onehot = tf.one_hot(
                    actions.flatten().astype(np.int32), self.action_space)
                Q = tf.reduce_sum(qvalues * actions_onehot, axis=1, keepdims=True)

                priorities = (np.abs(TQ - Q) + 0.001) ** self.alpha

                return priorities, experiences


class Learner:

    def __init__(self, env_name="BreakoutDeterministic-v4",
                 num_workers=5, gamma=0.99, batch_size=512,
                 n_frames=4, lr=0.00025,
                 update_period=4, target_update_period=500000,
                 reward_clip=True, nstep=3, alpha=0.6, beta_init=0.4,
                 global_buffer_size=2000000, priority_capacity=2**21,
                 local_buffer_size=50,
                 compress=False):

        self.env_name = env_name

        self.gamma = gamma

        self.n_frames = n_frames

        self.actors = [
            Actor(pid, env_name=env_name, epsilon=0.5, buffer_size=local_buffer_size,
                         gamma=gamma, n_frames=n_frames, alpha=alpha,
                         reward_clip=reward_clip, nstep=nstep, compress=compress)
            for pid in range(num_workers)]

        self.buffer = GlobalPrioritizedReplayBuffer(
            max_len=global_buffer_size, capacity=priority_capacity,
            alpha=alpha, compress=compress)

        self.action_space = gym.make(env_name).action_space.n

        self.qnet = DuelingQNetwork(action_space=self.action_space)

        self.target_qnet = DuelingQNetwork(action_space=self.action_space)

        self.define_network()

    def define_network(self):

        env = gym.make(self.env_name)
        frame = preprocess_frame(env.reset())
        frames = [frame] * self.n_frames
        state = np.stack(frames, axis=2)[np.newaxis, ...]

        self.qnet(state)
        self.target_qnet(state)

        self.target_qnet.set_weights(self.qnet.get_weights())

    def learn(self, n_updates):

        updates = 0

        current_weights = self.qnet.get_weights()

        wip = [actor.set_weights_and_rollout(current_weights) for actor in self.actors]

        for i in range(10):
            pass

        else:
            updates += 1



def main():
    ray.init(local_mode=True)
    learner = Learner(num_workers=1)
    learner.learn(30)


if __name__ == "__main__":
    main()
