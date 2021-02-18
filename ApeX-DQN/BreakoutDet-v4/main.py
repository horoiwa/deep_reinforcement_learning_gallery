import collections
import time
import pickle
import zlib

import ray
import gym
import numpy as np
import tensorflow as tf
from PIL import Image

from model import DuelingQNetwork
from buffer import GlobalReplayBuffer, LocalReplayBuffer


def preprocess_frame(frame):
    """Breakout only"""
    image = Image.fromarray(frame)
    image = image.convert("L").crop((0, 34, 160, 200)).resize((84, 84))
    image_scaled = np.array(image) / 255.0
    return image_scaled.astype(np.float32)


@ray.remote(num_cpus=1)
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
            reward_clip=reward_clip, gamma=gamma, nstep=nstep)

        self.local_qnet = DuelingQNetwork(action_space=self.action_space)

        self.compress = compress

        self.define_network()

    def define_network(self):

        #: define by run
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

                priorities = ((np.abs(TQ - Q) + 0.001) ** self.alpha).flatten()

                if self.compress:
                    experiences = [zlib.compress(pickle.dumps(exp)) for exp in experiences]

                return priorities, experiences, self.pid


@ray.remote(num_cpus=1, num_gpus=1)
class Learner:

    def __init__(self, env_name, gamma, n_frames, lr):

        self.env_name = env_name

        self.gamma = gamma

        self.n_frames = n_frames

        self.action_space = gym.make(env_name).action_space.n

        self.qnet = DuelingQNetwork(action_space=self.action_space)

        self.target_qnet = DuelingQNetwork(action_space=self.action_space)

    def define_network(self):

        env = gym.make(self.env_name)
        frame = preprocess_frame(env.reset())
        frames = [frame] * self.n_frames
        state = np.stack(frames, axis=2)[np.newaxis, ...]

        #: define by run
        self.qnet(state)
        self.target_qnet(state)
        self.target_qnet.set_weights(self.qnet.get_weights())

        return self.qnet.get_weights()

    def update(self, minibatchs):
        for batch in minibatchs:
            pass

        weights = self.qnet.get_weights()
        return weights

    def learn(self, n_updates):

        learning_steps = 0

        current_weights = ray.put(self.qnet.get_weights())

        work_in_progresses = [
            actor.set_weights_and_rollout.remote(current_weights)
            for actor in self.actors]

        s = time.time()
        for _ in range(12):
            finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
            priorities, experiences, pid = ray.get(finished[0])
            self.buffer.push_on_batch.remote(priorities, experiences)
            work_in_progresses.append(
                self.actors[pid].set_weights_and_rollout.remote(current_weights))
        print(time.time() - s)


def main(env_name="BreakoutDeterministic-v4",
         num_actors=5, gamma=0.99, batch_size=512,
         n_frames=4, lr=0.00025, epsilon=0.4, eps_alpha=0.7,
         update_period=50, target_update_period=2500,
         reward_clip=True, nstep=3, alpha=0.6, beta=0.4,
         global_buffer_size=2000000, priority_capacity=2**21,
         local_buffer_size=100, compress=True):

    ray.init(local_mode=False)

    learner = Learner.remote(
        env_name=env_name, gamma=gamma, n_frames=n_frames, lr=lr)

    current_weights = ray.put(ray.get(learner.define_network.remote()))

    global_buffer = GlobalReplayBuffer(
        max_len=global_buffer_size, capacity=priority_capacity,
        alpha=alpha, beta=beta, compress=compress)

    actors = [Actor.remote(
        pid=i, env_name=env_name,
        epsilon=epsilon ** (1 + eps_alpha * i / (num_actors - 1)),
        buffer_size=local_buffer_size,
        gamma=gamma, n_frames=n_frames, alpha=alpha,
        reward_clip=reward_clip, nstep=nstep,
        compress=compress) for i in range(num_actors)]

    learning_steps = 0
    while True:
        #: collect trajectory and add global_buffer
        break
        #: 1 update
        pass
        if learning_steps % 100 == 0:
            gobal_buffer.remove()


if __name__ == "__main__":
    main(num_actors=4)
