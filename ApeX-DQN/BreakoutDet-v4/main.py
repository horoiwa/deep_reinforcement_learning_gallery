import collections
import time
import pickle
import zlib
import queue

import ray
import gym
import numpy as np
import tensorflow as tf
from PIL import Image

from model import DuelingQNetwork
from buffer import GlobalReplayBuffer
from worker import Actor
from util import preprocess_frame


@ray.remote(num_cpus=1, num_gpus=1)
class Learner:

    def __init__(self, env_name, gamma, n_frames, lr):

        self.env_name = env_name

        self.gamma = gamma

        self.n_frames = n_frames

        self.action_space = gym.make(env_name).action_space.n

        self.qnet = DuelingQNetwork(action_space=self.action_space)

        self.target_qnet = DuelingQNetwork(action_space=self.action_space)

        self.in_queue = queue.Queue(maxsize=16)

        self.out_queue = queue.Queue(maxsize=16)

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

    def update(self, mini_batchs, compress):
        """
            解凍, 勾配計算、重み返却
        """
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

    work_in_progreses = [actor.rollout.remote(current_weights) for actor in actors]

    for i in range(50000 // local_buffer_size):
        finished, work_in_progreses = ray.wait(work_in_progreses, num_returns=1)
        priorities, experiences, pid = ray.get(finished)
        global_buffer.push(priorities, experiences)
        work_in_progreses.extend([actors[pid].rollout.remote(current_weights)])

    import pdb; pdb.set_trace()
    learning_steps = 0

if __name__ == "__main__":
    main(num_actors=4)
