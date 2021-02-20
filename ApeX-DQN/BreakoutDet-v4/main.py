import time
import pickle
import zlib

import ray
import tensorflow as tf
from tensorflow.python.client import device_lib
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
import gym
import numpy as np

from model import DuelingQNetwork
from buffer import GlobalReplayBuffer
from remote_actor import Actor
from util import preprocess_frame, Timer, huber_loss


@ray.remote
def create_batch(minibatch):
    pass


@ray.remote(num_cpus=1, num_gpus=1)
class Learner:

    def __init__(self, env_name, gamma, nstep, lr,
                 target_update_period, n_frames):

        self.env_name = env_name

        self.gamma = gamma

        self.nstep = nstep

        self.action_space = gym.make(env_name).action_space.n

        self.qnet = DuelingQNetwork(action_space=self.action_space)

        self.target_qnet = DuelingQNetwork(action_space=self.action_space)

        self.target_update_period = target_update_period

        self.n_frames = n_frames

        self.optimizer = tf.keras.optimizers.Adam(lr=lr)

        self.update_count = 0

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

    def update(self, in_queue):
        """
            解凍, 勾配計算、重み返却
            解凍をthreadingすると速い？
        """

        indices_all = []

        td_errors_all = []

        for (indices, per_weights, experiences) in in_queue:

            #: prepare
            per_weights = tf.convert_to_tensor(
                per_weights.reshape(-1, 1), dtype=tf.float32)
            experiences = [pickle.loads(zlib.decompress(exp)) for exp in experiences]

            states = np.vstack([exp.state for exp in experiences]).astype(np.float32)
            actions = np.vstack([exp.action for exp in experiences]).astype(np.float32)
            rewards = np.array([exp.reward for exp in experiences]).reshape(-1, 1)
            next_states = np.vstack(
                [exp.next_state for exp in experiences]).astype(np.float32)
            dones = np.array([exp.done for exp in experiences]).reshape(-1, 1)

            #: compute gradients
            next_actions, _ = self.qnet.sample_actions(next_states)
            _, next_qvalues = self.target_qnet.sample_actions(next_states)

            next_actions_onehot = tf.one_hot(next_actions, self.action_space)
            max_next_qvalues = tf.reduce_sum(
                next_qvalues * next_actions_onehot, axis=1, keepdims=True)

            target_q = rewards + self.gamma ** (self.nstep) * (1 - dones) * max_next_qvalues

            with tf.GradientTape() as tape:

                qvalues = self.qnet(states)
                actions_onehot = tf.one_hot(
                    actions.flatten().astype(np.int32), self.action_space)
                q = tf.reduce_sum(
                    qvalues * actions_onehot, axis=1, keepdims=True)
                td_loss = huber_loss(target_q, q)

                loss = tf.reduce_mean(per_weights * td_loss)

            grads = tape.gradient(loss, self.qnet.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.qnet.trainable_variables))

            indices_all += indices
            td_errors_all += td_loss.numpy().flatten().tolist()
            self.update_count += 1

            if self.update_count % self.target_update_period == 0:
                self.target_qnet.set_weights(self.qnet.get_weights())

        current_weights = self.qnet.get_weights()
        return current_weights, indices_all, td_errors_all


def main(env_name="BreakoutDeterministic-v4",
         num_actors=5, gamma=0.99, batch_size=512,
         n_frames=4, lr=0.00025, epsilon=0.4, eps_alpha=0.7,
         target_update_period=2500,
         reward_clip=True, nstep=3, alpha=0.6, beta=0.4,
         global_buffer_size=2000000, priority_capacity=2**21,
         local_buffer_size=100, compress=True):

    ray.init(local_mode=False)

    learner = Learner.remote(
        env_name=env_name, gamma=gamma, nstep=nstep, lr=lr,
        target_update_period=target_update_period, n_frames=n_frames)

    global_buffer = GlobalReplayBuffer(
        max_len=global_buffer_size, capacity=priority_capacity,
        alpha=alpha, beta=beta)

    actors = [Actor.remote(
        pid=i, env_name=env_name,
        epsilon=epsilon ** (1 + eps_alpha * i / (num_actors - 1)),
        buffer_size=local_buffer_size,
        gamma=gamma, n_frames=n_frames, alpha=alpha,
        reward_clip=reward_clip, nstep=nstep,
        ) for i in range(num_actors)]

    current_weights = ray.put(ray.get(learner.define_network.remote()))
    work_in_progreses = [actor.rollout.remote(current_weights) for actor in actors]

    with Timer("10000遷移収集"):
        for _ in range(100):
            finished, work_in_progreses = ray.wait(work_in_progreses, num_returns=1)
            priorities, experiences, pid = ray.get(finished[0])
            global_buffer.push(priorities, experiences)
            work_in_progreses.extend([actors[pid].rollout.remote(current_weights)])

    in_queue = []
    with Timer("16バッチ作成"):
        for _ in range(16):
            minibatch = global_buffer.sample_minibatch(batch_size)
            in_queue.append(minibatch)

    with Timer("16バッチ学習"):
        orf = learner.update.remote(in_queue)
        current_weights, indices, td_errors = ray.get(orf)
        current_weights = ray.put(current_weights)

    ray.shutdown()


if __name__ == "__main__":
    """ Memo
    """
    main(num_actors=6)
