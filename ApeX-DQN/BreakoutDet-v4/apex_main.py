import time
import pickle
import zlib
import shutil
from pathlib import Path
from concurrent import futures

import ray
import tensorflow as tf
import gym
import numpy as np

from model import DuelingQNetwork
from buffer import GlobalReplayBuffer
from remote_actor import Actor, RemoteTestActor
from util import preprocess_frame, Timer, huber_loss


@ray.remote(num_cpus=1, num_gpus=1)
class Learner:

    def __init__(self, env_name, gamma, nstep,
                 target_update_period, n_frames):

        self.env_name = env_name

        self.gamma = gamma

        self.nstep = nstep

        self.action_space = gym.make(env_name).action_space.n

        self.qnet = DuelingQNetwork(action_space=self.action_space)

        self.target_qnet = DuelingQNetwork(action_space=self.action_space)

        self.target_update_period = target_update_period

        self.n_frames = n_frames

        #self.optimizer = tf.keras.optimizers.Adam(lr=0.0001)

        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate= 0.00025 / 4, rho=0.95, momentum=0.0,
            epsilon=1.5e-07, centered=True)

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

    def save(self, save_path):
        self.qnet.save_weights(save_path)

    def update_qnetwork(self, compressed_minibatchs):

        indices_all, td_errors_all = [], []
        loss_list = []
        with futures.ThreadPoolExecutor(max_workers=4) as executor:
            """ batchをdecompressして整形する作業がわりと重いのでthreading
            """
            work_in_progresses = [
                executor.submit(self.prepare_minibatch, compressed)
                for compressed in compressed_minibatchs]

            for ready_batch in futures.as_completed(work_in_progresses):

                indices, per_weights, minibacth = ready_batch.result()
                states, actions, rewards, next_states, dones = minibacth

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

                    #td_loss = huber_loss(target_q, q)
                    td_loss = tf.square(target_q - q)
                    loss = tf.reduce_mean(per_weights * td_loss)

                grads = tape.gradient(loss, self.qnet.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, 40.0)
                self.optimizer.apply_gradients(
                    zip(grads, self.qnet.trainable_variables))

                indices_all += indices
                td_errors_all += td_loss.numpy().flatten().tolist()
                loss_list.append(loss.numpy())
                self.update_count += 1

                if self.update_count % self.target_update_period == 0:
                    print("== target_update ==")
                    self.target_qnet.set_weights(self.qnet.get_weights())

        loss_mean = np.array(loss_list).mean()
        current_weights = self.qnet.get_weights()
        return current_weights, indices_all, td_errors_all, loss_mean

    @staticmethod
    def prepare_minibatch(compressed_minibatch):
        indices, per_weights, experiences = compressed_minibatch

        per_weights = tf.convert_to_tensor(
            per_weights.reshape(-1, 1), dtype=tf.float32)

        experiences = [pickle.loads(zlib.decompress(exp)) for exp in experiences]

        states = np.vstack([exp.state for exp in experiences]).astype(np.float32)
        actions = np.vstack([exp.action for exp in experiences]).astype(np.float32)
        rewards = np.array([exp.reward for exp in experiences]).reshape(-1, 1)
        next_states = np.vstack(
            [exp.next_state for exp in experiences]).astype(np.float32)
        dones = np.array([exp.done for exp in experiences]).reshape(-1, 1)

        return indices, per_weights, (states, actions, rewards, next_states, dones)


def main(num_actors, env_name="BreakoutDeterministic-v4",
         gamma=0.99, batch_size=512,
         n_frames=4, epsilon=0.5, eps_alpha=7.,
         target_update_period=2400, num_minibatchs=16,
         reward_clip=True, nstep=3, alpha=0.6, beta=0.4,
         global_buffer_size=2**21,
         local_buffer_size=100, compress=True):

    ray.init(local_mode=False)

    logdir = Path(__file__).parent / "log"
    if logdir.exists():
        shutil.rmtree(logdir)
    summary_writer = tf.summary.create_file_writer(str(logdir))

    global_buffer = GlobalReplayBuffer(
        capacity=global_buffer_size,
        alpha=alpha, beta=beta)

    #epsilons = np.linspace(0.05, 0.4, num_actors)
    epsilons = [epsilon ** (1 + eps_alpha * i / (num_actors - 1)) for i in range(num_actors)]
    epsilons = [max(0.01, eps) for eps in epsilons]

    actors = [Actor.remote(
        pid=i, env_name=env_name,
        epsilon=epsilons[i],
        buffer_size=local_buffer_size,
        gamma=gamma, n_frames=n_frames, alpha=alpha,
        reward_clip=reward_clip, nstep=nstep,
        ) for i in range(num_actors)]

    learner = Learner.remote(
        env_name=env_name, gamma=gamma, nstep=nstep,
        target_update_period=target_update_period,
        n_frames=n_frames)

    current_weights = ray.put(ray.get(learner.define_network.remote()))

    test_actor = RemoteTestActor.remote(env_name=env_name)

    work_in_progreses = [actor.rollout.remote(current_weights) for actor in actors]

    learner_count = 0
    MIN_EXPERIENCES = 50000
    for _ in range(MIN_EXPERIENCES // local_buffer_size):
        finished, work_in_progreses = ray.wait(work_in_progreses, num_returns=1)
        priorities, experiences, pid = ray.get(finished[0])
        global_buffer.push(priorities, experiences)
        work_in_progreses.extend([actors[pid].rollout.remote(current_weights)])

    print("Setup finished")

    minibatchs = [global_buffer.sample_batch(batch_size) for _ in range(num_minibatchs)]
    learner_future = learner.update_qnetwork.remote(minibatchs)
    learner_count += 1

    next_minibatchs = [global_buffer.sample_batch(batch_size) for _ in range(num_minibatchs)]

    tester_future = test_actor.play.remote(current_weights, epsilon=0.01)

    s = time.time()
    count = 0
    while learner_count <= 5000:

        actor_finished, work_in_progreses = ray.wait(work_in_progreses, num_returns=1)
        priorities, experiences, pid = ray.get(actor_finished[0])
        global_buffer.push(priorities, experiences)
        work_in_progreses.extend([actors[pid].rollout.remote(current_weights)])
        count += 1

        learner_finished, _ = ray.wait([learner_future], timeout=0)

        if learner_finished:
            print("Actor cycle", count)
            print("Leaner", learner_count)
            current_weights, indices, td_errors, loss_mean = ray.get(learner_finished[0])
            current_weights = ray.put(current_weights)

            learner_future = learner.update_qnetwork.remote(next_minibatchs)

            global_buffer.update_priorities(indices, td_errors)
            next_minibatchs = [global_buffer.sample_batch(batch_size) for _ in range(num_minibatchs)]

            learner_count += 1
            count = 0
            with summary_writer.as_default():
                tf.summary.scalar("learner_loss", loss_mean, step=learner_count)

            if learner_count % 10 == 0:
                episode_steps, episode_rewards = ray.get(tester_future)
                print("TEST:", episode_steps, episode_rewards)
                layers = ray.get(test_actor.get_layers.remote(-3))
                tester_future = test_actor.play.remote(current_weights, epsilon=0.01)
                elapsed_time = (time.time() - s) / 10

                with summary_writer.as_default():
                    tf.summary.scalar("test_steps", episode_steps, step=learner_count)
                    tf.summary.scalar("test_rewards", episode_rewards, step=learner_count)
                    tf.summary.scalar("buffer_size", len(global_buffer), step=learner_count)
                    tf.summary.scalar("Elapsed time", elapsed_time, step=learner_count)

                    for layer in layers:
                        for var in layer.variables:
                            tf.summary.histogram(var.name, var, step=learner_count)

                print(learner_count, episode_steps,  episode_rewards)
                s = time.time()

            if learner_count % 500 == 0:
                print("Model Saved")
                learner.save.remote("checkpoints/qnet")


def test_play(env_name="BreakoutDeterministic-v4"):

    ray.init()
    test_actor = RemoteTestActor.remote(env_name=env_name)
    res = test_actor.play_with_video.remote(
            checkpoint_path="checkpoints/qnet", monitor_dir="mp4", epsilon=0.01)
    rewards = ray.get(res)
    print(rewards)


if __name__ == "__main__":
    start = time.time()
    main(num_actors=21)
    print("Finished:", time.time() - start)
    ray.shutdown()
    test_play()
