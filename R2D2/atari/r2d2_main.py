import shutil
import pickle
from concurrent import futures
import time
from pathlib import Path

import gym
import matplotlib.pyplot as plt
import numpy as np
import ray
import tensorflow as tf
import lz4.frame as lz4f

import util
from buffer import SegmentReplayBuffer
from model import RecurrentDuelingQNetwork
from actor import Actor, Tester


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def value_function_rescaling(x):
    """https://github.com/google-research/seed_rl/blob/f53c5be4ea083783fb10bdf26f11c3a80974fa03/agents/r2d2/learner.py#L180
    """
    eps = 0.001
    return tf.math.sign(x) * (tf.math.sqrt(tf.math.abs(x) + 1.) - 1.) + eps * x


def inverse_value_function_rescaling(x):
    """https://github.com/google-research/seed_rl/blob/f53c5be4ea083783fb10bdf26f11c3a80974fa03/agents/r2d2/learner.py#L186
    """
    eps = 0.001
    return tf.math.sign(x) * (
        tf.math.square(
            ((tf.math.sqrt(1. + 4. * eps * (tf.math.abs(x) + 1. + eps))) - 1.) / (2. * eps)
            ) - 1.)


@ray.remote(num_cpus=1, num_gpus=1)
class Learner:

    def __init__(self, env_name, target_update_period,
                 n_frames, gamma, eta, alpha,
                 burnin_length, unroll_length):
        self.env_name = env_name
        self.n_frames = n_frames
        self.action_space = gym.make(self.env_name).action_space.n
        self.frame_process_func = util.get_preprocess_func(env_name)

        self.q_network = RecurrentDuelingQNetwork(self.action_space)
        self.target_q_network = RecurrentDuelingQNetwork(self.action_space)
        self.target_update_period = target_update_period
        self.optimizer = tf.keras.optimizers.Adam(lr=0.00025, epsilon=0.001)

        self.gamma = gamma
        self.eta = eta
        self.alpha = alpha

        self.burnin_len = burnin_length
        self.unroll_len = unroll_length

        self.num_updated = 0

    def define_network(self):

        env = gym.make(self.env_name)

        frame = self.frame_process_func(env.reset())
        frames = [frame] * self.n_frames
        state = np.stack(frames, axis=2)[np.newaxis, ...]

        c, h = self.q_network.lstm.get_initial_state(batch_size=1, dtype=tf.float32)

        self.q_network(np.atleast_2d(state), states=[c, h], prev_action=[0])
        self.target_q_network(np.atleast_2d(state), states=[c, h], prev_action=[0])
        self.target_q_network.set_weights(self.q_network.get_weights())
        current_weights = self.q_network.get_weights()

        return current_weights

    def save(self, save_path):
        self.q_network.save_weights(save_path)

    @staticmethod
    def decompress_segments(minibatch):
        inidices, weights, compressed_segments = minibatch
        segments = [pickle.loads(lz4f.decompress(compressed_seg))
                    for compressed_seg in compressed_segments]
        return (inidices, weights, segments)

    def update_network(self, minibatchs, preprocess="concurrent"):
        """
        Args:
            minibatchs (List[Tuple(indices, weights, compressed_segments)])
              indices (List[float]): indices of replay buffer
              weights (List[float]): Importance sampling weights
              compressed_segments (List[Segment]): list of lz4-compressed segments
                Segment: sequence of transitions
        """

        indices_all = []
        priorities_all = []
        losses = []

        t = time.time()
        with futures.ThreadPoolExecutor(max_workers=2) as executor:
            """ segmentsをdecompressする作業がやや重い(0.2sec程度)のでthreading
            """
            work_in_progresses = [
                executor.submit(self.decompress_segments, mb)
                for mb in minibatchs]

            for ready_minibatch in futures.as_completed(work_in_progresses):
                (indices, weights, segments) = ready_minibatch.result()
                indices, priorities, loss = self._update(indices, weights, segments)

                indices_all += indices
                priorities_all += priorities
                losses.append(loss)

        current_weights = self.q_network.get_weights()
        loss_mean = np.array(losses).mean()
        print(time.time() - t)

        return current_weights, indices_all, priorities_all, loss_mean

    def _update(self, indices, weights, segments):

        states = np.stack([np.vstack(seg.states) for seg in segments], axis=1)    # (burnin_len+unroll_len, batch_size, obs_dim)
        actions = np.stack([seg.actions for seg in segments], axis=1)  # (burnin+unroll_len, batch_size)
        rewards = np.stack([seg.rewards for seg in segments], axis=1)  # (unroll_len, batch_size)
        dones = np.stack([seg.dones for seg in segments], axis=1)      # (unroll_len, batch_size)

        last_state = np.vstack([seg.last_state for seg in segments])   # (batch_size, obs_dim)
        last_state = tf.expand_dims(last_state, axis=0)                # (1, batch_size, obs_dim)
        next_states = tf.concat([states, last_state], axis=0)[1:]      # (burnin+unroll_len, batch_size, obs_dim)

        assert next_states.shape == states.shape

        c0 = tf.convert_to_tensor(
            np.vstack([seg.c_init for seg in segments]), dtype=tf.float32)  # (batch_size, lstm_out_dim)
        h0 = tf.convert_to_tensor(
            np.vstack([seg.h_init for seg in segments]), dtype=tf.float32)  # (batch_size, lstm_out_dim)

        a0 = np.atleast_2d([seg.prev_action_init for seg in segments])      # (batch_size, 1)
        prev_actions = np.vstack([a0, actions])[:-1]
        assert prev_actions.shape == actions.shape

        #: convert to tensor for performance reason
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        prev_actions = tf.convert_to_tensor(prev_actions, dtype=tf.int32)

        """ Compute Target Q values """
        #: Burn-in of lstem-state for target network
        _, (c, h) = self.target_q_network(
             states[0], states=[c0, h0], prev_action=prev_actions[0])
        for t in range(self.burnin_len):
            _, (c, h) = self.target_q_network(
                next_states[t], states=[c, h], prev_action=actions[t])

        #: Compute TQ
        next_qvalues = []
        for t in range(self.burnin_len, self.burnin_len + self.unroll_len):
            q, (c, h) = self.target_q_network(
                next_states[t], states=[c, h], prev_action=actions[t])
            next_qvalues.append(q)

        next_qvalues = tf.stack(next_qvalues)                              # (unroll_len, batch_size, action_space)
        next_actions = tf.argmax(next_qvalues, axis=2)                     # (unroll_len, batch_size)
        next_actions_onehot = tf.one_hot(next_actions, self.action_space)  # (unroll_len, batch_size, action_space)
        next_maxQ = tf.reduce_sum(
            next_qvalues * next_actions_onehot, axis=2, keepdims=False)    # (unroll_len, batch_size)
        next_maxQ = inverse_value_function_rescaling(next_maxQ)

        TQ = rewards + self.gamma * (1 - dones) * next_maxQ                # (unroll_len, batch_size)
        TQ = value_function_rescaling(TQ)

        """ Compute Q values and TD error """
        #: Burn-in of lstem-state for online network
        c, h = c0, h0
        for t in range(self.burnin_len):
            _, (c, h) = self.q_network(
                states[t], states=[c, h],
                prev_action=prev_actions[t])

        with tf.GradientTape() as tape:
            qvalues = []
            for t in range(self.burnin_len, self.burnin_len + self.unroll_len):
                q, (c, h) = self.q_network(
                    states[t], states=[c, h],
                    prev_action=prev_actions[t])
                qvalues.append(q)
            qvalues = tf.stack(qvalues)                            # (unroll_len, batch_size, action_space)
            actions_onehot = tf.one_hot(
                actions[self.burnin_len:], self.action_space)
            Q = tf.reduce_sum(
                qvalues * actions_onehot, axis=2, keepdims=False)  # (unroll_len, batch_size)

            td_errors = TQ - Q

            loss = tf.reduce_sum(tf.square(td_errors), axis=0)
            loss_weighted = tf.reduce_mean(loss * weights)

        grads = tape.gradient(
            loss_weighted, self.q_network.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 40.0)
        self.optimizer.apply_gradients(
            zip(grads, self.q_network.trainable_variables))

        #: Compute priority
        td_errors_abs = tf.abs(td_errors)
        priorities = self.eta * tf.reduce_max(td_errors_abs, axis=0) \
            + (1 - self.eta) * tf.reduce_mean(td_errors_abs, axis=0)
        priorities = (priorities + 0.001) ** self.alpha

        self.num_updated += 1
        if self.num_updated % self.target_update_period == 0:
            self.target_q_network.set_weights(self.q_network.get_weights())

        return indices, priorities.numpy().tolist(), loss_weighted


def main(num_actors,
         env_name="BreakoutDeterministic-v4",
         target_update_period=2400,
         buffer_size=2**16,
         n_frames=4, nstep=5,
         batch_size=32, update_iter=16,
         gamma=0.997, eta=0.9, alpha=0.9,
         burnin_length=30, unroll_length=30):

    ray.init(local_mode=False)

    logdir = Path(__file__).parent / "log"
    if logdir.exists():
        shutil.rmtree(logdir)
    summary_writer = tf.summary.create_file_writer(str(logdir))

    history = []

    epsilons = [0.5 ** (1 + 7. * i / (num_actors - 1)) for i in range(num_actors)]
    epsilons = [max(0.02, eps) for eps in epsilons]

    actors = [Actor.remote(pid=i, env_name=env_name, n_frames=n_frames,
                           epsilon=epsilons[i],
                           gamma=gamma, eta=eta, alpha=alpha,
                           nstep=nstep,
                           burnin_length=burnin_length,
                           unroll_length=unroll_length)
              for i in range(num_actors)]

    replay = SegmentReplayBuffer(buffer_size=buffer_size)

    learner = Learner.remote(env_name=env_name,
                             target_update_period=target_update_period,
                             n_frames=n_frames,
                             gamma=gamma, eta=eta, alpha=alpha,
                             burnin_length=burnin_length,
                             unroll_length=unroll_length)

    current_weights = ray.get(learner.define_network.remote())
    current_weights = ray.put(current_weights)

    tester = Tester.remote(env_name=env_name, n_frames=n_frames)

    wip_actors = [actor.sync_weights_and_rollout.remote(current_weights)
                  for actor in actors]

    for _ in range(100):
        finished, wip_actors = ray.wait(wip_actors, num_returns=1)
        priorities, segments, pid = ray.get(finished[0])
        replay.add(priorities, segments)
        wip_actors.extend(
            [actors[pid].sync_weights_and_rollout.remote(current_weights)])

    print("===="*5)

    # minibatchs: (indices, weights, segments)
    minibatchs = [replay.sample_minibatch(batch_size=batch_size)
                  for _ in range(update_iter)]
    wip_learner = learner.update_network.remote(minibatchs)

    wip_tester = tester.test_play.remote(current_weights, epsilon=0.02)

    minibatchs = [replay.sample_minibatch(batch_size=batch_size)
                  for _ in range(update_iter)]

    learner_cycles = 1
    actor_cycles = 0
    n_segment_added = 0
    s = time.time()
    while learner_cycles <= 5000:
        actor_cycles += 1
        finished, wip_actors = ray.wait(wip_actors, num_returns=1)
        priorities, segments, pid = ray.get(finished[0])
        replay.add(priorities, segments)
        wip_actors.extend(
            [actors[pid].sync_weights_and_rollout.remote(current_weights)])
        n_segment_added += len(segments)

        #if n_segment_added < (batch_size * update_iter) * 1.2:
        #: The appropriate seconds depend on your machine power
        if time.time() - s < 9.0:
            finished_learner, _ = ray.wait([wip_learner], timeout=0)
        else:
            finished_learner, _ = ray.wait([wip_learner], num_returns=1)

        if finished_learner:
            current_weights, indices, priorities, loss = ray.get(finished_learner[0])
            wip_learner = learner.update_network.remote(minibatchs)
            current_weights = ray.put(current_weights)
            #: 優先度の更新とminibatchの作成はlearnerよりも十分に速いという前提
            replay.update_priority(indices, priorities)
            minibatchs = [replay.sample_minibatch(batch_size=batch_size)
                          for _ in range(update_iter)]

            print("Actor cycle:", actor_cycles,
                  "Added segs:", n_segment_added,
                  "Elapsed time[sec]:", time.time() - s)

            learner_cycles += 1
            actor_cycles = 0
            n_segment_added = 0
            s = time.time()

            with summary_writer.as_default():
                tf.summary.scalar("learner_loss", loss, step=learner_cycles)

            if learner_cycles % 5 == 0:

                test_rewards = ray.get(wip_tester)
                history.append((learner_cycles-5, test_rewards))
                wip_tester = tester.test_play.remote(current_weights, epsilon=0.02)
                print("Cycle:", learner_cycles, "Score:", test_rewards)

                with summary_writer.as_default():
                    tf.summary.scalar("test_rewards", test_rewards, step=learner_cycles)
                    tf.summary.scalar("buffer_size", len(replay), step=learner_cycles)

            if learner_cycles % 500 == 0:
                print("Model Saved")
                learner.save.remote("checkpoints/qnet")

    ray.shutdown()

    cycles, scores = zip(*history)
    plt.plot(cycles, scores)
    plt.ylabel("test_score(epsilon=0.01)")
    plt.savefig("log/history.png")


def test_play(env_name):

    ray.init()
    tester = Tester.remote(env_name=env_name, n_frames=4)
    res = tester.test_with_video.remote(
            checkpoint_path="checkpoints/qnet", monitor_dir="mp4", epsilon=0.02)
    rewards = ray.get(res)
    print(rewards)


if __name__ == '__main__':
    env_name = "BreakoutDeterministic-v4"
    #env_name = "MsPacmanDeterministic-v4"
    main(env_name=env_name, num_actors=20)
    test_play(env_name)
