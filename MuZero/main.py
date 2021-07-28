import pickle
import shutil
import time
from pathlib import Path

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import categorical_crossentropy
import ray
import lz4.frame as lz4f

import util
from buffer import PrioritizedReplay
from actor import Actor
from networks import DynamicsNetwork, PVNetwork, RepresentationNetwork


@ray.remote(num_cpus=1, num_gpus=1)
class Learner:

    def __init__(self, env_id, unroll_steps=5, td_steps=5, n_frames=8,
                 V_min=-30, V_max=30, gamma=0.998, target_update_period=1600):

        self.env_id = env_id

        self.unroll_steps = unroll_steps

        self.td_steps = td_steps

        self.n_frames = n_frames

        self.V_min, self.V_max = V_min, V_max

        self.n_supports = V_max - V_min + 1

        self.supports = tf.range(V_min, V_max+1, dtype=tf.float32)

        self.gamma = gamma

        self.target_update_period = target_update_period

        self.action_space = gym.make(env_id).action_space.n

        self.repr_network = RepresentationNetwork(
            action_space=self.action_space)

        self.pv_network = PVNetwork(action_space=self.action_space,
                                    V_min=V_min, V_max=V_max)

        self.target_repr_network = RepresentationNetwork(
            action_space=self.action_space)

        self.target_pv_network = PVNetwork(action_space=self.action_space,
                                           V_min=V_min, V_max=V_max)

        self.dynamics_network = DynamicsNetwork(action_space=self.action_space,
                                                V_min=V_min, V_max=V_max)

        self.preprocess_func = util.get_preprocess_func(self.env_id)

        self.optimizer = tf.keras.optimizers.Adam(lr=0.0004)

        self.update_count = 0

        self.setup()

    def setup(self):
        """ initialize network parameter """

        env = gym.make(self.env_id)
        frame = self.preprocess_func(env.reset())

        frame_history = [frame] * self.n_frames
        action_history = [0] * self.n_frames

        hidden_state, obs = self.repr_network.predict(frame_history, action_history)
        policy, value = self.pv_network.predict(hidden_state)
        next_state, reward = self.dynamics_network.predict(hidden_state, action=0)

        hidden_state, obs = self.target_repr_network.predict(frame_history, action_history)
        policy, value = self.target_pv_network.predict(hidden_state)

        self.target_repr_network.set_weights(self.repr_network.get_weights())
        self.target_pv_network.set_weights(self.pv_network.get_weights())

    def save(self):
        self.repr_network.save_weights("checkpoints/repr_net")
        self.pv_network.save_weights("checkpoints/pv_net")
        self.dynamics_network.save_weights("checkpoints/dynamics_net")

    def load(self, repr_path, pv_path, dynamics_path):

        self.repr_network.load_weights(repr_path)
        self.pv_network.load_weights(pv_path)
        self.dynamics_network.load_weights(dynamics_path)

        self.target_repr_network.set_weights(self.repr_network.get_weights())
        self.target_pv_network.set_weights(self.pv_network.get_weights())

    def get_weights(self):

        weights = (self.repr_network.get_weights(),
                   self.pv_network.get_weights(),
                   self.dynamics_network.get_weights())

        return weights

    def update_network(self, minibatchs):

        indices_all, priorities_all, losses = [], [], []

        with util.Timer("Learner update:"):

            for (indices, weights, samples) in minibatchs:

                samples = [pickle.loads(lz4f.decompress(s)) for s in samples]

                priorities, loss_info = self.update(weights, samples)

                indices_all += indices

                priorities_all += priorities

                losses.append(loss_info)

        current_weights = self.get_weights()

        total_loss = sum([l[0] for l in losses]) / len(losses)
        policy_loss = sum([l[1] for l in losses]) / len(losses)
        value_loss = sum([l[2] for l in losses]) / len(losses)
        reward_loss = sum([l[3] for l in losses]) / len(losses)

        losses_mean = (total_loss, policy_loss, value_loss, reward_loss)

        return (current_weights, indices_all, priorities_all, losses_mean)

    def update(self, weights, samples):

        #: (batchsize, ...)
        observations = tf.concat([s.observation for s in samples], axis=0)

        #: (unroll_steps, batchsize)
        actions = tf.stack([s.actions for s in samples], axis=1)

        #: (unroll_steps, batch_size, action_space)
        target_policies = tf.stack([s.target_policies for s in samples], axis=1)

        #: (unroll_steps, batch_size, 1)
        target_rewards_scalar = tf.expand_dims(
            tf.stack([s.target_rewards for s in samples], axis=1), axis=2)

        #: (unroll_steps, batch_size, n_supports)
        target_rewards = self.scalar_to_supports(target_rewards_scalar)

        #: (unroll_steps, batch_size, 1)
        nstep_returns = tf.expand_dims(
            tf.stack([s.nstep_returns for s in samples], axis=1), axis=2)

        #: (unroll_steps, batch_size, 1)
        dones = tf.expand_dims(
            tf.cast(tf.stack([s.dones for s in samples], axis=1), tf.float32),
            axis=2)

        residual_values = []

        for i in range(self.unroll_steps):
            #: (batch_size, ...)
            last_observations = tf.concat(
                [s.last_observations[i] for s in samples], axis=0)
            #: (batch_size, 1)
            _, values = self.target_pv_network.predict(
                self.target_repr_network(last_observations))

            residual_values.append(values)

        #: (unroll_steps, batch_size, 1)
        residual_values = tf.stack(residual_values, axis=0)

        #: (unroll_steps, batch_size, 1)
        target_values_scalar = util.value_rescaling(
            nstep_returns + (1. - dones) * (self.gamma ** self.td_steps) * residual_values)

        #: (unroll_steps, batch_size, n_supports)
        target_values = self.scalar_to_supports(target_values_scalar)

        scale = 1.0 / self.unroll_steps

        with tf.GradientTape() as tape:

            policy_loss, value_loss, reward_loss = 0., 0., 0.

            hidden_states = self.repr_network(observations, training=True)

            for t in range(self.unroll_steps):
                policy_preds, value_preds = self.pv_network(hidden_states, training=True)

                hidden_states, reward_preds = self.dynamics_network(
                    hidden_states, actions[t], training=True)

                #: cross_entoropy
                ploss = categorical_crossentropy(target_policies[t], policy_preds)

                vloss = categorical_crossentropy(target_values[t], value_preds)

                rloss = categorical_crossentropy(target_rewards[t], reward_preds)

                policy_loss += scale * ploss + (1. - scale) * tf.stop_gradient(ploss)

                value_loss += scale * vloss + (1. - scale) * tf.stop_gradient(vloss)

                reward_loss += scale * rloss + (1. - scale) * tf.stop_gradient(rloss)

                hidden_states = 0.5 * hidden_states + 0.5 * tf.stop_gradient(hidden_states)

                #: To compute priority
                if t == 0:
                    vpred_0 = value_preds

            policy_loss = tf.reduce_mean(policy_loss)
            value_loss = tf.reduce_mean(value_loss)
            reward_loss = tf.reduce_mean(reward_loss)

            loss = policy_loss + value_loss + reward_loss

        #: Gather trainable variables
        variables = [self.repr_network.trainable_variables,
                     self.pv_network.trainable_variables,
                     self.dynamics_network.trainable_variables]

        grads = tape.gradient(loss, variables)
        for i in range(len(variables)):
            grads_i, _ = tf.clip_by_global_norm(grads[i], 40.0)
            self.optimizer.apply_gradients(zip(grads_i, variables[i]))

        #: Compute prioritit
        vpreds = tf.reduce_sum(self.supports * vpred_0, axis=1).numpy()
        vtargets = target_values_scalar[0].numpy().flatten()

        priorities = [abs(vtarg - vpred) for vtarg, vpred
                      in zip(vtargets, vpreds)]

        if self.update_count % self.target_update_period == 0:
            print("==== Target Update ====")
            self.target_repr_network.set_weights(self.repr_network.get_weights())
            self.target_pv_network.set_weights(self.pv_network.get_weights())

        if self.update_count % 10000 == 0:
            print("==== Save weights ====")
            self.save()

        self.update_count += 1

        return priorities, (loss, policy_loss, value_loss, reward_loss)

    def scalar_to_supports(self, X):
        """Convert scalar reward/value to categorical distribution

        Args:
            X: shape (unroll_steps, batchsize, 1)
        Returns:
            X_dist: shape (unroll_steps, batchsize, n_supports)
        """
        timesteps, batchsize = X.shape[0], X.shape[1]
        X_dist = np.zeros((timesteps, batchsize, self.n_supports))

        for t in range(timesteps):

            x = X[t].numpy().flatten()

            x_ceil = np.ceil(x).astype(np.int8)
            x_floor = np.floor(x).astype(np.int8)

            ceil_indices = x_ceil - self.V_min
            floor_indices = x_floor - self.V_min

            ceil_probs = x - x_floor
            floor_probs = 1.0 - ceil_probs

            X_dist[t, np.arange(batchsize), floor_indices] += floor_probs
            X_dist[t, np.arange(batchsize), ceil_indices] += ceil_probs

        return tf.convert_to_tensor(X_dist, dtype=tf.float32)


def main(env_id="BreakoutDeterministic-v4",
         num_actors=20,
         n_episodes=30000, unroll_steps=3,
         n_frames=4, gamma=0.997, td_steps=5,
         V_min=-30, V_max=30, dirichlet_alpha=0.25,
         buffer_size=2**16, num_mcts_simulations=20,
         batchsize=32, num_minibatchs=64, resume_step=None):
    """
    Args:
        n_frames (int): num of stacked RGB frames. Defaults to 8. (original 32)
        gamma (float): discount factor. Defaults to 0.997.
        V_min, V_max (int):
            assumed range of rescaled rewards,
            -30 ~ 30 corresponds to roughly score -1000 ~ 1000
            (original -300 ~ 300)
        resume_step (int): num steps to resume learning from

    Changes from original paper:
        - Use Greyscale image instead of RGB frame
        - Reduce the number of residual blocks for compuational efficiency.
    """

    ray.init(local_mode=False)

    learner = Learner.remote(env_id=env_id, unroll_steps=unroll_steps,
                             td_steps=td_steps, n_frames=n_frames,
                             V_min=V_min, V_max=V_max, gamma=gamma)

    logdir = Path(__file__).parent / "log"

    if resume_step:
        learner.load.remote("checkpoints/repr_net",
                            "checkpoints/pv_net",
                            "checkpoints/dynamics_net")
    else:
        if logdir.exists():
            shutil.rmtree(logdir)

    summary_writer = tf.summary.create_file_writer(str(logdir))

    n = 0 if resume_step is None else resume_step

    current_weights = ray.put(ray.get(learner.get_weights.remote()))

    buffer = PrioritizedReplay(capacity=buffer_size)

    actors = [Actor.remote(pid=pid, env_id=env_id, n_frames=n_frames,
                           unroll_steps=unroll_steps, td_steps=td_steps,
                           num_mcts_simulations=num_mcts_simulations,
                           V_min=V_min, V_max=V_max, gamma=gamma,
                           dirichlet_alpha=0.25)
              for pid in range(num_actors)]

    tester = Actor.remote(pid=0, env_id=env_id, n_frames=n_frames,
                          unroll_steps=unroll_steps, td_steps=td_steps,
                          num_mcts_simulations=num_mcts_simulations,
                          V_min=V_min, V_max=V_max, gamma=gamma,
                          dirichlet_alpha=0.25)

    wip_actors = [actor.sync_weights_and_rollout.remote(current_weights, T=1.0)
                  for actor in actors]

    for _ in range(20):
        finished_actor, wip_actors = ray.wait(wip_actors, num_returns=1)
        pid, samples, priorities = ray.get(finished_actor[0])
        buffer.add_samples(priorities, samples)
        wip_actors.extend(
            [actors[pid].sync_weights_and_rollout.remote(current_weights, T=1.0)])
        n += 1

    minibatchs = [buffer.sample_minibatch(batchsize=batchsize)
                  for _ in range(num_minibatchs)]

    wip_learner = learner.update_network.remote(minibatchs)

    wip_tester = tester.testplay.remote(current_weights)

    minibatchs = [buffer.sample_minibatch(batchsize=batchsize)
                  for _ in range(num_minibatchs)]

    t = time.time()

    actor_count = 0

    while n <= n_episodes:

        finished_actors, wip_actors = ray.wait(wip_actors, timeout=0)

        for i in range(len(finished_actors)):

            pid, samples, priorities = ray.get(finished_actor[i])

            buffer.add_samples(priorities, samples)

            T = 1.0 if n < 1000 else 0.5 if n < 3000 else 0.25

            wip_actors.extend(
                [actors[pid].sync_weights_and_rollout.remote(current_weights, T=T)])

            n += 1

            actor_count += 1

        finished_learner, _ = ray.wait([wip_learner], timeout=0)

        if finished_learner:

            current_weights, indices, priorities, info = ray.get(finished_learner[0])

            current_weights = ray.put(current_weights)

            wip_learner = learner.update_network.remote(minibatchs)

            buffer.update_priority(indices, priorities)

            with util.Timer("Make minibatchs:"):
                minibatchs = [buffer.sample_minibatch(batchsize=batchsize)
                              for _ in range(num_minibatchs)]

            with summary_writer.as_default():
                tf.summary.scalar("Buffer", len(buffer), step=n)
                tf.summary.scalar("loss", info[0], step=n)
                tf.summary.scalar("policy_loss", info[1], step=n)
                tf.summary.scalar("value_loss", info[2], step=n)
                tf.summary.scalar("reward_loss", info[3], step=n)
                tf.summary.scalar("time", time.time() - t, step=n)
                tf.summary.scalar("actor_count", actor_count, step=n)

            t = time.time()
            actor_count = 0

        if n % 50 == 0:

            finished_tester, _ = ray.wait([wip_tester], timeout=0)

            if finished_tester:

                score, step = ray.get(finished_tester[0])

                wip_tester = tester.testplay.remote(current_weights)

                with summary_writer.as_default():
                    tf.summary.scalar("Test Score", score, step=n)
                    tf.summary.scalar("Test Step", step, step=n)


if __name__ == '__main__':
    main(num_actors=18, resume=False)
