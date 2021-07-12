import collections
from dataclasses import dataclass
import collections
import pickle
import shutil
from pathlib import Path

import gym
import numpy as np
import tensorflow as tf
import lz4.frame as lz4f

import util
from buffer import PrioritizedReplay
from mcts import AtariMCTS
from networks import DynamicsNetwork, PVNetwork, RepresentationNetwork


@dataclass
class Sample:

    observation: tf.Tensor
    actions: list
    target_policies: list
    target_rewards: list
    nstep_returns: list
    last_observations: list
    dones: list


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

        self.optimizer = tf.keras.optimizers.Adam(lr=0.00015)

        self.update_count = 0

    def build_network(self):
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

        weights = (self.repr_network.get_weights(),
                   self.pv_network.get_weights(),
                   self.dynamics_network.get_weights())

        return weights

    def update_network(self, minibatchs):

        indices_all, priorities_all, losses = [], [], []

        for (indices, weights, samples) in minibatchs:

            samples = [pickle.loads(lz4f.decompress(sample)) for sample in samples]

            priorities, loss_info = self.update(weights, samples)

            indices_all += indices

            priorities_all += priorities

            losses.append(loss_info)

        current_weights = self.q_network.get_weights()

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
        residual_values = tf.expand_dims(
            tf.stack(residual_values, axis=0), axis=2)

        #: (unroll_steps, batch_size, 1)
        target_values_scalar = util.value_rescaling(
            nstep_returns + (1. - dones) * (self.gamma ** self.td_steps) * residual_values)

        #: (unroll_steps, batch_size, n_supports)
        target_values = self.scalar_to_supports(target_values_scalar)

        with tf.GradientTape() as tape:

            policy_loss, value_loss, reward_loss = 0., 0., 0.

            hidden_states = self.repr_network(observations, training=True)

            for t in range(self.unroll_steps):
                policy_preds, value_preds = self.pv_network(hidden_states, training=True)

                hidden_states, reward_preds = self.dynamics_network(
                    hidden_states, actions[t], training=True)

                #: cross_entoropy
                policy_loss += (1. / self.unroll_steps) * tf.reduce_sum(
                    -target_policies[t] * tf.math.log(policy_preds + 0.00001),
                    axis=1, keepdims=True)
                value_loss += (1. / self.unroll_steps) * tf.reduce_sum(
                    -target_values[t] * tf.math.log(value_preds + 0.00001),
                    axis=1, keepdims=True)
                reward_loss += (1. / self.unroll_steps) * tf.reduce_sum(
                    -target_rewards[t] * tf.math.log(reward_preds + 0.00001),
                    axis=1, keepdims=True)

                hidden_states = 0.5 * hidden_states + 0.5 * tf.stop_gradient(hidden_states)

                #: compute priority
                if t == 0:
                    value_preds_scalar = tf.reduce_sum(
                        self.supports * value_preds, axis=1).numpy()
                    targets = target_values_scalar[0].numpy().flatten()

                    priorities = [abs(t - vpred) for t, vpred
                                  in zip(targets, value_preds_scalar)]

            policy_loss = tf.reduce_mean(policy_loss)
            value_loss = tf.reduce_mean(value_loss)
            reward_loss = tf.reduce_mean(reward_loss)

            loss = policy_loss + 0.25 * value_loss + reward_loss

        #: Gather trainable variables
        variables = [self.repr_network.trainable_variables,
                     self.pv_network.trainable_variables,
                     self.dynamics_network.trainable_variables]

        grads = tape.gradient(loss, variables)

        for i in range(len(variables)):
            self.optimizer.apply_gradients(zip(grads[i], variables[i]))

        if self.update_count % self.target_update_period:
            print("==== Target Update ====")
            self.target_repr_network.set_weights(self.repr_network.get_weights())
            self.target_pv_network.set_weights(self.pv_network.get_weights())

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


class Actor:

    def __init__(self, env_id, n_frames,
                 num_mcts_simulations, unroll_steps,
                 gamma, V_max, V_min, td_steps,
                 dirichlet_alpha):

        self.env_id = env_id

        self.unroll_steps = unroll_steps

        self.num_mcts_simulations = num_mcts_simulations

        self.action_space = gym.make(env_id).action_space.n

        self.n_frames = n_frames

        self.gamma = gamma

        self.dirichlet_alpha = dirichlet_alpha

        self.V_min, self.V_max = V_min, V_max

        self.td_steps = td_steps

        self.preprocess_func = util.get_preprocess_func(self.env_id)

        self.repr_network = RepresentationNetwork(
            action_space=self.action_space)

        self.pv_network = PVNetwork(action_space=self.action_space,
                                    V_min=V_min, V_max=V_max)

        self.dynamics_network = DynamicsNetwork(action_space=self.action_space,
                                                V_min=V_min, V_max=V_max)

        self._build_network()

    def _build_network(self):

        env = gym.make(self.env_id)
        frame = self.preprocess_func(env.reset())

        frame_history = [frame] * self.n_frames
        action_history = [0] * self.n_frames

        state, obs = self.repr_network.predict(frame_history, action_history)
        policy, value = self.pv_network(state)
        next_state, reward = self.dynamics_network.predict(state, action=0)

    def sync_weights_and_rollout(self, current_weights, T):

        #: 最新のネットワークに同期
        self.sync_weights(current_weights)

        #: 1episodeのrollout
        game_history = self.rollout(T)

        samples, priorities = self.make_samples(game_history)

        return samples, priorities

    def make_samples(self, game_history):
        """
            zは割引が必要であることに注意
            またbootstrapping return
            absorbing states.
        """

        samples = []

        observations, actions, rewards, mcts_policies, root_values = game_history

        episode_len = len(observations)

        #: States past the end of games are treated as absorbing states.
        rewards += [0] * self.td_steps

        #: NOOP
        actions += [0] * self.td_steps

        #: Uniform policy
        mcts_policies += [
            np.array([1. / self.action_space] * self.action_space, dtype=np.float32)
            for _ in range(self.td_steps)]

        #: n-step bootstrapping value
        nstep_returns = []

        priorities = []

        for idx in range(episode_len):

            bootstrap_idx = idx + self.td_steps

            nstep_return = sum([r * self.gamma ** i for i, r
                                in enumerate(rewards[idx:bootstrap_idx])])

            residual_value = root_values[bootstrap_idx] if bootstrap_idx < episode_len else 0

            #: value = r_0 + γr_1 + ... r_n + v(n+1)
            value = nstep_return + residual_value

            priorities.append(abs(value - root_values[idx]))

            nstep_returns.append(nstep_return)

        nstep_returns += [0] * self.td_steps

        for idx in range(episode_len):

            bootstrap_idx = idx + self.td_steps

            #: shape == (unroll_steps, ...)
            _actions = np.array(actions[idx:idx+self.unroll_steps])
            _nstep_returns = np.array(nstep_returns[idx:idx+self.unroll_steps], dtype=np.float32)
            target_policies = np.vstack(mcts_policies[idx:idx+self.unroll_steps])
            target_rewards = np.array(rewards[idx:idx+self.unroll_steps])
            last_observations = [observations[i] if i < episode_len else observations[-1]
                                 for i in range(bootstrap_idx, bootstrap_idx+self.unroll_steps)]
            dones = [False if i < episode_len else True
                     for i in range(bootstrap_idx, bootstrap_idx+self.unroll_steps)]

            sample = Sample(
                observation=observations[idx],
                actions=_actions,
                target_policies=target_policies,
                target_rewards=target_rewards,
                nstep_returns=_nstep_returns,
                last_observations=last_observations,
                dones=dones)

            samples.append(sample)

        #: Compress for memory efficiency
        samples = [lz4f.compress(pickle.dumps(sample)) for sample in samples]

        return samples, priorities

    def sync_weights(self, weights):

        self.repr_network.set_weights(weights[0])

        self.pv_network.set_weights(weights[1])

        self.dynamics_network.set_weights(weights[2])

    def rollout(self, T):

        env = gym.make(self.env_id)

        observations, actions, rewards, mcts_policies, root_values = [], [], [], [], []

        frame = self.preprocess_func(env.reset())

        frame_history = collections.deque(
            [frame] * self.n_frames, maxlen=self.n_frames)

        action_history = collections.deque(
            [0] * self.n_frames, maxlen=self.n_frames)

        mcts = AtariMCTS(
            action_space=self.action_space,
            pv_network=self.pv_network,
            dynamics_network=self.dynamics_network,
            gamma=self.gamma,
            dirichlet_alpha=self.dirichlet_alpha)

        done = False

        while not done:

            with tf.device("/cpu:0"):
                hidden_state, obs = self.repr_network.predict(frame_history, action_history)

            debug = True
            if not debug:

                mcts_policy, root_value = mcts.search(
                    hidden_state, self.num_mcts_simulations, T)

                action = np.random.choice(
                    range(self.action_space), p=mcts_policy)

            else:
                mcts_policy, root_value = np.array([0.1, 0.1, 0.3, 0.5], dtype=np.float32), 1.0
                action = np.random.choice(range(self.action_space), p=mcts_policy)

            frame, reward, done, info = env.step(action)

            frame_history.append(self.preprocess_func(frame))
            action_history.append(action)

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            mcts_policies.append(mcts_policy)
            root_values.append(root_value)

        game_history = (observations, actions, rewards, mcts_policies, root_values)

        return game_history


class Tester(Actor):

    def __init__(self, env_id, n_frames,
                 num_mcts_simulations, unroll_steps,
                 gamma, V_max, V_min, td_steps,
                 dirichlet_alpha):

        super().__init__(env_id, n_frames, num_mcts_simulations, unroll_steps,
                         gamma, V_max, V_min, td_steps, dirichlet_alpha)

    def play(self, current_weights):

        #: 最新のネットワークに同期
        self.sync_weights(current_weights)

        env = gym.make(self.env_id)

        frame = self.preprocess_func(env.reset())

        frame_history = collections.deque(
            [frame] * self.n_frames, maxlen=self.n_frames)

        action_history = collections.deque(
            [0] * self.n_frames, maxlen=self.n_frames)

        mcts = AtariMCTS(
            action_space=self.action_space,
            pv_network=self.pv_network,
            dynamics_network=self.dynamics_network,
            gamma=self.gamma,
            dirichlet_alpha=self.dirichlet_alpha)

        total_rewards = 0

        done = False

        while not done:

            with tf.device("/cpu:0"):
                hidden_state, obs = self.repr_network.predict(frame_history, action_history)

            mcts_policy, root_value = mcts.search(
                hidden_state, self.num_mcts_simulations, T=0.25)

            action = np.random.choice(
                range(self.action_space), p=mcts_policy)

            frame, reward, done, info = env.step(action)

            total_rewards += reward

            frame_history.append(self.preprocess_func(frame))
            action_history.append(action)


        return total_rewards


def main(env_id="BreakoutDeterministic-v4",
         n_episodes=10000, unroll_steps=5,
         n_frames=8, gamma=0.997, td_steps=5,
         V_min=-30, V_max=30, dirichlet_alpha=0.25,
         buffer_size=2**21, num_mcts_simulations=10,
         batchsize=32):
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

    logdir = Path(__file__).parent / "log"
    if logdir.exists():
        shutil.rmtree(logdir)
    summary_writer = tf.summary.create_file_writer(str(logdir))

    learner = Learner(env_id=env_id, unroll_steps=unroll_steps,
                      td_steps=td_steps, n_frames=n_frames,
                      V_min=V_min, V_max=V_max, gamma=gamma)

    current_weights = learner.build_network()

    buffer = PrioritizedReplay(capacity=buffer_size)

    actor = Actor(env_id=env_id, n_frames=n_frames, unroll_steps=unroll_steps,
                  num_mcts_simulations=num_mcts_simulations, td_steps=td_steps,
                  V_min=V_min, V_max=V_max, gamma=gamma,
                  dirichlet_alpha=0.25)

    n = 0

    for _ in range(0):
        samples, priorities = actor.sync_weights_and_rollout(current_weights, T=1.0)
        buffer.add_samples(priorities, samples)
        n += 1

    while n <= n_episodes:

        samples, priorities = actor.sync_weights_and_rollout(current_weights, T=1.0)
        buffer.add_samples(priorities, samples)
        n += 1

        minibatchs = [buffer.sample_minibatch(batchsize=batchsize)]

        current_weights, priorities, indices, info = learner.update_network(minibatchs)

        #current_weights = ray.put(current_weights)

        buffer.update_priority(indices, priorities)

        with summary_writer.as_default():
            tf.summary.scalar("loss", info[0], step=n)
            tf.summary.scalar("policy_loss", info[1], step=n)
            tf.summary.scalar("value_loss", info[2], step=n)
            tf.summary.scalar("reward_loss", info[3], step=n)
        import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()