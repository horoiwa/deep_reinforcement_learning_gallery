import time

import gym
import ray
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from segment_tree import SumTree
from model import RecurrentQNetwork
from buffer import EpisodeBuffer


@ray.remote
class Actor:

    def __init__(self, pid, env_name, epsilon, gamma,
                 burnin_length, unroll_length):

        self.pid = pid
        self.env_name = env_name
        self.action_space = gym.make(env_name).action_space.n

        self.q_network = RecurrentQNetwork(self.action_space)
        self.epsilon = epsilon
        self.gamma = gamma

        self.burnin_len = burnin_length
        self.unroll_len = unroll_length

        self.define_network()

    def define_network(self):
        tf.config.set_visible_devices([], 'GPU')
        env = gym.make(self.env_name)
        state = env.reset()

        c, h = self.q_network.lstm.get_initial_state(batch_size=1, dtype=tf.float32)
        self.q_network(np.atleast_2d(state), states=[c, h])

    def rollout(self, current_weights):

        #: グローバルQ-Networkと重みを同期
        self.q_network.set_weights(current_weights)

        env = gym.make(self.env_name)
        episode_buffer = EpisodeBuffer(burnin_length=self.burnin_len,
                                       unroll_length=self.unroll_len)

        state = env.reset().astype(np.float32)
        c, h = self.q_network.lstm.get_initial_state(
            batch_size=1, dtype=tf.float32)
        done = False

        episode_rewards = 0

        while not done:
            action, (next_c, next_h) = self.q_network.sample_action(state, c, h, self.epsilon)
            next_state, reward, done, _ = env.step(action)
            episode_rewards += reward

            transition = (state, action, reward, next_state, done, c, h)
            episode_buffer.put(transition)

            state, c, h = next_state, next_c, next_h

        print(episode_rewards)

        segments = episode_buffer.pull()

        """ Compute initial priority
        """
        states = np.stack([seg.states for seg in segments], axis=1)    # (timestep, batch_size, obs_dim)
        actions = np.stack([seg.actions for seg in segments], axis=1)  # (timestep, batch_size)
        rewards = np.stack([seg.rewards for seg in segments], axis=1)  # (timestep, batch_size)
        dones = np.stack([seg.dones for seg in segments], axis=1)      # (timestep, batch_size)
        last_state = np.stack([seg.last_state for seg in segments])   # (batch_size, obs_dim)

        c = tf.convert_to_tensor(
            np.vstack([seg.c_init for seg in segments]), dtype=tf.float32)  # (batch_size, lstm_out_dim)
        h = tf.convert_to_tensor(
            np.vstack([seg.h_init for seg in segments]), dtype=tf.float32)  # (batch_size, lstm_out_dim)

        #: burn-in with stored-state
        for t in range(self.burnin_len):
            _, (c, h) = self.q_network(states[t], states=[c, h])

        qvalues = []
        for t in range(self.burnin_len, self.burnin_len+self.unroll_len):
            q, (c, h) = self.q_network(states[t], states=[c, h])
            qvalues.append(q)
        qvalues = tf.stack(qvalues)  #: (unroll_len, batch_size, action_space)
        actions_onehot = tf.one_hot(actions, self.action_space)
        Q = tf.reduce_sum(qvalues * actions_onehot, axis=1, keepdims=True)

        #: compute qvlalue of last next-state in segment
        remaining_qvalue, _ = self.q_network(last_state, states=[c, h])
        remaining_qvalue = tf.expand_dims(remaining_qvalue, axis=0)  # (1, batch_size, action_space)

        next_qvalues = tf.concat([qvalues[1:], remaining_qvalue], axis=0)  # (unroll_len, batch_size, action_space)
        next_actions = tf.argmax(next_qvalues, axis=2)  # (unroll_len, batch_size)
        next_actions_onehot = tf.one_hot(next_actions, self.action_space)  # (unroll_len, batch_size, action_space)

        import pdb; pdb.set_trace()

        TQ = rewards + self.gamma * (1 - dones) * next_maxQ
        td_errors = (TQ - Q).numpy().flatten()

        return td_errors, segments, self.pid


class Replay:

    def __init__(self, buffer_size):

        self.buffer_size = buffer_size
        self.priorities = SumTree(capacity=self.buffer_size)
        self.buffer = [None] * self.buffer_size

        self.alpha = 0.6
        self.beta = 0.4

        self.count = 0
        self.is_full = False

    def add(self, td_errors, transitions):
        assert len(td_errors) == len(transitions)
        priorities = (np.abs(td_errors) + 0.001) ** self.alpha
        for priority, transition in zip(priorities, transitions):
            self.priorities[self.count] = priority
            self.buffer[self.count] = transition
            self.count += 1
            if self.count == self.buffer_size:
                self.count = 0
                self.is_full = True

    def update_priority(self, sampled_indices, td_errors):
        assert len(sampled_indices) == len(td_errors)
        for idx, td_error in zip(sampled_indices, td_errors):
            priority = (abs(td_error) + 0.001) ** self.alpha
            self.priorities[idx] = priority**self.alpha

    def sample_minibatch(self, batch_size):

        sampled_indices = [self.priorities.sample() for _ in range(batch_size)]

        #: compute prioritized experience replay weights
        weights = []
        current_size = len(self.buffer) if self.is_full else self.count
        for idx in sampled_indices:
            prob = self.priorities[idx] / self.priorities.sum()
            weight = (prob * current_size)**(-self.beta)
            weights.append(weight)
        weights = np.array(weights) / max(weights)

        experiences = [self.buffer[idx] for idx in sampled_indices]

        return sampled_indices, weights, experiences


class EpisodicPrioritizedReplayBuffer:

    def __init__(self, buffer_size=2**12):

        self.buffer_size = buffer_size
        self.priorities = SumTree(capacity=self.buffer_size)
        self.buffer = [None] * self.buffer_size

        self.alpha = 0.6
        self.beta = 0.4

        self.count = 0
        self.is_full = False

    def add(self, td_errors, transitions):
        assert len(td_errors) == len(transitions)

        priorities = (np.abs(td_errors) + 0.001) ** self.alpha

        for priority, transition in zip(priorities, transitions):
            self.priorities[self.count] = priority
            self.buffer[self.count] = transition
            self.count += 1
            if self.count == self.buffer_size:
                self.count = 0
                self.is_full = True

    def update_priority(self, sampled_indices, td_errors):
        assert len(sampled_indices) == len(td_errors)
        for idx, td_error in zip(sampled_indices, td_errors):
            priority = (abs(td_error) + 0.001) ** self.alpha
            self.priorities[idx] = priority**self.alpha

    def sample_minibatch(self, batch_size):

        sampled_indices = [self.priorities.sample() for _ in range(batch_size)]

        #: compute prioritized experience replay weights
        weights = []
        current_size = len(self.buffer) if self.is_full else self.count
        for idx in sampled_indices:
            prob = self.priorities[idx] / self.priorities.sum()
            weight = (prob * current_size)**(-self.beta)
            weights.append(weight)
        weights = np.array(weights) / max(weights)

        experiences = [self.buffer[idx] for idx in sampled_indices]

        return sampled_indices, weights, experiences


@ray.remote(num_cpus=1, num_gpus=1)
class Learner:

    def __init__(self, gamma, env_name):
        self.env_name = env_name
        self.action_space = gym.make(self.env_name).action_space.n
        self.q_network = RecurrentQNetwork(self.action_space)
        self.target_q_network = RecurrentQNetwork(self.action_space)
        self.gamma = gamma
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)

    def define_network(self):
        env = gym.make(self.env_name)
        state = env.reset()
        c, h = self.q_network.lstm.get_initial_state(batch_size=1, dtype=tf.float32)

        self.q_network(np.atleast_2d(state), states=[c, h])
        self.target_q_network(np.atleast_2d(state), states=[c, h])
        self.target_q_network.set_weights(self.q_network.get_weights())
        current_weights = self.q_network.get_weights()
        return current_weights

    def update_network(self, minibatchs):

        indices_all = []
        td_errors_all = []

        for (indices, weights, transitions) in minibatchs:

            states, actions, rewards, next_states, dones = zip(*transitions)

            states = np.vstack(states)
            actions = np.array(actions)
            rewards = np.vstack(rewards)
            next_states = np.vstack(next_states)
            dones = np.vstack(dones)

            next_qvalues = self.q_network(next_states)
            next_actions = tf.cast(tf.argmax(next_qvalues, axis=1), tf.int32)
            next_actions_onehot = tf.one_hot(next_actions, self.action_space)
            next_maxQ = tf.reduce_sum(
                next_qvalues * next_actions_onehot, axis=1, keepdims=True)
            TQ = rewards + self.gamma * (1 - dones) * next_maxQ

            with tf.GradientTape() as tape:
                qvalues = self.q_network(states)
                actions_onehot = tf.one_hot(actions, self.action_space)
                Q = tf.reduce_sum(qvalues * actions_onehot, axis=1, keepdims=True)
                td_errors = tf.square(TQ - Q)
                loss = tf.reduce_mean(weights * td_errors)

            grads = tape.gradient(loss, self.q_network.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 40.0)
            self.optimizer.apply_gradients(
                zip(grads, self.q_network.trainable_variables))

            indices_all += indices
            td_errors_all += td_errors.numpy().flatten().tolist()

        current_weights = self.q_network.get_weights()
        return current_weights, indices_all, td_errors_all


@ray.remote
class Tester:

    def __init__(self, env_name):

        self.env_name = env_name
        self.action_space = gym.make(self.env_name).action_space.n
        self.q_network = RecurrentQNetwork(self.action_space)
        self.define_network()

    def define_network(self):
        env = gym.make(self.env_name)
        state = env.reset()
        c, h = self.q_network.lstm.get_initial_state(batch_size=1, dtype=tf.float32)
        self.q_network(np.atleast_2d(state), states=[c, h])

    def test_play(self, current_weights, epsilon):

        self.q_network.set_weights(current_weights)

        env = gym.make(self.env_name)
        state = env.reset()
        episode_rewards = 0
        done = False
        while not done:
            action = self.q_network.sample_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            episode_rewards += reward
            state = next_state

        return episode_rewards


def main(num_actors, gamma=0.997, env_name="CartPole-v0",
         burnin_length=4, unroll_length=4):

    s = time.time()

    ray.init(local_mode=True)
    history = []

    epsilons = np.linspace(0.05, 0.5, num_actors) if num_actors > 1 else [0.3]
    actors = [Actor.remote(pid=i, env_name=env_name, epsilon=epsilons[i],
                           gamma=gamma, burnin_length=burnin_length,
                           unroll_length=unroll_length)
              for i in range(num_actors)]

    replay = Replay(buffer_size=2**14)

    learner = Learner.remote(env_name=env_name, gamma=gamma)
    current_weights = ray.get(learner.define_network.remote())
    current_weights = ray.put(current_weights)

    tester = Tester.remote(env_name=env_name)

    wip_actors = [actor.rollout.remote(current_weights) for actor in actors]

    for _ in range(30):
        finished, wip_actors = ray.wait(wip_actors, num_returns=1)
        td_errors, transitions, pid = ray.get(finished[0])
        replay.add(td_errors, transitions)
        wip_actors.extend([actors[pid].rollout.remote(current_weights)])

    minibatchs = [replay.sample_minibatch(batch_size=32) for _ in range(16)]
    wip_learner = learner.update_network.remote(minibatchs)
    minibatchs = [replay.sample_minibatch(batch_size=32) for _ in range(16)]
    wip_tester = tester.test_play.remote(current_weights, epsilon=0.01)

    update_cycles = 1
    actor_cycles = 0
    while update_cycles <= 200:
        actor_cycles += 1
        finished, wip_actors = ray.wait(wip_actors, num_returns=1)
        td_errors, transitions, pid = ray.get(finished[0])
        replay.add(td_errors, transitions)
        wip_actors.extend([actors[pid].rollout.remote(current_weights)])

        finished_learner, _ = ray.wait([wip_learner], timeout=0)
        if finished_learner:
            current_weights, indices, td_errors = ray.get(finished_learner[0])
            wip_learner = learner.update_network.remote(minibatchs)
            current_weights = ray.put(current_weights)

            #: 優先度の更新とminibatchの作成はlearnerよりも十分に速いという前提
            replay.update_priority(indices, td_errors)
            minibatchs = [replay.sample_minibatch(batch_size=32) for _ in range(16)]
            print(actor_cycles)
            update_cycles += 1
            actor_cycles = 0

            if update_cycles % 5 == 0:
                test_score = ray.get(wip_tester)
                print(update_cycles, test_score)
                history.append((update_cycles-5, test_score))
                wip_tester = tester.test_play.remote(current_weights, epsilon=0.01)

    wallclocktime = round(time.time() - s, 2)
    cycles, scores = zip(*history)
    plt.plot(cycles, scores)
    plt.title(f"total time: {wallclocktime} sec")
    plt.ylabel("test_score(epsilon=0.01)")
    plt.savefig("log/history.png")



if __name__ == '__main__':
    main(num_actors=1)
