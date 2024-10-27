import time

import gym
import ray
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from segment_tree import SumTree
from model import QNetwork


@ray.remote
class Actor:

    def __init__(self, pid, epsilon, gamma, env_name):

        self.pid = pid
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.action_space = self.env.action_space.n

        self.q_network = QNetwork(self.action_space)
        self.epsilon = epsilon
        self.gamma = gamma
        self.buffer = []

        self.state = self.env.reset()
        self.define_network()

        self.episode_rewards = 0

    def define_network(self):
        tf.config.set_visible_devices([], 'GPU')
        env = gym.make(self.env_name)
        state = env.reset()
        self.q_network(np.atleast_2d(state))

    def rollout(self, current_weights):
        #: グローバルQ関数と重みを同期
        self.q_network.set_weights(current_weights)

        #: rollout 100step
        for _ in range(100):
            state = self.state
            action = self.q_network.sample_action(state, self.epsilon)
            next_state, reward, done, _ = self.env.step(action)
            self.episode_rewards += reward
            transition = (state, action, reward, next_state, done)
            self.buffer.append(transition)

            if done:
                print(self.episode_rewards)
                self.state = self.env.reset()
                self.episode_rewards = 0
            else:
                self.state = next_state

        #: 初期優先度の計算
        states = np.vstack([transition[0] for transition in self.buffer])
        actions = np.array([transition[1] for trainsition in self.buffer])
        rewards = np.vstack([transition[2] for trainsition in self.buffer])
        next_states = np.vstack([transition[3] for transition in self.buffer])
        dones = np.vstack([transition[4] for transition in self.buffer])

        next_qvalues = self.q_network(next_states)
        next_actions = tf.cast(tf.argmax(next_qvalues, axis=1), tf.int32)
        next_actions_onehot = tf.one_hot(next_actions, self.action_space)
        next_maxQ = tf.reduce_sum(
            next_qvalues * next_actions_onehot, axis=1, keepdims=True)

        TQ = rewards + self.gamma * (1 - dones) * next_maxQ

        qvalues = self.q_network(states)
        actions_onehot = tf.one_hot(actions, self.action_space)
        Q = tf.reduce_sum(qvalues * actions_onehot, axis=1, keepdims=True)

        td_errors = (TQ - Q).numpy().flatten()
        transitions = self.buffer
        self.buffer = []

        return td_errors, transitions, self.pid


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


@ray.remote(num_cpus=1, num_gpus=1)
class Learner:

    def __init__(self, gamma, env_name):
        self.env_name = env_name
        self.action_space = gym.make(self.env_name).action_space.n
        self.q_network = QNetwork(self.action_space)
        self.target_q_network = QNetwork(self.action_space)
        self.gamma = gamma
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)

    def define_network(self):
        env = gym.make(self.env_name)
        state = env.reset()
        self.q_network(np.atleast_2d(state))
        self.target_q_network(np.atleast_2d(state))
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
        self.q_network = QNetwork(self.action_space)
        self.define_network()

    def define_network(self):
        env = gym.make(self.env_name)
        state = env.reset()
        self.q_network(np.atleast_2d(state))

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


def main(num_actors, gamma=0.98, env_name="CartPole-v0"):

    s = time.time()

    ray.init()
    history = []

    epsilons = np.linspace(0.01, 0.5, num_actors)
    actors = [Actor.remote(pid=i, env_name=env_name, epsilon=epsilons[i], gamma=gamma)
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
    main(num_actors=8)
