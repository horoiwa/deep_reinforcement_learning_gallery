import ray
import gym
import tensorflow as tf
import numpy as np

from networks import QNetwork


@ray.remote(num_cpus=1)
class Actor:

    def __init__(self, pid, env_name, epsilon, gamma=0.98):

        self.pid = pid
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.action_space = self.env.action_space.n

        self.q_network = QNetwork(self.action_space)
        self.epsilon = epsilon
        self.gamma = gamma
        self.buffer = []

        self.state = self.env.reset()
        self.setup()

        self.episode_rewards = 0

    def setup(self):
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
                #print(self.episode_rewards)
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

    def test_play(self, current_weights):

        self.q_network.set_weights(current_weights)

        env = gym.make(self.env_name)
        state = env.reset()
        episode_rewards = 0
        done = False
        while not done:
            action = self.q_network.sample_action(state, self.epsilon)
            next_state, reward, done, _ = env.step(action)
            episode_rewards += reward
            state = next_state

        return episode_rewards


@ray.remote(num_cpus=1, num_gpus=1)
class Learner:

    def __init__(self, env_name, gamma=0.98):
        self.env_name = env_name
        self.action_space = gym.make(self.env_name).action_space.n
        self.q_network = QNetwork(self.action_space)
        self.target_q_network = QNetwork(self.action_space)
        self.gamma = gamma
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)
        self.setup()

    def setup(self):
        env = gym.make(self.env_name)
        state = env.reset()
        self.q_network(np.atleast_2d(state))
        self.target_q_network(np.atleast_2d(state))
        self.target_q_network.set_weights(self.q_network.get_weights())

    def get_weights(self):
        current_weights = self.q_network.get_weights()
        return current_weights

    def update_network(self, minibatchs):

        indices_all = []
        td_errors_all = []
        losses = []

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
            losses.append(loss)

        loss_info = np.array(losses).mean()
        current_weights = self.q_network.get_weights()

        return current_weights, indices_all, td_errors_all, loss_info
