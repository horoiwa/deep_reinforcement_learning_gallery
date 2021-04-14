import random

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


class FQFNetwork(tf.keras.Model):

    def __init__(self, action_space, num_quantiles=32,
                 quantile_embedding_dim=64, state_embedding_dim=3136):
        """
        Note:
            num_quantiles (int): 提案する％分位の数. Defaults to 32.
            quantile_embedding_dim (int): quantile_fractionの埋め込み表現次元数. Defaults to 64.
            state_embedding_dim (int): state埋め込み表現の次元数. validationにのみ使用. Defaults to 3136.
        """

        super(FQFNetwork, self).__init__()

        self.action_space = action_space

        self.num_quantiles = num_quantiles

        self.state_embedding_dim = state_embedding_dim

        self.quantile_embedding_dim = quantile_embedding_dim

        self.state_embedding_layer = StateEmbeddingNetwork()

        self.fraction_proposal_layer = FractionProposalNetwork(
            num_quantiles, self.quantile_embedding_dim, self.state_embedding_dim)

        self.quantile_function = QuantileFunctionNetwork(
            self.action_space, self.state_embedding_dim, self.quantile_embedding_dim)

    def call(self, state):
        """
            Note:

            taus: τ = 0, ..., 1. (length==num_quantiles+1)
            taus_hat: τ^_{i} = (τ_{i} + τ_{i+1}) / 2 , (length==num_quantiles)
        """

        state_embedded = self.state_embedding_layer(state)

        assert state_embedded.shape[1] == self.state_embedding_dim

        _, taus_hat, taus_hat_probs = self.propose_fractions(state_embedded)

        quantiles_tau_hat = self.quantile_function(
            state_embedded, taus_hat)

        return (taus_hat, taus_hat_probs, quantiles_tau_hat)

    @tf.function
    def propose_fractions(self, state_embedded):

        taus = self.fraction_proposal_layer(state_embedded)
        taus_hat = (taus[:, 1:] + taus[:, :-1]) / 2.
        taus_hat_probs = taus[:, 1:] - taus[:, :-1]

        return taus, taus_hat, taus_hat_probs

    def sample_action(self, x, epsilon):

        if random.random() > epsilon:
            selected_actions, _ = self.greedy_action(x)
            selected_action = selected_actions[0][0].numpy()
        else:
            selected_action = np.random.choice(self.action_space)

        return selected_action

    def greedy_action(self, state):

        _, taus_hat_probs, quantiles_tau_hat = self(state)

        taus_hat_probs = tf.repeat(
            tf.expand_dims(taus_hat_probs, axis=1),
            self.action_space, axis=1)

        weighted_quantiles_tau_hat = quantiles_tau_hat * taus_hat_probs
        q_means = tf.reduce_mean(
            weighted_quantiles_tau_hat, axis=2, keepdims=True)
        selected_actions = tf.argmax(q_means, axis=1)

        return selected_actions, quantiles_tau_hat

    def greedy_action_on_given_taus(self, state, taus_hat, taus_hat_probs):

        assert taus_hat.shape[1] == taus_hat_probs.shape[1]

        state_embedded = self.state_embedding_layer(state)

        quantiles_tau_hat = self.quantile_function(state_embedded, taus_hat)

        taus_hat_probs = tf.repeat(
            tf.expand_dims(taus_hat_probs, axis=1),
            self.action_space, axis=1)

        weighted_quantiles_tau_hat = quantiles_tau_hat * taus_hat_probs
        q_means = tf.reduce_mean(
            weighted_quantiles_tau_hat, axis=2, keepdims=True)
        selected_actions = tf.argmax(q_means, axis=1)

        return selected_actions, quantiles_tau_hat

    def compute_gradients_tau(self, states, actions):

        state_embedded = self.state_embedding_layer(states)
        quantiles_all = self.fraction_proposal_layer(state_embedded)

        quantiles = quantiles_all[:, 1:-1]
        quantile_values = self.quantile_function_layer(state_embedded, quantiles)

        quantiles_hat = (quantiles_all[:, 1:] + quantiles_all[:, :-1]) / 2.
        quantile_hat_values = self.quantile_function_layer(state_embedded, quantiles_hat)

        gradients_tau = 2 * quantile_values - quantile_hat_values[:, :, 1:] - quantile_hat_values[:, :, :-1]

        return gradients_tau


class StateEmbeddingNetwork(tf.keras.layers.Layer):

    def __init__(self):

        super(StateEmbeddingNetwork, self).__init__()

        self.conv1 = kl.Conv2D(32, 8, strides=4, activation="relu",
                               kernel_initializer="he_normal")
        self.conv2 = kl.Conv2D(64, 4, strides=2, activation="relu",
                               kernel_initializer="he_normal")
        self.conv3 = kl.Conv2D(64, 3, strides=1, activation="relu",
                               kernel_initializer="he_normal")

        self.flatten1 = kl.Flatten()

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        out = self.flatten1(x)
        return out


class FractionProposalNetwork(tf.keras.layers.Layer):

    def __init__(self, num_quantiles, quantile_embedding_dim, state_embedding_dim):

        super(FractionProposalNetwork, self).__init__()

        self.num_quantiles = num_quantiles

        self.quantile_embedding_dim = quantile_embedding_dim

        self.state_embedding_dim = state_embedding_dim

        self.dense_1 = kl.Dense(num_quantiles, activation=None,
                                kernel_initializer="he_normal")

    def call(self, state_embedded):

        batch_size = state_embedded.shape[0]

        logits = self.dense_1(state_embedded)
        taus_excluding_zero = tf.cumsum(
            tf.nn.softmax(logits, axis=1), axis=1)
        tau_zero = tf.zeros([batch_size, 1], dtype=tf.float32)
        taus = tf.concat([tau_zero, taus_excluding_zero], axis=-1)

        return taus


class QuantileFunctionNetwork(tf.keras.layers.Layer):

    def __init__(self, action_space,
                 state_embedding_dim, quantile_embedding_dim):

        super(QuantileFunctionNetwork, self).__init__()

        self.action_space = action_space

        self.state_embedding_dim = state_embedding_dim

        self.quantile_embedding_dim = quantile_embedding_dim

        self.dense1 = kl.Dense(self.state_embedding_dim, activation="relu",
                               kernel_initializer="he_normal")

        self.dense2 = kl.Dense(512, activation="relu",
                               kernel_initializer="he_normal")

        self.out = kl.Dense(self.action_space, activation=None,
                            kernel_initializer="he_normal")

    def call(self, state_embedded, quantiles):

        batch_size, N = state_embedded.shape[0], quantiles.shape[1]

        """ Repeat for the number of quantiles
        """
        state_embedded = tf.repeat(
            tf.expand_dims(state_embedded, axis=1), N, axis=1)
        state_embedded = tf.reshape(
            state_embedded,
            [batch_size * N, self.state_embedding_dim])

        """ cosine embedding of τ
        """
        quantiles_embedded = tf.expand_dims(quantiles, axis=2)
        quantiles_embedded = tf.repeat(
            quantiles_embedded, self.quantile_embedding_dim, axis=2)

        pis = tf.range(1, self.quantile_embedding_dim+1, 1, dtype=tf.float32) * np.pi
        quantiles_embedded = tf.cos(quantiles_embedded * pis)

        #: DenseLayerに適用するための不自然なreshape
        quantiles_embedded = tf.reshape(
            quantiles_embedded, [batch_size * N, self.quantile_embedding_dim])
        quantiles_embedded = self.dense1(quantiles_embedded)

        assert quantiles_embedded.shape.as_list() == [batch_size * N, self.state_embedding_dim]
        assert state_embedded.shape.as_list() == [batch_size * N, self.state_embedding_dim]

        x = state_embedded * quantiles_embedded
        x = self.dense2(x)
        quantile_values = self.out(x)

        quantile_values = tf.reshape(
            quantile_values, (batch_size, N, self.action_space))
        quantile_values = tf.transpose(quantile_values, [0, 2, 1])

        return quantile_values


if __name__ == "__main__":
    import gym
    import numpy as np
    import collections
    import util

    env = gym.make("BreakoutDeterministic-v4")
    frames = collections.deque(maxlen=4)
    frame = util.frame_preprocess(env.reset())
    for _ in range(4):
        frames.append(frame)

    state1 = np.stack(frames, axis=2)
    state2 = np.stack(frames, axis=2) * 0.77

    state = np.stack([state1, state2], axis=0)
    state = tf.convert_to_tensor(state, dtype=tf.float32)

    action = tf.convert_to_tensor([[1], [1]], dtype=tf.float32)

    fqf_net = FQFNetwork(action_space=4, num_quantiles=10, quantile_embedding_dim=12)

    quantile_hat_values, quantile_hat_probs, quantiles_all = fqf_net(state)
    fqf_net.sample_actions(state)
    fqf_net.compute_tau_gradients(state, action)

    #selected_actions, quantile_qvalues = qnet.sample_actions(state)
    #print(selected_actions.numpy().flatten())
