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

        self.quatile_embedding_dim = quantile_embedding_dim

        self.state_embedding_layer = StateEmbeddingNetwork()

        self.fraction_proposal_layer = FractionProposalNetwork(
            num_quantiles, self.quatile_embedding_dim, self.state_embedding_dim)

        self.quantile_function_layer = QuantileFunctionNetwork(self.action_space, self.num_quantiles)

    def call(self, x, quantiles=None):

        batch_size = x.shape[0]

        state_embedded = self.state_embedding_layer(x)
        assert state_embedded.shape[1] == self.state_embedding_dim

        quantiles, quantiles_embedded, quantiles_hat, probs = self.fraction_proposal_layer(state_embedded, batch_size)

        quantile_values = self.quantile_function_layer(state_embedded, quantiles_embedded)

        return quantiles, quantile_values

    def sample_action(self, x, epsilon):

        if np.random.random() > epsilon:
            selected_actions, _ = self.sample_actions(x)
            selected_action = selected_actions[0][0].numpy()
        else:
            selected_action = np.random.choice(self.action_space)

        return selected_action


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

        self.dense_2 = kl.Dense(state_embedding_dim, activation="relu",
                                kernel_initializer="he_normal")

    def call(self, state_embedded, batch_size):

        logits = self.dense_1(state_embedded)
        quantiles_excluding_zero = tf.cumsum(
            tf.nn.softmax(logits, axis=1), axis=1)  # (batch_size, num_quantiles)

        tau_zero = tf.zeros([batch_size, 1], dtype=tf.float32)
        quantiles_all = tf.concat([tau_zero, quantiles_excluding_zero], axis=-1)  # (batch_size, num_quantiles+1)

        quantiles_hat = (quantiles_all[:, 1:] + quantiles_all[:, :-1]) / 2.  # (batch_size, num_quantiles)
        probs = quantiles_all[:, 1:] - quantiles_all[:, :-1]

        #: excluding τ＝0, 1 for embedding
        quantiles_embedded = quantiles_all[:, 1:-1]  # (batch_size, num_quantiles-1)
        quantiles_embedded = tf.expand_dims(quantiles_embedded, axis=2)  # (batch_size, num_quantiles-1, 1)
        quantiles_embedded = tf.repeat(
            quantiles_embedded, self.quantile_embedding_dim, axis=2)  # (batch_size, num_quantiles-1, quant_emb_dim)

        #: Compute cos(iπτ), shape==(batch_size, num_quantiles-1, quant_embed_dim)
        pis = tf.range(1, self.quantile_embedding_dim+1, 1, dtype=tf.float32) * np.pi
        quantiles_embedded = tf.cos(quantiles_embedded * pis)

        #: DenseLayerに適用するための不自然なreshape
        quantiles_embedded = tf.reshape(
            quantiles_embedded, [batch_size * (self.num_quantiles - 1), self.quantile_embedding_dim])

        #: (batch_size * (num_quantiles-1), quant_embed_dim)
        quantiles_embedded = self.dense_2(quantiles_embedded)

        import pdb; pdb.set_trace()

        return quantiles, quantile_embedding, quantiles_hat, probs


class QuantileFunctionNetwork(tf.keras.layers.Layer):

    def __init__(self, action_space, num_quantiles):
        super(QuantileFunctionNetwork, self).__init__()
        self.action_space = action_space
        self.num_quantiles = num_quantiles

    def call(self, state_embedded, quantile_embedded):
        #quantile_values = tf.reshape(out, (batch_size, self.action_space, self.N))
        return None


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

    state = np.stack(frames, axis=2)[np.newaxis, ...]
    state = tf.convert_to_tensor(state, dtype=tf.float32)

    fqf_net = FQFNetwork(action_space=4, num_quantiles=10, quantile_embedding_dim=12)

    out = fqf_net(state)

    #selected_actions, quantile_qvalues = qnet.sample_actions(state)
    #print(selected_actions.numpy().flatten())
