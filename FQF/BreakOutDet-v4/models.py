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

        self.quantile_function_layer = QuantileFunctionNetwork(
            self.action_space, self.num_quantiles, self.state_embedding_dim)

    def call(self, x, quantiles=None):

        batch_size = x.shape[0]

        state_embedded = self.state_embedding_layer(x, batch_size)
        assert state_embedded.shape[1] == self.state_embedding_dim

        """
            quantiles: 提案分位点からτ=0,1を除外したもの.
                       shape[1] == num_quantiles-1
            quantiles_hat: 提案分位点の中間点. τ^_{i} = (τ{i} + τ_{i+1}) / 2
                       shape[1] == num_quantiles

            E[Z(s, a)]を計算するだけならquantiles_hatだけでOK
        """
        (quantiles, quantiles_embedded,
         quantiles_hat, quantiles_hat_embedded,
         quantile_hat_probs) = self.fraction_proposal_layer(state_embedded, batch_size)

        quantile_values = self.quantile_function_layer(
            state_embedded, quantiles_embedded, batch_size)

        quantile_hat_values = self.quantile_function_layer(
            state_embedded, quantiles_hat_embedded, batch_size)

        quantile_loss = None

        return quantiles_hat, quantiles_hat_values, quantiles_loss

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

    def call(self, x, batch_size):
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
        quantile_hat_probs = quantiles_all[:, 1:] - quantiles_all[:, :-1]

        """ Embed proposed quantiles excluding τ＝0, 1
        """
        quantiles = quantiles_all[:, 1:-1]  # (batch_size, num_quantiles-1)
        quantiles_embedded = tf.expand_dims(quantiles, axis=2)  # (batch_size, num_quantiles-1, 1)
        quantiles_embedded = tf.repeat(
            quantiles_embedded, self.quantile_embedding_dim, axis=2)  # (batch_size, num_quantiles-1, quant_emb_dim)
        #: Compute cos(iπτ), shape==(batch_size, num_quantiles-1, quant_embedding_dim)
        pis = tf.range(1, self.quantile_embedding_dim+1, 1, dtype=tf.float32) * np.pi
        quantiles_embedded = tf.cos(quantiles_embedded * pis)
        #: DenseLayerに適用するための不自然なreshape
        quantiles_embedded = tf.reshape(
            quantiles_embedded, [batch_size * (self.num_quantiles - 1), self.quantile_embedding_dim])
        #: (batch_size * (num_quantiles-1), quant_embed_dim)
        quantiles_embedded = self.dense_2(quantiles_embedded)

        assert quantiles_embedded.shape.as_list() == [batch_size * (self.num_quantiles-1), self.state_embedding_dim]

        """ Embed quantiles_hat
        """
        quantiles_hat_embedded = tf.expand_dims(quantiles_hat, axis=2)  # (batch_size, num_quantiles, 1)
        quantiles_hat_embedded = tf.repeat(
            quantiles_hat_embedded, self.quantile_embedding_dim, axis=2)  # (batch_size, num_quantiles, quant_emb_dim)

        pis = tf.range(1, self.quantile_embedding_dim+1, 1, dtype=tf.float32) * np.pi
        quantiles_hat_embedded = tf.cos(quantiles_hat_embedded * pis)

        quantiles_hat_embedded = tf.reshape(
            quantiles_hat_embedded, [batch_size * self.num_quantiles, self.quantile_embedding_dim])
        quantiles_hat_embedded = self.dense_2(quantiles_hat_embedded)

        assert quantiles_hat_embedded.shape.as_list() == [batch_size * self.num_quantiles, self.state_embedding_dim]

        return (quantiles, quantiles_embedded,
                quantiles_hat, quantiles_hat_embedded,
                quantile_hat_probs)


class QuantileFunctionNetwork(tf.keras.layers.Layer):

    def __init__(self, action_space, num_quantiles, state_embedding_dim):

        super(QuantileFunctionNetwork, self).__init__()
        self.action_space = action_space
        self.num_quantiles = num_quantiles
        self.state_embedding_dim = state_embedding_dim

        self.dense1 = kl.Dense(512, activation="relu",
                                kernel_initializer="he_normal")

        self.out = kl.Dense(self.num_actions
                                kernel_initializer="he_normal")

    def call(self, state_embedded, quantiles_embedded, batch_size):
        """
        Note:
            quantiles_embedded:
              if quantiles_embedded,
                shape == (batch_size * (num_quantiles-1), 3136)
                N == num_quantiles - 1
              elif quantiles_hat_embedded,
                shape == (batch_size * num_quantiles, 3136)
                N == num_quantiles
        """
        N = int(quantiles_embedded.shape[0] / batch_size)

        state_embedded = tf.repeat(
            tf.expand_dims(state_embedded, axis=1), N, axis=1)
        state_embedded = tf.reshape(
            state_embedded,
            [batch_size * N, self.state_embedding_dim])
        assert state_embedded.shape == quantiles_embedded.shape

        x = state_embedded * quantiles_embedded
        x = self.dense1(x)
        quantile_values = self.out(x)

        import pdb; pdb.set_trace()

        #quantile_values = tf.reshape(out, (batch_size, self.action_space, self.N))
        return  quantile_values


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

    fqf_net = FQFNetwork(action_space=4, num_quantiles=10, quantile_embedding_dim=12)

    out = fqf_net(state)

    #selected_actions, quantile_qvalues = qnet.sample_actions(state)
    #print(selected_actions.numpy().flatten())
