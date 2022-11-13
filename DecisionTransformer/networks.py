import tensorflow as tf
import tensorflow.keras.layers as kl


class DecisionTransformer(tf.keras.Model):

    def __init__(self, action_space, max_timestep, context_length,
                 n_transformer_blocks=6, n_heads=8, embed_dim=128):

        super(DecisionTransformer, self).__init__()

        self.state_shape = (84, 84, 4)
        self.action_space = action_space

        self.n_layers, self.n_heads = n_layers, n_heads
        self.embed_dim = embed_dim

        self.rtgs_embedding = kl.Dense(self.embed_dim, activation="tanh")
        self.state_embedding = StateEmbedding(self.embed_dim)
        self.action_embedding = kl.Embedding(self.action_space, self.embed_dim)
        self.pos_embedding = PositionalEmbedding(
            max_timestep=max_timestep,
            context_length=context_length,
            embed_dim=embed_dim)

        self.dropout = kl.Dropout(0.1)

        self.blocks = [TransformerBlock(n_heads) for _ in range(n_transformer_blocks)]

    def call(self, rtgs, states, actions, timesteps, training=False):
        """
        Args:
            rtgs: dtype=tf.float32, shape=(B, L, 1)
            states: dtype=tf.float32, shape=(B, L, 84, 84, 4)
            actions dtype=tf.uint8, shape=(B, L, 1)
            timesteps dtype=tf.int32, shape=(B, 1, 1)

        Notes:
            - B: batch_size, L: context_length
            - timestepはstart_timestepのみなので(B, 1, 1)
            - Tokenにまとめる処理
                rtgs = np.array([1., 2., 3.,])
                states = np.array([10., 20., 30.,])
                actions = np.array([100., 200., 300.,])
                tokens = np.stack([rtgs, states, actions], axis=0).T.reshape(1, -1)
                -> [[  1.,  10., 100.,   2.,  20., 200.,   3.,  30., 300.]]
        """
        B, L = rtgs.shape[0], rtgs.shape[1]

        rtgs_embed = self.rtgs_embedding(rtgs)  #  -> (B, L, embed_dim)
        states = tf.reshape(states, shape=(-1,)+self.state_shape)  # (B*L, 84, 84, 4)
        states_embed = tf.reshape(
            self.state_embedding(states), shape=(B, L, -1))  # (B, L, embed_dim)
        action_embed = tf.math.tanh(
            self.action_embedding(tf.squeeze(actions, axis=-1))
            )  # (B, L, 1) -> (B, L) -> (B, L, embed_dim)
        pos_embed = self.pos_embedding(timesteps, L)  # (B, 3L, embed_dim)

        _tokens = tf.stack(
            [rtgs_embed, states_embed, action_embed], axis=1)  # (B, 3, L, embed_dim)
        tokens = tf.reshape(
            tf.transpose(_tokens, (0, 2, 1, 3)),
            (B, 3*L, self.embed_dim))  # (B, 3L, embed_dim)

        x = self.dropout(tokens + pos_embed, training=training)

        for block in self.blocks:
            x = block(x)

        return x


class StateEmbedding(tf.keras.layers.Layer):

    def __init__(self, embed_dim):

        super(StateEmbedding, self).__init__()

        self.conv1 = kl.Conv2D(32, 8, strides=4, activation="relu",
                               kernel_initializer="he_normal")
        self.conv2 = kl.Conv2D(64, 4, strides=2, activation="relu",
                               kernel_initializer="he_normal")
        self.conv3 = kl.Conv2D(64, 3, strides=1, activation="relu",
                               kernel_initializer="he_normal")

        self.flatten1 = kl.Flatten()

        self.dense = kl.Dense(embed_dim, activation="tanh",
                              kernel_initializer="he_normal")

    def call(self, states):

        x = states / 255.
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten1(x)
        x = self.dense(x)

        return x


class PositionalEmbedding(tf.keras.layers.Layer):

    def __init__(self, max_timestep, context_length, embed_dim):

        super(PositionalEmbedding, self).__init__()

        self.max_timestep = max_timestep

        self.context_length = context_length

        self.embed_dim = embed_dim

        self.absolute_pos_embed = tf.Variable(
            tf.zeros([1, self.max_timestep+1, self.embed_dim], tf.float32)
            )

        self.relative_pos_embed = tf.Variable(
            tf.zeros([1, self.context_length*3+1, self.embed_dim], tf.float32)
            )

    def call(self, timesteps, L):

        B = timesteps.shape[0]

        # (B, 1, 1) -> (B, 1)
        timesteps = tf.squeeze(timesteps, axis=-1)

        # (1, max_timestep+1, emb_dim) -> (B, max_timestep+1, emb_dim)
        absolute_pos_emb_all_steps = tf.repeat(self.absolute_pos_embed, B, axis=0)
        # (B, max_timestep+1, emb_dim) -> (B, 1, emb_dim)
        absolute_pos_embed = tf.gather(
            absolute_pos_emb_all_steps, indices=timesteps, axis=1, batch_dims=1)

        # (1, context_length*3+1, embed_dim) -> (1, 3L, embed_dim)
        relative_pos_embed = self.relative_pos_embed[:, :3*L, :]

        # (B, 1, emb_dim) + (B, 3L, embed_dim) -> (B, 3L, embed_dim)
        pos_embed = relative_pos_embed + absolute_pos_embed

        return pos_embed


class TransformerBlock(tf.keras.layers.Layer):

    def __init__(self, n_heads):
        super(TransformerBlock, self).__init__()

        self.n_heads = n_heads

    def call(self):
        pass


class MaskedMultiheadSelfAttention(tf.kreas.layers.Layer):

    def __init__(self):
        super(MaskedMultiheadSelfAttention, self).__init__()

    def call(self):
        pass
