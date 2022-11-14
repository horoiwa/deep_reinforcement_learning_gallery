import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


class DecisionTransformer(tf.keras.Model):

    def __init__(self, action_space, max_timestep, context_length,
                 n_blocks=6, n_heads=8, embed_dim=128):

        super(DecisionTransformer, self).__init__()

        self.state_shape = (84, 84, 4)
        self.action_space = action_space

        self.embed_dim = embed_dim

        self.rtgs_embedding = kl.Dense(self.embed_dim, activation="tanh")
        self.state_embedding = StateEmbedding(self.embed_dim)
        self.action_embedding = kl.Embedding(self.action_space, self.embed_dim)
        self.pos_embedding = PositionalEmbedding(
            max_timestep=max_timestep,
            context_length=context_length,
            embed_dim=embed_dim)

        self.dropout = kl.Dropout(0.1)

        self.blocks = [DecoderBlock(n_heads, embed_dim, context_length) for _ in range(n_blocks)]

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
            x = block(x, training=training)

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


class DecoderBlock(tf.keras.layers.Layer):

    def __init__(self, n_heads, embed_dim, context_length):
        super(DecoderBlock, self).__init__()

        self.n_heads, self.embed_dim = n_heads, embed_dim

        self.layer_norm_1 = kl.LayerNormalization()
        self.masked_attention = MaskedMultiHeadAttention(n_heads, embed_dim, context_length)

        self.layer_norm_2 = kl.LayerNormalization()
        self.dense_1 = kl.Dense(self.embed_dim * 4, activation="gelu")
        self.dense_2 = kl.Dense(self.embed_dim, activation=None)
        self.drop_1 = kl.Dropout(0.1)

    def call(self, x, training=False):

        x = self.layer_norm_1(x, training=training)
        x = x + self.masked_attention(x)

        x_ = self.layer_norm_2(x, training=training)
        x_ = self.dense_1(x_)
        x_ = self.dense_2(x_)
        x_ = self.drop_1(x_)

        x = x + x_

        return x


class MaskedMultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, n_heads, embed_dim, context_length):
        super(MaskedMultiHeadAttention, self).__init__()

        self.H, self.embed_dim = n_heads, embed_dim
        self.T = context_length * 3

        self.key = kl.Dense(embed_dim, activation=None)
        self.query = kl.Dense(embed_dim, activation=None)
        self.value = kl.Dense(embed_dim, activation=None)

        self.drop_1 = kl.Dropout(0.1)
        self.drop_2 = kl.Dropout(0.1)

        #: 下三角行列
        self.mask = tf.constant(
            np.tril(np.ones(shape=(1, self.H, self.T, self.T))),
            dtype=tf.bool)

    def call(self, x):

        B, T, C = x.shape  # batch, block_length, embed_dim

        # (B, T, C) -> (B, T, C) -> (B, T, H, C//H) -> (B, H, T, C//H)
        key = tf.transpose(
            tf.reshape(self.key(x), [B, T, self.H, C // self.H]),
            perm=[0, 2, 1, 3])
        query = tf.transpose(
            tf.reshape(self.query(x), [B, T, self.H, C // self.H]),
            perm=[0, 2, 1, 3])
        value = tf.transpose(
            tf.reshape(self.value(x), [B, T, self.H, C // self.H]),
            perm=[0, 2, 1, 3])

        # (B, H, T, C//H) @ (B, H, C//H, T) -> (B, H, T, T)
        attention = tf.matmul(key, query, transpose_b=True)
        attention /= tf.math.sqrt(tf.constant(C//self.H, tf.float32))

        mask = self.mask[:, :, :T, :T]
        masked_attention = tf.where(mask, attention, tf.constant(-np.inf))
        masked_attention = tf.math.softmax(masked_attention, axis=-1)
        masked_attention = self.drop_1(masked_attention)

        # (B, H, T, T) @ (B, H, T, C//H) -> (B, H, T, C//H) ->
        # (B, H, T, C//H) -> (B, T, H, C//H) -> (B, T, C)
        y = tf.matmul(masked_attention, value)
        y = tf.reshape(tf.transpose(y, [0, 2, 1, 3]), shape=[B, T, C])
        y = self.drop_2(y)

        return y
