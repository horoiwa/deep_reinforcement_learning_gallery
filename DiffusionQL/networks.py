import numpy as np
import tensorflow as  tf
import tensorflow.keras.layers as kl
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


def softplus(x):
    return tf.math.log(tf.math.exp(x) + 1)

def mish(x):
     return x * tf.math.tanh(softplus(x))


class QNetwork(tf.keras.Model):

    def __init__(self):
        super(QNetwork, self).__init__()

        self.dense1 = kl.Dense(256, activation=mish)
        self.dense2 = kl.Dense(256, activation=mish)
        self.dense3 = kl.Dense(256, activation=mish)
        self.q = kl.Dense(1)

    def call(self, states, actions):
        x = tf.concat([states, actions], 1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense2(x)
        q = self.q(x)
        return q


class DualQNetwork(tf.keras.Model):

    def __init__(self):
        super(DualQNetwork, self).__init__()
        self.qnet1 = QNetwork()
        self.qnet2 = QNetwork()

    @tf.function
    def call(self, states, actions):
        q1 = self.qnet1(states, actions)
        q2 = self.qnet2(states, actions)
        return q1, q2


def get_noise_schedule(T: int, b_max=10.0, b_min=0.1):
    """
    Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben
    Poole. Score-based generative modeling through stochastic differential equations.
    """
    t = tf.cast(tf.range(1, T + 1), tf.float32)
    alphas = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alphas
    return alphas, betas


class DiffusionPolicy(tf.keras.Model):

    def __init__(self, action_space: int):
        super(DiffusionPolicy, self).__init__()
        self.n_timesteps = 5
        self.action_space = action_space

        self.time_embedding = SinusoidalPositionalEmbedding(L=self.n_timesteps, D=12)
        self.dense1 = kl.Dense(256, activation=mish)
        self.dense2 = kl.Dense(256, activation=mish)
        self.dense3 = kl.Dense(256, activation=mish)
        self.out = kl.Dense(self.action_space, activation=None)

        self.alphas, self.betas = get_noise_schedule(T=self.n_timesteps)
        self.alphas_cumprod = tf.math.cumprod(self.alphas)
        self.alphas_cumprod_prev = tf.concat([[1.], self.alphas_cumprod[:-1]], axis=0)
        self.variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def call(self, x, timesteps, states):

        t = self.time_embedding(timesteps)
        x = tf.concat([x, t, states], axis=1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        eps = self.out(x)

        return eps

    @tf.function
    def compute_bc_loss(self, actions, states):

        x_0 = actions
        batch_size = x_0.shape[0]

        timesteps = tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=self.n_timesteps, dtype=tf.int32),
        alphas_cumprod_t = tf.reshape(tf.gather(self.alphas_cumprod, indices=timesteps), (-1, 1))  # (1, B, 1) -> (B, 1)

        eps = tf.random.normal(shape=x_0.shape, mean=0., stddev=1.)
        x_t = tf.sqrt(alphas_cumprod_t) * x_0 + tf.sqrt(1. - alphas_cumprod_t) * eps

        eps_pred = self(x_t, timesteps, states)
        bc_loss = tf.reduce_mean(tf.square(eps - eps_pred))

        return bc_loss

    @tf.function
    def sample_actions(self, states):
        batch_size = states.shape[0]
        x_t = tf.random.normal(shape=(batch_size, self.action_space), mean=0., stddev=1.)
        for t in reversed(range(0, self.n_timesteps)):
            t = t * tf.ones(shape=(batch_size, 1), dtype=tf.int32)  # (B, 1)
            x_t = self.inv_diffusion(x_t, t, states)

        x_0 = tf.clip_by_value(x_t, -1.0, 1.0)
        return x_0

    def inv_diffusion(self, x_t, t, states):

        beta_t = tf.reshape(tf.gather(self.betas, indices=t), (-1, 1))  # (1, B, 1) -> (B, 1)
        alphas_cumprod_t = tf.reshape(tf.gather(self.alphas_cumprod, indices=t), (-1, 1))  # (1, B, 1) -> (B, 1)

        eps_t = self(x_t, t, states)
        mu = (1.0 / tf.sqrt(1.0 - beta_t)) * (x_t - (beta_t / tf.sqrt(1.0 - alphas_cumprod_t)) * eps_t)
        sigma = tf.sqrt(tf.reshape(tf.gather(self.variance, indices=t), (-1, 1)))
        noise = tf.random.normal(shape=x_t.shape, mean=0., stddev=1.)

        x_t_minus_1 = mu + sigma * noise

        return x_t_minus_1



class SinusoidalPositionalEmbedding(tf.keras.Model):
    def __init__(self, L: int, D: int = 16):
        super(SinusoidalPositionalEmbedding, self).__init__()
        assert D % 2 == 0
        self.L, self.D = L, D
        self.table = self.get_time_embedding_table()


    def get_time_embedding_table(self):
        pos = tf.stack([tf.range(self.L, dtype=tf.float32) for _ in range(self.D // 2)], axis=0)  #(D//2, L)
        _emb = tf.reshape(tf.pow(10000, 2.0 * tf.range(self.D // 2, dtype=tf.float32) / self.D), shape=(-1, 1))  # (D//2, 1)
        _emb = tf.repeat(_emb, repeats=self.L, axis=1)  #(D//2, L)
        emb = pos * _emb  #(D//2, L)

        emb_sin = tf.sin(emb)
        emb_cos = tf.cos(emb)

        _time_embedding_table = tf.concat([emb_cos, emb_sin], axis=0)  #(D, L)
        time_embedding_table = tf.transpose(_time_embedding_table)  # (L, D)
        return time_embedding_table

    def call(self, timesteps):
        timesteps = tf.reshape(timesteps, (-1,))  # (B, 1) -> (B,)
        time_emb = tf.gather(self.table, indices=timesteps)  # (B, D)
        return time_emb
