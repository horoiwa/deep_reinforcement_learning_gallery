import tensorflow as tf
import numpy as np
import tensorflow_probability


@tf.function
def compute_logprob(means, stdevs, actions):
    logprob = - 0.5 * np.log(2*np.pi)
    logprob += - tf.math.log(stdevs)
    logprob += - 0.5 * tf.square((actions - means) / stdevs)
    logprob = tf.reduce_sum(logprob, axis=1, keepdims=True)
    return logprob


@tf.function
def compute_kl(old_means, old_stdevs, new_means, new_stdevs):
    old_logstdevs = tf.math.log(old_stdevs)
    new_logstdevs = tf.math.log(new_stdevs)
    kl = new_logstdevs - old_logstdevs
    kl += (tf.square(old_stdevs) + tf.square(old_means - new_means)) / (2.0 * tf.square(new_stdevs))
    kl += -0.5
    return kl


def cg(hvp_func, g, iters=25):
    """
        Ax = b の近似解を共役勾配法で得る
        ※AがH, bがgに当たる
    """

    x = tf.zeros_like(g)
    r = tf.identity(g)
    p = tf.identity(r)
    r_dot_r = tf.reduce_sum(r*r)

    for _ in range(iters):
        Ap = hvp_func(p)
        v = r_dot_r / (tf.matmul(tf.transpose(p), Ap))
        x += v*p
        r -= v*Ap
        new_r_dot_r = tf.reduce_sum(r*r)
        mu = new_r_dot_r / r_dot_r
        p = r + mu * p
        r_dot_r = new_r_dot_r
        if r_dot_r < 1e-10:
            break

    return x


def restore_shape(flatvars, target_variables):
    n = 0
    weights = []
    for var in target_variables:
        size = var.shape[0] * var.shape[1] if len(var.shape) == 2 else var.shape[0]
        tmp = flatvars[n:n+size].numpy().reshape(var.shape)
        weights.append(tmp)
        n += size

    assert n == flatvars.shape[0]
    return weights
