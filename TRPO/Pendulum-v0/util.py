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


def restore_shape(flatvars, target_shape):
    pass


def main():
    mu = tf.convert_to_tensor(np.array([[10., 15.]]), dtype=tf.float32)
    std = tf.convert_to_tensor(np.array([[3., 2.]]), dtype=tf.float32)
    actions = tf.convert_to_tensor(np.array([[11, 14]]), dtype=tf.float32)

    tfd = tensorflow_probability.distributions
    dist = tfd.Normal(loc=[10., 15.], scale=[3., 2.])
    print("prob by tfp")
    print(dist.log_prob([11, 14]))
    print()
    print("prob by above")
    print(compute_logprob(mu, std, actions))


def main2():
    tfd = tensorflow_probability.distributions
    dist1 = tfd.Normal(loc=[10., 12], scale=[1., 3])
    dist2 = tfd.Normal(loc=[11., 13], scale=[2., 4])

    kl = tfd.kl_divergence(dist1, dist2)
    print("BY TFP", kl)
    print()

    old_means = tf.convert_to_tensor(np.array([[10, 12]]), dtype=tf.float32)
    old_stdevs = tf.convert_to_tensor(np.array([[1, 3]]), dtype=tf.float32)
    new_means = tf.convert_to_tensor(np.array([[11, 13]]), dtype=tf.float32)
    new_stdevs = tf.convert_to_tensor(np.array([[2, 4]]), dtype=tf.float32)
    kl2 = compute_kl(old_means, old_stdevs, new_means, new_stdevs)
    print("By manual", kl2)


def main3():
    arr = tf.convert_to_tensor(np.arange(6).reshape(-1, 1))
    arr_t = tf.transpose(arr)
    res = tf.matmul(arr_t, arr)
    print(arr, arr_t)
    print(res)
    arr2 = tf.convert_to_tensor(np.arange(12).reshape(3,4))
    arr3 = tf.convert_to_tensor(np.arange(12).reshape(3,4))
    arr4 = tf.reshape(arr2, shape=[1, -1])
    arr5 = tf.reshape(arr3, shape=[1, -1])
    print(tf.concat([arr4, arr5], axis=1))


def main4():
    arr2 = tf.convert_to_tensor(np.arange(12).reshape(3,4))
    arr3 = arr2.numpy()
    arr3[:, :] = 100

    print(arr2)
    print(arr3)


def main5():

    class M(tf.keras.Model):
        def __init__(self):
            super(M, self).__init__()
            self.d1 = tf.keras.layers.Dense(6)
            self.d2 = tf.keras.layers.Dense(2)

        def call(self, x):
            x = self.d1(x)
            x = self.d2(x)
            return x

    model = M()
    model(np.array([[1,2,3,4]]))

    var1 = model.trainable_variables
    print(var1)



if __name__ == "__main__":
    #main()
    #main2()
    #main3()
    #main4()
    main5()
