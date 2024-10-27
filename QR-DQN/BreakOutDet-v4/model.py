import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


class QuantileQNetwork(tf.keras.Model):

    def __init__(self, actions_space, N=200):

        super(QuantileQNetwork, self).__init__()

        self.action_space = actions_space

        self.N = N

        self.conv1 = kl.Conv2D(32, 8, strides=4, activation="relu",
                               kernel_initializer="he_normal")
        self.conv2 = kl.Conv2D(64, 4, strides=2, activation="relu",
                               kernel_initializer="he_normal")
        self.conv3 = kl.Conv2D(64, 3, strides=1, activation="relu",
                               kernel_initializer="he_normal")

        self.flatten1 = kl.Flatten()
        self.dense1 = kl.Dense(512, activation="relu",
                               kernel_initializer="he_normal")
        self.out = kl.Dense(self.action_space * self.N,
                            kernel_initializer="he_normal")

    @tf.function
    def call(self, x):

        batch_size = x.shape[0]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten1(x)
        x = self.dense1(x)

        out = self.out(x)
        quantile_values = tf.reshape(out, (batch_size, self.action_space, self.N))

        return quantile_values

    def sample_action(self, x, epsilon):

        if np.random.random() > epsilon:
            selected_actions, _ = self.sample_actions(x)
            selected_action = selected_actions[0][0].numpy()
        else:
            selected_action = np.random.choice(self.action_space)

        return selected_action

    def sample_actions(self, x):
        """ quantilesを均等幅で取っている場合はE[Z(s, a)]は単純平均と一致
        """
        quantile_qvalues = self(x)
        q_means = tf.reduce_mean(quantile_qvalues, axis=2, keepdims=True)
        selected_actions = tf.argmax(q_means, axis=1)
        return selected_actions, quantile_qvalues



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
    qnet = QuantileQNetwork(actions_space=4, N=10)
    qnet(state)

    selected_actions, quantile_qvalues = qnet.sample_actions(state)
    print(selected_actions.numpy().flatten())
