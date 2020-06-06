import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


class QNetwork(tf.keras.Model):

    def __init__(self, action_space=4):

        super(QNetwork, self).__init__()

        self.action_space = action_space

        self.conv1 = kl.Conv2D(32, 8, strides=4, activation="relu",
                               kernel_initializer="he_normal")

        self.conv2 = kl.Conv2D(64, 4, strides=2, activation="relu",
                               kernel_initializer="he_normal")

        self.conv3 = kl.Conv2D(64, 3, strides=1, activation="relu",
                               kernel_initializer="he_normal")

        self.flatten1 = kl.Flatten()

        self.dense1 = kl.Dense(512, activation="relu",
                               kernel_initializer="he_normal")

        self.dense2 = kl.Dense(512, activation="relu",
                               kernel_initializer="he_normal")

        self.values = kl.Dense(1, kernel_initializer="he_normal")

        self.advantages = kl.Dense(self.action_space,
                                   kernel_initializer="he_normal")

        self.optimizer = tf.keras.optimizers.Adam(lr=0.00005)

        self.loss_func = tf.losses.Huber()

    def call(self, x, training=True):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten1(x)

        x1 = self.dense1(x)
        values = self.values(x1)

        x2 = self.dense2(x)
        advantages = self.advantages(x2)

        #scaled_advantages = advantages - tf.reduce_max(advantages)
        scaled_advantages = advantages - tf.reduce_mean(advantages)

        q_values = values + scaled_advantages

        return q_values

    def predict(self, states):
        if len(states.shape) == 3:
            states = states[np.newaxis, ...]
        return self(states).numpy()

    def update(self, states, selected_actions, target_values):

        selected_actions_onehot = tf.one_hot(selected_actions,
                                             self.action_space)
        with tf.GradientTape() as tape:

            selected_action_values = tf.reduce_sum(
                self(states) * selected_actions_onehot, axis=1)

            loss = self.loss_func(target_values, selected_action_values)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def summary(self):
        """確認用: self.callのtf.functionを外さないとエラー吐くことに注意
        """
        x = kl.Input(shape=(84, 84, 4))
        return tf.keras.Model(inputs=[x], outputs=self.call(x)).summary()


if __name__ == "__main__":

    frames = np.zeros((1, 84, 84, 4))
    for i in range(4):
        frames[0, :, :, i] = np.random.randint(255, size=(84, 84)) / 255
    frames = frames.astype(np.float32)

    qnet = QNetwork(action_space=4)
    #print(qnet.summary())

    pred = qnet.predict(frames)
    print()
    print(pred)

