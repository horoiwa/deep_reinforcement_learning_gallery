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

        self.out = kl.Dense(self.action_space,
                            kernel_initializer="he_normal")

        self.optimizer = tf.keras.optimizers.Adam(lr=0.00005)

        self.loss_func = tf.losses.Huber()

    @tf.function
    def call(self, x, training=True):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        out = self.out(x)

        return out

    def predict(self, states):
        if len(states.shape) == 3:
            states = states[np.newaxis, ...]
        return self(states).numpy()

    @tf.function
    def update(self, states, selected_actions, target_values):

        with tf.GradientTape() as tape:
            selected_actions_onehot = tf.one_hot(selected_actions,
                                                 self.action_space)

            selected_action_values = tf.reduce_sum(
                self(states) * selected_actions_onehot, axis=1)

            loss = tf.reduce_mean(
                self.loss_func(target_values, selected_action_values))

        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def summary(self):
        """確認用: self.callのtf.functionを外さないとエラー吐くことに注意
        """
        x = kl.Input(shape=(84, 84, 4))
        return tf.keras.Model(inputs=[x], outputs=self.call(x)).summary()


if __name__ == "__main__":
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    print()

    frames = np.zeros((1, 84, 84, 4))
    for i in range(4):
        frames[0, :, :, i] = np.random.randint(255, size=(84, 84)) / 255
    frames = frames.astype(np.float32)

    qnet = QNetwork(action_space=4)
    #print(qnet.summary())

    pred = qnet.predict(frames)
    print(pred)

