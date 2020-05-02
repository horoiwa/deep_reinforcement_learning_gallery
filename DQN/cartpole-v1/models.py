import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


class QNetwork(tf.keras.Model):

    def __init__(self, action_space):

        super(QNetwork, self).__init__()

        self.action_space = action_space

        self.dense1 = kl.Dense(32, activation="relu", name="dense1")

        self.dense2 = kl.Dense(32, activation="relu", name="dense2")

        self.drop1 = kl.Dropout(0.2)

        self.out = kl.Dense(action_space, name="output")

        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)

    @tf.function
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.drop1(x)
        out = self.out(x)
        return out

    def predict(self, states):
        states = np.atleast_2d(states).astype(np.float32)
        return self(states).numpy()

    def update(self, states, selected_actions, target_values):

        with tf.GradientTape() as tape:
            selected_actions_onehot = tf.one_hot(selected_actions,
                                                 self.action_space)

            selected_action_values = tf.reduce_sum(
                self(states) * selected_actions_onehot, axis=1)

            loss = tf.reduce_mean(
                tf.square(target_values - selected_action_values))

        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))



if __name__ == "__main__":
    states = np.array([[-0.10430691, -1.55866031, 0.19466207, 2.51363456],
                       [-0.10430691, -1.55866031, 0.19466207, 2.51363456],
                       [-0.10430691, -1.55866031, 0.19466207, 2.51363456]])
    states.astype(np.float32)
    actions = [0, 1, 1]
    target_values = [1, 1, 1]
    qnet = QNetwork(action_space=2)
    pred = qnet.predict(states)

    print(pred)

    qnet.update(states, actions, target_values)
