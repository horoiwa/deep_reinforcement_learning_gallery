
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow_probability as tfp
import numpy as np


class ActorCriticNet(tf.keras.Model):

    VALUE_COEF = 0.5

    ENTROPY_COEF = 0.005

    MAX_GRAD_NORM = 0.5

    def __init__(self, action_space, lr=0.00005):

        super(ActorCriticNet, self).__init__()

        self.action_space = action_space

        self.conv1 = kl.Conv2D(32, 8, strides=4, activation="relu",
                               kernel_initializer="he_normal")

        self.conv2 = kl.Conv2D(64, 4, strides=2, activation="relu",
                               kernel_initializer="he_normal")

        self.conv3 = kl.Conv2D(64, 3, strides=1, activation="relu",
                               kernel_initializer="he_normal")

        self.flat1 = kl.Flatten()

        self.dense1 = kl.Dense(512, activation="relu",
                               kernel_initializer="he_normal")

        self.logits = kl.Dense(self.action_space,
                               kernel_initializer="he_normal")

        self.values = kl.Dense(1, kernel_initializer="he_normal")

        self.optimizer = tf.keras.optimizers.Adam(lr=lr)

    @tf.function
    def call(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.flat1(x)

        x = self.dense1(x)

        logits = self.logits(x)

        values = self.values(x)

        return values, logits

    def sample_action(self, states):

        states = tf.convert_to_tensor(np.atleast_2d(states), dtype=tf.float32)

        _, logits = self(states)

        action_probs = tf.nn.softmax(logits)

        cdist = tfp.distributions.Categorical(probs=action_probs)

        action = cdist.sample()

        return action.numpy()

    def predict(self, states):

        states = tf.convert_to_tensor(np.atleast_2d(states), dtype=tf.float32)

        values, logits = self(states)

        return values.numpy(), logits.numpy()

    def update(self, states, selected_actions, discouted_rewards):

        with tf.GradientTape() as tape:

            values, logits = self(states)

            advantages = discouted_rewards - values

            value_loss = advantages ** 2

            action_probs = tf.nn.softmax(logits)

            actions_onehot = tf.one_hot(selected_actions.flatten(),
                                        self.action_space, dtype=tf.float32)

            log_selected_action_probs = tf.reduce_sum(
                actions_onehot * tf.math.log(action_probs + 1e-20),
                axis=1, keepdims=True)

            action_entropy = tf.reduce_sum(
                -1 * action_probs * tf.math.log(action_probs + 1e-20),
                axis=1, keepdims=True)

            policy_loss = log_selected_action_probs * tf.stop_gradient(advantages)
            policy_loss += self.ENTROPY_COEF * action_entropy
            policy_loss *= -1

            total_loss = tf.reduce_mean(policy_loss + self.VALUE_COEF * value_loss)

        grads = tape.gradient(total_loss, self.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, self.MAX_GRAD_NORM)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))




if __name__ == "__main__":
    state = np.ones((84, 84, 4))
    states = np.array([state, state, state])
    print(states.shape)

    state.astype(np.float32)
    states.astype(np.float32)

    acnet = ActorCriticNet(action_space=4)
    values, logits = acnet(states)

    print(values)
    print(logits)

    actions = acnet.sample_action(states)

    print()
    print("Sample action")
    print(actions)

