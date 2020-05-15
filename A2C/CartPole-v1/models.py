
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow_probability as tfp
import numpy as np


class ActorCriticNet(tf.keras.Model):

    VALUE_COEF = 0.5

    ENTROPY_COEF = 0.01

    def __init__(self, action_space, lr=0.0005):

        super(ActorCriticNet, self).__init__()

        self.action_space = action_space

        self.dense1 = kl.Dense(128, activation="relu")

        self.dense2 = kl.Dense(128, activation="relu")

        self.values = kl.Dense(1, name="value")

        self.policy_logits = kl.Dense(action_space)

        self.optimizer = tf.optimizers.Adam(lr=lr)

    @tf.function
    def call(self, x):

        x1 = self.dense1(x)
        logits = self.policy_logits(x1)

        x2 = self.dense2(x)
        values = self.values(x2)

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

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))




if __name__ == "__main__":
    states = np.array([[-0.10430691, -1.55866031, 0.19466207, 2.51363456],
                       [-0.10430691, -1.55866031, 0.19466207, 2.51363456],
                       [-0.10430691, -1.55866031, 0.19466207, 2.51363456]])

    states.astype(np.float32)

    acnet = ActorCriticNet(2)

    values, logits = acnet(states)

    print(values)
    print(logits)

    actions = acnet.sample_action(states)

    print()
    print("Sample action")
    print(actions)

