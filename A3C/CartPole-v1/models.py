import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow_probability as tfp
import numpy as np


class ActorCriticNet(tf.keras.Model):

    def __init__(self, action_space):

        super(ActorCriticNet, self).__init__()

        self.action_space = action_space

        self.dense1 = kl.Dense(100, activation="relu")

        self.dense2 = kl.Dense(100, activation="relu")

        self.values = kl.Dense(1, name="value")

        self.policy_logits = kl.Dense(action_space)

    @tf.function
    def call(self, x):

        x1 = self.dense1(x)
        logits = self.policy_logits(x1)

        x2 = self.dense2(x)
        values = self.values(x2)

        return values, logits

    def sample_action(self, state):

        state = tf.convert_to_tensor(np.atleast_2d(state), dtype=tf.float32)

        _, logits = self(state)

        action_probs = tf.nn.softmax(logits)

        cdist = tfp.distributions.Categorical(probs=action_probs)

        action = cdist.sample()

        return action.numpy()[0]


if __name__ == "__main__":
    states = np.array([[-0.10430691, -1.55866031, 0.19466207, 2.51363456],
                       [-0.10430691, -1.55866031, 0.19466207, 2.51363456],
                       [-0.10430691, -1.55866031, 0.19466207, 2.51363456]])
    states.astype(np.float32)

    actions = [0, 1, 1]

    target_values = [1, 1, 1]

    acnet = ActorCriticNet(2)

    values, logits = acnet(states)

    print(values)
    print(logits)

    state = np.array([[-0.10430691, -1.55866031, 0.19466207, 2.51363456]])

    value, logit = acnet(state)

    print(tf.nn.softmax(logit))

    state = np.array([-0.10430691, -1.55866031, 0.19466207, 2.51363456])

    print(acnet.sample_action(state))

    print("weights exists?")
    print(len(acnet.get_weights()))

    acnet2 = ActorCriticNet(2)

    acnet2.build(input_shape=(None, 4))

    print(len(acnet2.get_weights()))

    values, logits = acnet2(states)

    for var in acnet2.trainable_variables:
        print(var)

    #print(len(acnet2.get_weights()))
