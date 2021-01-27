import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


class NoisyDense(tf.keras.layers.Layer):
    def __init__(self, num_outputs, initializer="random_normal", trainable=True):
        super(NoisyDense, self).__init__()
        self.num_outputs = num_outputs
        self.initializer = initializer
        self.trainable = trainable

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(int(input_shape[-1]), self.num_outputs),
            initializer=self.initializer, trainable=self.trainable)

        self.b = self.add_weight(
            shape=(self.num_outputs,),
            initializer=self.initializer, trainable=self.trainable)

    def call(self, x):
        return tf.matmul(x, self.w) + self.b


class DuelingQNetwork(tf.keras.Model):

    def __init__(self, actions_space):

        super(DuelingQNetwork, self).__init__()

        self.action_space = actions_space

        self.conv1 = kl.Conv2D(32, 8, strides=4, activation="relu",
                               kernel_initializer="he_normal")
        self.conv2 = kl.Conv2D(64, 4, strides=2, activation="relu",
                               kernel_initializer="he_normal")
        self.conv3 = kl.Conv2D(64, 3, strides=1, activation="relu",
                               kernel_initializer="he_normal")
        self.flatten1 = kl.Flatten()

        self.dense1 = kl.Dense(512, activation="relu",
                               kernel_initializer="he_normal")
        self.value = kl.Dense(1, activation="relu",
                              kernel_initializer="he_normal")

        self.dense2 = kl.Dense(512, activation="relu",
                               kernel_initializer="he_normal")

        self.advanteges = kl.Dense(self.action_space, activation="relu",
                                   kernel_initializer="he_normal")

        self.qvalues = kl.Dense(self.action_space,
                                kernel_initializer="he_normal")

    @tf.function
    def call(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten1(x)

        x1 = self.dense1(x)
        value = self.value(x1)

        x2 = self.dense2(x)
        advantages = self.advantages(x2)

        scaled_advantages = advantages - tf.reduce_mean(advantages)
        q_values = value + scaled_advantages

        return q_values

    def sample_action(self, x, epsilon=None):

        if (epsilon is None) or (np.random.random() > epsilon):
            selected_actions, _ = self.sample_actions(x)
            selected_action = selected_actions.numpy()[0]
        else:
            selected_action = np.random.choice(self.action_space)

        return selected_action

    def sample_actions(self, x):
        qvalues = self(x)
        selected_actions = tf.cast(tf.argmax(qvalues, axis=1), tf.int32)
        return selected_actions, qvalues


class TestModel(tf.keras.Model):
    def __init__(self):
        super(TestModel, self).__init__()
        self.noisydense1 = NoisyDense(10)

    def call(self, x):
        x = self.noisydense1(x)
        return x


if __name__ == "__main__":
    import numpy as np
    x = np.arange(12).reshape(3,4).astype(np.float32)
    model = TestModel()
    x = model(x)
