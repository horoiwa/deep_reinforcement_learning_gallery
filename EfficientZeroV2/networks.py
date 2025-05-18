from typing import Literal

import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.regularizers import l2


class EFZeroNetwork(tf.keras.Model):
    def __init__(
        self,
        action_space: int,
        n_supports: int,
        reward_range: tuple[float, float] = (-2.0, 2.0),
        value_range: tuple[float, float] = (-20.0, 20.0),  # original (-299, 299)
    ):
        super(EFZeroNetwork, self).__init__()

        self.representation_network = RepresentationNetwork()
        self.pv_network = PolicyValueNetwork(
            action_space=action_space, n_supports=n_supports, value_range=value_range
        )
        self.reward_network = RewardNetwork(
            n_supports=n_supports, reward_range=reward_range
        )
        self.dynamics_network = DynamicsNetwork(action_space=action_space)

        self.p1_network = P1Network()
        self.p2_network = P2Network()

    @tf.function
    def encode(self, observations, training=False):
        state = self.representation_network(observations, training=training)
        return state

    @tf.function
    def predict_pv(self, z, training=False):
        policy_logit, value_logits = self.pv_network(z, training=training)
        policy_dist = tf.nn.softmax(policy_logit, axis=-1)
        value_dist = tf.nn.softmax(value_logits, axis=-1)
        value_scalar = tf.reduce_sum(
            value_dist * self.pv_network.supports, axis=-1, keepdims=True
        )
        return (policy_logit, policy_dist, value_dist, value_scalar)

    @tf.function
    def unroll(self, state, actions, training=False):
        next_state = self.dynamics_network(state, actions, training=training)
        reward_logit = self.reward_network(next_state, training=training)
        reward_dist = tf.nn.softmax(reward_logit, axis=-1)
        reward_scalar = tf.reduce_sum(
            reward_dist * self.reward_network.supports, axis=-1, keepdims=True
        )
        return next_state, reward_dist, reward_scalar

    def scalar_to_dist(self, x, mode: Literal["value", "reward"]):
        if mode == "value":
            supports = self.pv_network.supports
        elif mode == "reward":
            supports = self.reward_network.supports

        x = tf.reshape(tf.cast(x, dtype=tf.float32), shape=(-1, 1))
        supports = tf.expand_dims(supports, axis=0)  # (1, n_supports)

        # Calculate distances between x and supports
        distances = tf.abs(supports - x)  # (batch_size, n_supports)

        # Find the two closest supports for interpolation
        lower_indices = tf.maximum(
            tf.searchsorted(supports[0], x[:, 0], side="left") - 1, 0
        )
        upper_indices = tf.minimum(lower_indices + 1, supports.shape[1] - 1)

        lower_supports = tf.gather(supports[0], lower_indices)
        upper_supports = tf.gather(supports[0], upper_indices)

        # Compute interpolation weights
        upper_weights = (x[:, 0] - lower_supports) / (
            upper_supports - lower_supports + 1e-8
        )
        lower_weights = 1.0 - upper_weights

        # Create the distribution
        dist = tf.zeros_like(distances)
        batch_indices = tf.range(tf.shape(x)[0], dtype=tf.int32)

        # Update the distribution with lower and upper weights
        lower_indices = tf.stack([batch_indices, lower_indices], axis=1)
        upper_indices = tf.stack([batch_indices, upper_indices], axis=1)

        dist = tf.tensor_scatter_nd_update(dist, lower_indices, lower_weights)
        dist = tf.tensor_scatter_nd_update(dist, upper_indices, upper_weights)

        return dist


class RepresentationNetwork(tf.keras.Model):
    def __init__(self):
        super(RepresentationNetwork, self).__init__()
        self.conv_1 = kl.Conv2D(
            32,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=False,
            activation=None,
        )
        self.bn_1 = kl.GroupNormalization(axis=-1, groups=32)
        self.resblock_1 = ResidualBlock(dims=32)
        self.resblock_downsample = DownSampleResidualBlock(32, 64, strides=2)
        self.resblock_2 = ResidualBlock(dims=64)
        self.pooling1 = kl.AveragePooling2D(pool_size=3, strides=2, padding="same")
        self.bn_pool1 = kl.GroupNormalization(axis=-1, groups=32)
        self.resblock_3 = ResidualBlock(dims=64)
        self.pooling2 = kl.AveragePooling2D(pool_size=3, strides=2, padding="same")
        self.bn_pool2 = kl.GroupNormalization(axis=-1, groups=32)
        self.resblock_4 = ResidualBlock(dims=64)

    def call(self, observations, training=False):
        x = self.conv_1(observations)  # (96, 96, 4) -> (48, 48, 32)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)

        x = self.resblock_1(x, training=training)  # (48, 48, 32)
        x = self.resblock_downsample(x, training=training)  # (24, 24, 64)
        x = self.resblock_2(x, training=training)  # (24, 24, 64)
        x = self.pooling1(x)  # (12, 12, 64)
        x = self.bn_pool1(x, training=training)
        x = tf.nn.relu(x)
        x = self.resblock_3(x, training=training)  # (12, 12, 64)
        x = self.pooling2(x)  # (6, 6, 64)
        x = self.bn_pool2(x, training=training)
        x = tf.nn.relu(x)
        states = self.resblock_4(x, training=training)  # (6, 6, 64)

        return states


class PolicyValueNetwork(tf.keras.Model):
    def __init__(
        self, action_space: int, n_supports: int, value_range: tuple[float, float]
    ):
        super(PolicyValueNetwork, self).__init__()
        self.n_supports = n_supports
        self.value_range = value_range
        self.supports = tf.linspace(value_range[0], value_range[1], n_supports)

        self.resblock_1 = ResidualBlock(dims=64)

        self.v_conv_1 = kl.Conv2D(
            16,
            kernel_size=1,
            strides=1,
            padding="valid",
            use_bias=True,
            activation=None,
            kernel_regularizer=l2(0.0005),
        )
        self.v_bn_1 = kl.GroupNormalization(axis=-1, groups=16)
        self.v_fc_1 = kl.Dense(
            512,
            use_bias=True,
            activation=None,
            kernel_initializer="he_normal",
            kernel_regularizer=l2(0.0005),
        )
        self.v_bn_2 = kl.LayerNormalization(axis=-1)
        self.v_fc_2 = kl.Dense(
            n_supports,
            use_bias=True,
            activation=None,
            kernel_initializer="zeros",
            kernel_regularizer=l2(0.0005),
        )

        self.p_conv_1 = kl.Conv2D(
            16,
            kernel_size=1,
            strides=1,
            padding="valid",
            use_bias=True,
            activation=None,
            kernel_regularizer=l2(0.0005),
        )
        self.p_bn_1 = kl.GroupNormalization(axis=-1, groups=16)
        self.p_fc_1 = kl.Dense(
            512,
            use_bias=True,
            activation=None,
            kernel_initializer="he_normal",
            kernel_regularizer=l2(0.0005),
        )
        self.p_bn_2 = kl.LayerNormalization(axis=-1)
        self.p_fc_2 = kl.Dense(
            action_space,
            use_bias=True,
            activation=None,
            kernel_initializer="zeros",
            kernel_regularizer=l2(0.0005),
        )

    def call(self, z, training=False):
        B = z.shape[0]
        z = self.resblock_1(z, training=training)

        _value = self.v_conv_1(z)  # (6, 6, 64) -> (6, 6, 16)
        _value = self.v_bn_1(_value, training=training)
        _value = tf.nn.relu(_value)

        _value = tf.reshape(_value, shape=(B, -1))
        _value = self.v_fc_1(_value)
        _value = self.v_bn_2(_value, training=training)
        _value = tf.nn.relu(_value)
        value_logits = self.v_fc_2(_value)

        _policy = self.p_conv_1(z)  # (6, 6, 64) -> (6, 6, 16)
        _policy = self.p_bn_1(_policy, training=training)
        _policy = tf.nn.relu(_policy)

        _policy = tf.reshape(_policy, shape=(B, -1))
        _policy = self.p_fc_1(_policy)
        _policy = self.p_bn_2(_policy, training=training)
        _policy = tf.nn.relu(_policy)
        policy_logits = self.p_fc_2(_policy)

        return policy_logits, value_logits


class RewardNetwork(tf.keras.Model):
    def __init__(self, n_supports: int, reward_range: tuple[float, float]):
        super(RewardNetwork, self).__init__()
        self.n_supports = n_supports
        self.reward_range = reward_range
        self.supports = tf.linspace(reward_range[0], reward_range[1], n_supports)

        self.conv_1 = kl.Conv2D(
            16,
            kernel_size=1,
            strides=1,
            padding="valid",
            use_bias=True,
            activation=None,
            kernel_regularizer=l2(0.0005),
        )
        self.bn_1 = kl.GroupNormalization(axis=-1, groups=16)

        self.fc_1 = kl.Dense(
            512,
            use_bias=True,
            activation=None,
            kernel_initializer="he_normal",
            kernel_regularizer=l2(0.0005),
        )
        self.bn_2 = kl.LayerNormalization(axis=-1)
        self.fc_2 = kl.Dense(
            n_supports,
            use_bias=True,
            activation=None,
            kernel_initializer="zeros",
            kernel_regularizer=l2(0.0005),
        )

    def call(self, z, training=False):
        B = z.shape[0]

        x = self.conv_1(z)  # (6, 6, 64) -> (6, 6, 16)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)
        x = tf.reshape(x, shape=(B, -1))
        x = self.fc_1(x)
        x = self.bn_2(x, training=training)
        x = tf.nn.relu(x)
        logits = self.fc_2(x)

        return logits


class DynamicsNetwork(tf.keras.Model):
    def __init__(self, action_space: int):
        super(DynamicsNetwork, self).__init__()
        self.action_space = float(action_space)

        self.conv_action = kl.Conv2D(
            16,
            kernel_size=1,
            strides=1,
            padding="valid",
            use_bias=True,
            kernel_regularizer=l2(0.0005),
        )
        self.action_ln = kl.LayerNormalization(axis=-1)
        self.conv_1 = kl.Conv2D(
            64,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_regularizer=l2(0.0005),
        )
        self.bn_1 = kl.GroupNormalization(axis=-1, groups=32)
        self.resblock_1 = ResidualBlock(dims=64)

    def call(self, states, actions, training=False):
        B, H, W, C = states.shape
        action_plane = tf.ones((B, H, W, 1), dtype=tf.float32)  # (B, 6, 6, 1)
        actions = tf.reshape(
            tf.cast(actions, dtype=tf.float32), shape=(B, 1, 1, 1)
        )  # (B, 1) -> (B, 1, 1, 1)
        action_plane = action_plane * actions / self.action_space  # (B, 6, 6, 1)

        action_plane = self.conv_action(action_plane)  # (B, 6, 6, 16)
        action_plane = self.action_ln(action_plane)
        action_plane = tf.nn.relu(action_plane)

        x = tf.concat([states, action_plane], axis=-1)  # (B, 6, 6, 64+16)

        x = self.conv_1(x)  # (B, 6, 6, 64+16) -> (B, 6, 6, 64)
        x = self.bn_1(x, training=training)
        x += states
        x = tf.nn.relu(x)

        next_states = self.resblock_1(x, training=training)  # (B, 6, 6, 64)

        return next_states


class P1Network(tf.keras.Model):
    def __init__(self):
        super(P1Network, self).__init__()

        self.dense_1 = kl.Dense(
            512,
            use_bias=True,
            activation=None,
            kernel_regularizer=l2(0.0005),
        )
        self.bn_1 = kl.LayerNormalization(axis=-1)
        self.dense_2 = kl.Dense(
            512,
            use_bias=True,
            activation=None,
            kernel_regularizer=l2(0.0005),
        )
        self.bn_2 = kl.LayerNormalization(axis=-1)
        self.dense_3 = kl.Dense(
            1024,
            use_bias=True,
            activation=None,
            kernel_regularizer=l2(0.0005),
        )
        # self.bn_3 = kl.BatchNormalization(axis=-1)

    def call(self, z, training=False):
        x = tf.reshape(z, shape=(z.shape[0], -1))
        x = self.dense_1(x)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)

        x = self.dense_2(x)
        x = self.bn_2(x, training=training)
        x = tf.nn.relu(x)

        projection = self.dense_3(x)
        # projection = self.bn_3(projection, training=training)

        return projection


class P2Network(tf.keras.Model):
    def __init__(self):
        super(P2Network, self).__init__()

        self.dense_1 = kl.Dense(
            512,
            use_bias=True,
            activation=None,
            kernel_regularizer=l2(0.0005),
        )
        self.bn_1 = kl.LayerNormalization(axis=-1)
        self.dense_2 = kl.Dense(
            1024,
            use_bias=True,
            activation=None,
            kernel_regularizer=l2(0.0005),
        )

    def call(self, x, training=False):
        x = self.dense_1(x)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)

        prediction = self.dense_2(x)

        return prediction


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, dims: int):
        super(ResidualBlock, self).__init__()
        self.conv_1 = kl.Conv2D(
            dims,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            activation=None,
            kernel_regularizer=l2(0.0005),
        )
        self.bn_1 = kl.GroupNormalization(axis=-1, groups=32)

        self.conv_2 = kl.Conv2D(
            dims,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            activation=None,
            kernel_regularizer=l2(0.0005),
        )
        self.bn_2 = kl.GroupNormalization(axis=-1, groups=32)

    def call(self, inputs, training=False):
        _x = inputs
        x = self.conv_1(inputs)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv_2(x)
        x = self.bn_2(x, training=training)

        out = x + _x
        out = tf.nn.relu(out)
        return out


class DownSampleResidualBlock(tf.keras.layers.Layer):

    def __init__(self, dims_in: int, dims_out: int, strides: int = 2):
        super(DownSampleResidualBlock, self).__init__()

        self.conv_1 = kl.Conv2D(
            dims_in,
            kernel_size=3,
            strides=strides,
            padding="same",
            use_bias=False,
            activation=None,
            kernel_regularizer=l2(0.0005),
        )
        self.bn_1 = kl.GroupNormalization(axis=-1, groups=32)

        self.conv_2 = kl.Conv2D(
            dims_out,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            activation=None,
            kernel_regularizer=l2(0.0005),
        )
        self.bn_2 = kl.GroupNormalization(axis=-1, groups=32)

        self.downsample = kl.Conv2D(
            dims_out,
            kernel_size=3,
            strides=strides,
            padding="same",
            use_bias=False,
            activation=None,
            kernel_regularizer=l2(0.0005),
        )
        self.bn_down = kl.GroupNormalization(axis=-1, groups=32)

    def call(self, inputs, training=False):

        _x = self.downsample(inputs)
        _x = self.bn_down(_x, training=training)

        x = self.conv_1(inputs)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv_2(x)
        x = self.bn_2(x, training=training)
        x = tf.nn.relu(x)

        out = x + _x
        out = tf.nn.relu(out)
        return out


def test_scalar_to_dist(mode: Literal["value", "reward"]):
    model = EFZeroNetwork(action_space=3, n_supports=51)
    if mode == "reward":
        supports = model.reward_network.supports
        x = tf.constant([0.0, 0.3, 0.5, 0.8, 1.0, 1.1], dtype=tf.float32)
        th = 0.01
    elif mode == "value":
        supports = model.pv_network.supports
        x = tf.constant([0.0, 1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
        th = 0.1

    x_dist = model.scalar_to_dist(x, mode=mode)
    x_scalar = tf.reduce_sum(supports * x_dist, axis=-1)

    x_diff = tf.abs(x - x_scalar)
    print(x_diff)
    for diff in x_diff:
        assert diff < th, f"Difference is too large: {diff.numpy():.3f}"


if __name__ == "__main__":
    test_scalar_to_dist(mode="value")
    test_scalar_to_dist(mode="reward")
