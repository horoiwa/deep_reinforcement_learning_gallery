import tensorflow as tf
import tensorflow.keras.layers as kl


class EFZeroNetwork(tf.keras.Model):
    def __init__(self, action_space: int, n_supports: int):
        super(EFZeroNetwork, self).__init__()

        self.representation_network = RepresentationNetwork()
        self.policy_value_network = PolicyValueNetwork(
            action_space=action_space, n_supports=n_supports
        )
        self.reward_network = RewardNetwork(n_supports=n_supports)
        self.transition_network = TransitionNetwork(action_space=action_space)

        # self.projection_network = ProjectionNetwork()
        # self.projection_head_network = ProjectionHeadNetwork()

    def call(self, states, actions, training=False):
        z = self.representation_network(states, training=training)  # (6, 6, 64)

        policy_prob, value_prob = self.policy_value_network(z, training=training)
        reward_prob = self.reward_network(z, training=training)
        z_next = self.transition_network(z, actions, training=training)

        # projection = self.projection_network(z, training=training)
        # target_projection = self.projection_network(z_next, training=training)

        return (
            z,
            policy_prob,
            value_prob,
            reward_prob,
            z_next,
            # projection,
            # target_projection,
        )


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
        self.bn_1 = kl.BatchNormalization(axis=-1)
        self.resblock_1 = ResidualBlock(dims=32)
        self.resblock_downsample = DownSampleResidualBlock(32, 64, strides=2)
        self.resblock_2 = ResidualBlock(dims=64)
        self.pooling1 = kl.AveragePooling2D(pool_size=3, strides=2, padding="same")
        self.resblock_3 = ResidualBlock(dims=64)
        self.pooling2 = kl.AveragePooling2D(pool_size=3, strides=2, padding="same")
        self.resblock_4 = ResidualBlock(dims=64)

    def call(self, states, training=False):
        x = self.conv_1(states)  # (96, 96, 4) -> (48, 48, 32)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)

        x = self.resblock_1(x, training=training)  # (48, 48, 32)
        x = self.resblock_downsample(x, training=training)  # (24, 24, 64)
        x = self.resblock_2(x, training=training)  # (24, 24, 64)
        x = self.pooling1(x)  # (12, 12, 64)
        x = self.resblock_3(x, training=training)  # (12, 12, 64)
        x = self.pooling2(x)  # (6, 6, 64)
        z = self.resblock_4(x, training=training)  # (6, 6, 64)

        return z


class PolicyValueNetwork(tf.keras.Model):
    def __init__(self, action_space: int, n_supports: int):
        super(PolicyValueNetwork, self).__init__()
        self.resblock_1 = ResidualBlock(dims=64)

        self.v_conv_1 = kl.Conv2D(
            16, kernel_size=1, strides=1, padding="valid", use_bias=True
        )
        self.v_bn_1 = kl.BatchNormalization(axis=-1)
        self.v_fc_1 = kl.Dense(
            32, use_bias=True, activation="elu", kernel_initializer="zeros"
        )
        self.v_bn_2 = kl.BatchNormalization(axis=-1)
        self.v_fc_2 = kl.Dense(
            n_supports, use_bias=True, activation="elu", kernel_initializer="zeros"
        )

        self.p_conv_1 = kl.Conv2D(
            16, kernel_size=1, strides=1, padding="valid", use_bias=True
        )
        self.p_bn_1 = kl.BatchNormalization(axis=-1)
        self.p_fc_1 = kl.Dense(
            32, use_bias=True, activation="elu", kernel_initializer="zeros"
        )
        self.p_bn_2 = kl.BatchNormalization(axis=-1)
        self.p_fc_2 = kl.Dense(
            action_space, use_bias=True, activation="elu", kernel_initializer="zeros"
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
        value = self.v_fc_2(_value)
        value = tf.nn.softmax(value, axis=-1)

        _policy = self.p_conv_1(z)  # (6, 6, 64) -> (6, 6, 16)
        _policy = self.p_bn_1(_policy, training=training)
        _policy = tf.nn.relu(_policy)
        _policy = tf.reshape(_policy, shape=(B, -1))
        _policy = self.p_fc_1(_policy)
        _policy = self.p_bn_2(_policy, training=training)
        policy = self.p_fc_2(_policy)
        policy = tf.nn.softmax(policy, axis=-1)

        return policy, value


class RewardNetwork(tf.keras.Model):
    def __init__(self, n_supports: int):
        super(RewardNetwork, self).__init__()
        self.conv_1 = kl.Conv2D(
            16, kernel_size=1, strides=1, padding="valid", use_bias=True
        )
        self.bn_1 = kl.BatchNormalization(axis=-1)

        self.fc_1 = kl.Dense(
            32, use_bias=True, activation="elu", kernel_initializer="zeros"
        )
        self.bn_2 = kl.BatchNormalization(axis=-1)
        self.fc_2 = kl.Dense(
            n_supports, use_bias=True, activation="elu", kernel_initializer="zeros"
        )

    def call(self, z, training=False):
        B = z.shape[0]

        x = self.conv_1(z)  # (6, 6, 64) -> (6, 6, 16)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)
        x = tf.reshape(x, shape=(B, -1))
        x = self.fc_1(x)
        x = self.bn_2(x, training=training)
        logits = self.fc_2(x)
        reward = tf.nn.softmax(logits, axis=-1)

        return reward


class TransitionNetwork(tf.keras.Model):
    def __init__(self, action_space: int):
        super(TransitionNetwork, self).__init__()
        self.action_space = float(action_space)

        self.conv_action = kl.Conv2D(
            16, kernel_size=1, strides=1, padding="valid", use_bias=True
        )
        self.action_ln = kl.LayerNormalization(axis=-1)
        self.conv_1 = kl.Conv2D(
            64, kernel_size=3, strides=1, padding="same", use_bias=False
        )
        self.bn_1 = kl.BatchNormalization(axis=-1)
        self.resblock_1 = ResidualBlock(dims=64)

    def call(self, z, actions, training=False):
        B, H, W, C = z.shape
        action_plane = tf.ones((B, H, W, 1), dtype=tf.float32)  # (B, 6, 6, 1)
        actions = tf.reshape(
            tf.cast(actions, dtype=tf.float32), shape=(B, 1, 1, 1)
        )  # (B, 1) -> (B, 1, 1, 1)
        action_plane = action_plane * actions / self.action_space  # (B, 6, 6, 1)

        action_plane = self.conv_action(action_plane)  # (B, 6, 6, 16)
        action_plane = self.action_ln(action_plane)
        action_plane = tf.nn.relu(action_plane)

        x = tf.concat([z, action_plane], axis=-1)  # (B, 6, 6, 64+16)

        x = self.conv_1(x)  # (B, 6, 6, 64+16) -> (B, 6, 6, 64)
        x = self.bn_1(x, training=training)
        x += z
        x = tf.nn.relu(x)

        z_next = self.resblock_1(x, training=training)  # (B, 6, 6, 64)

        return z_next


class ProjectionNetwork(tf.keras.Model):
    def __init__(self):
        super(ProjectionNetwork, self).__init__()

        self.dense_1 = kl.Dense(1024, use_bias=True, activation=None)
        self.bn_1 = kl.BatchNormalization(axis=-1)
        self.dense_2 = kl.Dense(1024, use_bias=True, activation=None)
        self.bn_2 = kl.BatchNormalization(axis=-1)
        self.dense_3 = kl.Dense(1024, use_bias=True, activation=None)
        self.bn_3 = kl.BatchNormalization(axis=-1)

    def call(self, z, training=False):
        x = self.dense_1(z)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)

        x = self.dense_2(x)
        x = self.bn_2(x, training=training)
        x = tf.nn.relu(x)

        x = self.dense_3(x)
        projection = self.bn_3(x, training=training)

        return projection


class ProjectionHeadNetwork(tf.keras.Model):
    def __init__(self):
        super(ProjectionHeadNetwork, self).__init__()

        self.dense_1 = kl.Dense(256, use_bias=True, activation=None)
        self.bn_1 = kl.BatchNormalization(axis=-1)
        self.dense_2 = kl.Dense(1024, use_bias=True, activation=None)

    def call(self, z, training=False):
        x = self.dense_1(z)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)

        target_projection = self.dense_2(x)

        return target_projection


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
        )
        self.bn_1 = kl.BatchNormalization(axis=-1)

        self.conv_2 = kl.Conv2D(
            dims,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            activation=None,
        )
        self.bn_2 = kl.BatchNormalization(axis=-1)

    def call(self, inputs, training=False):
        _x = inputs
        x = self.conv_1(inputs)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv_2(x)
        x = self.bn_2(x)

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
        )
        self.bn_1 = kl.BatchNormalization(axis=-1)

        self.conv_2 = kl.Conv2D(
            dims_out,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            activation=None,
        )
        self.bn_2 = kl.BatchNormalization(axis=-1)

        self.downsample = kl.Conv2D(
            dims_out,
            kernel_size=3,
            strides=strides,
            padding="same",
            use_bias=False,
            activation=None,
        )

    def call(self, inputs, training=False):

        _x = self.downsample(inputs)

        x = self.conv_1(inputs)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = tf.nn.relu(x)

        out = x + _x
        out = tf.nn.relu(out)
        return out
