from pathlib import Path
import shutil
import collections

import gym
import tensorflow as tf
import numpy as np

from buffers import ReplayBuffer, Experience
from networks import BBFNetwork
import utils


class BBFAgent:
    def __init__(self, env_id: str, max_steps: int, logdir: Path | None):

        self.env_id = env_id
        self.action_space = gym.make(self.env_id).action_space.n
        self.summary_writer = (
            tf.summary.create_file_writer(str(logdir)) if logdir else None
        )
        self.N = 200
        self.quantiles = [1 / (2 * self.N) + i * 1 / self.N for i in range(self.N)]

        self.network = BBFNetwork(action_space=self.action_space, N=self.N)
        self.target_network = BBFNetwork(action_space=self.action_space, N=self.N)
        self.optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0005)

        self.replay_buffer = ReplayBuffer(maxlen=max_steps)
        self.spr_weight = 5

        self.gamma = 0.997
        self.batch_size = 32
        self.replay_ratio = 2
        self.gamma = 0.99
        self.tau = 0.005

        self.shrink_factor = 0.5
        self.perturb_factor = 0.5

        self.setup()
        self.global_steps = 0

    def setup(self):
        env = gym.make(self.env_id)
        frames = collections.deque(maxlen=4)
        frame, _ = env.reset()
        for _ in range(4):
            frames.append(utils.preprocess_frame(frame))

        state = np.stack(frames, axis=2)[np.newaxis, ...]
        self.network(state)
        self.target_network(state)

        self.target_network.set_weights(self.network.get_weights())

    @property
    def epsilon(self):
        return 0.2

    @property
    def update_horizon(self):
        return 3

    def rollout(self):
        env = gym.make(self.env_id)
        frames = collections.deque(maxlen=4)

        frame, info = env.reset()
        for _ in range(4):
            frames.append(utils.preprocess_frame(frame))
        lives = info["lives"]

        ep_rewards, ep_steps = 0, 0
        done = False
        while not done:

            state = np.stack(frames, axis=2)[np.newaxis, ...]

            action = self.network.sample_action(state, epsilon=self.epsilon)
            next_frame, reward, done, info = env.step(action)
            ep_rewards += reward
            frames.append(utils.preprocess_frame(next_frame))

            if done:
                exp = Experience(state=state, action=action, reward=reward, is_done=1)
                self.replay_buffer.append(exp)
                break
            else:
                #: life loss as episode ends
                if info["lives"] != lives:
                    lives = info["lives"]
                    exp = Experience(
                        state=state, action=action, reward=reward, is_done=1
                    )
                else:
                    exp = Experience(
                        state=state, action=action, reward=reward, is_done=0
                    )
                self.replay_buffer.append(exp)

            if len(self.replay_buffer) > 100:
                for _ in range(self.replay_ratio):
                    loss, loss_qrdqn, loss_spr = self.update_network()
                self.update_target_network()

                if self.global_steps % 100 == 0:
                    with self.summary_writer.as_default():
                        tf.summary.scalar("loss", loss, step=self.global_steps)
                        tf.summary.scalar(
                            "loss_qrdqn", loss_qrdqn, step=self.global_steps
                        )
                        tf.summary.scalar("loss_spr", loss_spr, step=self.global_steps)

            ep_steps += 1
            self.global_steps += 1

        with self.summary_writer.as_default():
            tf.summary.scalar("rewards", ep_rewards, step=self.global_steps)
            tf.summary.scalar("steps", ep_steps, step=self.global_steps)

        return ep_rewards, ep_steps

    def update_network(self, k=1.0):
        """QR-DQN style loss function"""

        n_step = self.update_horizon
        states, actions, rewards, is_dones, next_states = (
            self.replay_buffer.sample_batch(
                batch_size=self.batch_size, n_step=n_step, gamma=self.gamma
            )
        )

        residual_quantile_values, _, spr_projections = (
            self.target_network.compute_quantile_values(
                states=next_states, actions=None
            )
        )
        _target_quantile_values = (
            rewards + (self.gamma**n_step) * (1 - is_dones) * residual_quantile_values
        )  # (B, N)

        target_quantile_values = tf.repeat(
            tf.expand_dims(_target_quantile_values, axis=1), self.N, axis=1
        )  # (B, N, N)

        with tf.GradientTape() as tape:
            # QR-DQN Loss
            _quantile_values, z_t, _ = self.network.compute_quantile_values(
                states=states, actions=actions
            )  # (B, N)
            quantile_values = tf.repeat(
                tf.expand_dims(_quantile_values, axis=2),
                self.N,
                axis=2,
            )  # (B, N, N)

            td_errors = target_quantile_values - quantile_values
            is_smaller_than_k = tf.abs(td_errors) < k
            squared_loss = 0.5 * tf.square(td_errors)
            linear_loss = k * (tf.abs(td_errors) - 0.5 * k)
            huberloss = tf.where(is_smaller_than_k, squared_loss, linear_loss)

            #: quantile huberloss
            indicator = tf.stop_gradient(tf.where(td_errors < 0, 1.0, 0.0))
            quantiles = tf.repeat(
                tf.expand_dims(self.quantiles, axis=1), self.N, axis=1
            )
            quantile_weights = tf.abs(quantiles - indicator)
            quantile_huberloss = quantile_weights * huberloss

            loss_qrdqn = tf.reduce_mean(
                tf.reduce_mean(tf.reduce_mean(quantile_huberloss, axis=2), axis=1)
            )

            # BYOL Loss: 正規化後のL2ノルムはコサイン類似度と等価
            spr_predictions = self.network.compute_prediction(
                z_t, actions=actions[..., :-1]
            )
            loss_spr = tf.reduce_mean(
                (spr_predictions - spr_projections) ** 2,
            )

            # Total Loss
            loss = loss_qrdqn + self.spr_weight * loss_spr

        variables = self.network.trainable_variables
        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

        return loss, loss_qrdqn, loss_spr

    def update_target_network(self):
        self.target_network.set_weights(
            [
                (1.0 - self.tau) * var_target + self.tau * var_online
                for var_target, var_online in zip(
                    self.target_network.get_weights(),
                    self.network.get_weights(),
                    strict=True,
                )
            ]
        )

    def save_weights(self, save_path: Path):
        pass

    def load_weights(self, load_path: Path):
        pass

    def test_play(self):
        pass


def train(env_id="BreakoutDeterministic-v4", max_steps=2**20):
    """
    NOTE: 2 ** 20 = 104_8576
    """

    LOGDIR = Path(__file__).parent / "log"
    if LOGDIR.exists():
        shutil.rmtree(LOGDIR)

    agent = BBFAgent(env_id=env_id, max_steps=max_steps, logdir=LOGDIR)

    episodes = 0
    while agent.global_steps < max_steps:
        rewards, steps = agent.rollout()
        episodes += 1
        steps += steps
        print(f"Episode {episodes}: {rewards}, {agent.global_steps} steps")

    agent.save("checkpoints/")
    print("Training finshed")


if __name__ == "__main__":
    train()
