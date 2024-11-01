from pathlib import Path
import shutil
import collections
import functools
from typing import Optional

import tensorflow as tf
import numpy as np
import gym
from gym import wrappers

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

        self.network = self.build_network()
        self.target_network = self.build_network()
        self.target_network.set_weights(self.network.get_weights())

        self.batch_size = 32
        self.optimizer = self.build_optimizer()
        self.replay_buffer = ReplayBuffer(maxlen=max_steps)
        self.target_update_tau = 0.005

        self.eps_min = 0.01
        self.warmup_steps = 500
        self.epsilon_decay_period = 5_000
        self.max_horizon = 10
        self.min_horizon = 3

        self.replay_ratio = 2  # 2 update on every 1 environment step
        # self.spr_weight = 5
        self.spr_weight = 1
        self.reset_period = 20_000  # every 40K gradient step
        self.cycle_period = 5_000  # first 10K gradient step
        self.shrink_factor = 0.5
        self.perturb_factor = 0.5

        self.global_steps = 0

    def build_network(self):
        dummy_states, dummy_actions = self.get_dummy()
        network = BBFNetwork(
            action_space=self.action_space, n_supports=self.N, width_scale=4
        )
        _, z_t, _ = network(dummy_states)
        network.compute_prediction(z_t, dummy_actions)
        return network

    def build_optimizer(self):
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=0.0005,
            weight_decay=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1.5e-4,
        )
        return optimizer

    @functools.cache
    def get_dummy(self):
        env = gym.make(self.env_id)
        frames = collections.deque(maxlen=4)
        frame, _ = env.reset()
        for _ in range(4):
            frames.append(utils.preprocess_frame(frame))
        dummy_states = tf.convert_to_tensor(
            np.stack(frames, axis=2)[np.newaxis, ...], dtype=tf.float32
        )
        dummy_actions = tf.convert_to_tensor([[0]], dtype=tf.int32)
        return dummy_states, dummy_actions

    @property
    def gamma(self):
        steps_left = self.global_steps % self.reset_period
        gamma_1, gamma_2 = 0.97, 0.997
        gamma = gamma_1 + min(1.0, steps_left / self.cycle_period) * (gamma_2 - gamma_1)
        return gamma

    @property
    def epsilon(self):
        if self.warmup_steps > self.global_steps:
            eps = 1.0
        else:
            steps_left = (
                self.warmup_steps + self.epsilon_decay_period - self.global_steps
            )
            eps = min(
                1.0, max(0.0, steps_left / self.epsilon_decay_period) + self.eps_min
            )
        return eps

    @property
    def n_step(self) -> int:
        steps_left = self.global_steps % self.reset_period
        n_steps: float = self.max_horizon * max(
            0.0, (self.cycle_period - steps_left) / self.cycle_period
        )
        n_steps: int = max(3, round(n_steps))
        return n_steps

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
            reward = np.clip(reward, -1, 1)
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

            if self.global_steps > self.warmup_steps:
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

            if self.global_steps % self.reset_period == 0:
                self.save("checkpoints/")
                self.reset_weights()
                self.optimizer = self.build_optimizer()

            if self.global_steps != 0 and self.global_steps % 1_000 == 0:
                score: int = self.test_play()
                with self.summary_writer.as_default():
                    tf.summary.scalar("test_score", score, step=self.global_steps)

            ep_steps += 1
            self.global_steps += 1

        with self.summary_writer.as_default():
            tf.summary.scalar("rewards", ep_rewards, step=self.global_steps)
            tf.summary.scalar("steps", ep_steps, step=self.global_steps)
            tf.summary.scalar("gamma", self.gamma, step=self.global_steps)
            tf.summary.scalar("epsilon", self.epsilon, step=self.global_steps)
            tf.summary.scalar("n_steps", self.n_step, step=self.global_steps)

        return ep_rewards, ep_steps

    def update_network(self, k=1.0):
        """QR-DQN style loss function"""

        states, actions_all, rewards, is_dones, next_states = (
            self.replay_buffer.sample_batch(
                batch_size=self.batch_size, n_step=self.n_step, gamma=self.gamma
            )
        )
        actions = actions_all[:, 0:1]

        residual_quantile_values, _, _spr_projections = (
            self.target_network.compute_quantile_values(
                states=next_states, actions=None
            )
        )
        spr_projections = _spr_projections / tf.norm(
            _spr_projections, ord=2, axis=-1, keepdims=True
        )
        _target_quantile_values = (
            rewards
            + (self.gamma**self.n_step) * (1 - is_dones) * residual_quantile_values
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
            _spr_predictions = self.network.compute_prediction(z_t, actions=actions_all)
            spr_predictions = _spr_predictions / tf.norm(
                _spr_predictions, ord=2, axis=-1, keepdims=True
            )
            loss_spr = tf.reduce_mean(
                tf.reduce_sum((spr_predictions - spr_projections) ** 2, axis=-1)
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
                (1.0 - self.target_update_tau) * var_target
                + self.target_update_tau * var_online
                for var_target, var_online in zip(
                    self.target_network.get_weights(),
                    self.network.get_weights(),
                    strict=True,
                )
            ]
        )

    def reset_weights(self):
        for _network in [self.network, self.target_network]:
            random_network = self.build_network()
            for key in ["encoder", "project", "q_head", "transition_model", "predict"]:
                subnet = getattr(_network, key)
                subnet_random = getattr(random_network, key)
                if key in ["encoder", "transition_model"]:
                    # shirink and perturb
                    subnet.set_weights(
                        [
                            self.shrink_factor * online_param
                            + self.perturb_factor * random_param
                            for online_param, random_param in zip(
                                subnet.get_weights(),
                                subnet_random.get_weights(),
                            )
                        ]
                    )
                else:
                    subnet.set_weights(subnet_random.get_weights())

    def save(self, save_dir="checkpoints/"):
        save_dir = Path(save_dir)
        self.network.save_weights(str(save_dir / "network"))

    def load(self, load_dir="checkpoints/"):
        load_dir = Path(load_dir)
        self.network.load_weights(str(load_dir / "network"))
        self.target_network.load_weights(str(load_dir / "network"))

    def test_play(self, tag: Optional[int] = None, monitor_dir: Optional[Path] = None):
        print("Test play")
        if monitor_dir:
            env = wrappers.RecordVideo(
                gym.make(self.env_id, render_mode="rgb_array"),
                video_folder=monitor_dir,
                step_trigger=lambda i: True,
                name_prefix=tag,
            )
        else:
            env = gym.make(self.env_id)

        frames = collections.deque(maxlen=4)
        frame, _ = env.reset()
        for _ in range(4):
            frames.append(utils.preprocess_frame(frame))

        ep_rewards = 0
        steps = 0
        done = False
        while not done or steps <= 5000:
            state = np.stack(frames, axis=2)[np.newaxis, ...]
            action = self.network.sample_action(state, epsilon=0.01)
            next_frame, reward, done, _ = env.step(action)
            ep_rewards += reward
            frames.append(utils.preprocess_frame(next_frame))
            steps += 1
        print(f"Test score: {ep_rewards}")
        return ep_rewards


def train(env_id="BreakoutDeterministic-v4", max_steps=100_000):

    LOGDIR = Path(__file__).parent / "log"
    if LOGDIR.exists():
        shutil.rmtree(LOGDIR)

    agent = BBFAgent(env_id=env_id, max_steps=max_steps, logdir=LOGDIR)

    episodes, total_steps = 0, 0
    while total_steps < max_steps:
        rewards, steps = agent.rollout()
        episodes += 1
        total_steps += steps
        print("----" * 5)
        print(f"Episode {episodes}: {rewards}, {total_steps} steps")
        print("----" * 5)

    print("Training finshed")


def test(env_id="BreakoutDeterministic-v4"):
    import warnings

    warnings.filterwarnings("ignore")

    MONITOR_DIR = Path(__file__).parent / "mp4"
    if MONITOR_DIR.exists():
        shutil.rmtree(MONITOR_DIR)

    agent = BBFAgent(env_id=env_id, max_steps=None, logdir=None)
    agent.load("checkpoints/")
    for i in range(1, 11):
        score = agent.test_play(tag=f"{i}", monitor_dir=MONITOR_DIR)
        print("----" * 10)
        print(f"{i}: Score {score}")
        print("----" * 10)


if __name__ == "__main__":
    train()
    test()
