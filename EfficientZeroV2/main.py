from pathlib import Path
import shutil
import collections
import contextlib
import random

import tensorflow as tf
import numpy as np
import ale_py
import gymnasium as gym
from PIL import Image
import time

import mcts
from buffers import ReplayBuffer, Experience
from networks import EFZeroNetwork


@contextlib.contextmanager
def timer(name):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"{name}: {end - start:.2f}s")


def process_frame(frame):
    image = Image.fromarray(frame)
    image = image.convert("L").resize((96, 96))
    image = np.array(image).astype(np.float32) / 255.0  #: (96, 96)
    return image


class EfficientZeroV2:

    def __init__(self, env_id: str, log_dir: str):
        self.env_id = env_id
        self.n_frames = 4
        self.action_space = gym.make(env_id).action_space.n
        self.n_supports = 51

        self.network = EFZeroNetwork(
            action_space=self.action_space, n_supports=self.n_supports
        )

        self.replay_buffer = ReplayBuffer(maxlen=100_000)
        self.batch_size = 48  # original 256
        self.gamma = 0.997
        self.unroll_steps = 3  # original 5
        self.num_simulations = 8  # original 16
        self.lambda_r, self.lambda_p, self.lambda_v, self.lambda_g = (
            1.0,
            1.0,
            0.25,
            2.0,
        )

        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-3)
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=0.2, weight_decay=0.0001, momentum=0.9
        )

        self.setup()
        self.summary_writer = (
            tf.summary.create_file_writer(str(log_dir)) if log_dir else None
        )
        self.total_steps = 0

    def setup(self):
        env = gym.make(self.env_id)
        frame, info = env.reset()
        frames = collections.deque(maxlen=self.n_frames)
        for _ in range(self.n_frames):
            frames.append(process_frame(frame))
        observations = np.stack(frames, axis=2)[np.newaxis, ...]

        states = self.network.encode(observations, training=False)
        _ = self.network.predict_policy_value_reward(states, training=False)
        next_states = self.network.predict_transition(
            states, actions=np.array([[2]]), training=False
        )
        proj = self.network.p1_network(states, training=False)
        pred = self.network.p2_network(proj, training=False)
        env.close()

    def save(self, save_dir: str):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.network.save_weights(str(save_dir / "network.weights.h5"))

    def load(self, load_dir="checkpoints/"):
        load_dir = Path(load_dir)
        self.network.load_weights(str(load_dir / "network.weights.h5"))

    def rollout(self):
        env = gym.make(self.env_id, render_mode="rgb_array")

        frame, info = env.reset()
        lives = info["lives"]
        frames = collections.deque(maxlen=self.n_frames)
        for _ in range(self.n_frames):
            frames.append(process_frame(frame))

        ep_rewards, ep_steps = 0, 0
        trajectory = []
        done, reward, info = False, 0, info
        while ep_steps < 2000:
            obs = np.stack(frames, axis=2)[np.newaxis, ...]
            action, policy, value, root_state, root_reward = mcts.search(
                observation=obs,
                action_space=self.action_space,
                network=self.network,
                num_simulations=self.num_simulations,
                gamma=self.gamma,
            )
            next_frame, next_reward, next_done, _, next_info = env.step(action)
            scolor, ecolor = ("\033[32m", "\033[0m") if reward > 0 else ("", "")
            print(
                f"{scolor}{ep_steps}, r: {reward}, r_pred:{root_reward[0]:.3f}, a:{action}, policy:{[round(p, 1) for p in policy.numpy()]}, v:{value:.1f}{ecolor}"
            )

            ep_rewards += reward
            reward = np.clip(reward, -1, 1)
            frames.append(process_frame(next_frame))

            if done:
                exp = Experience(observation=obs, action=action, reward=reward, done=1)
                trajectory.append(exp)
                break
            else:
                #: life loss as episode ends
                if info["lives"] != lives:
                    lives = info["lives"]
                    exp = Experience(
                        observation=obs, action=action, reward=reward, done=1
                    )
                else:
                    exp = Experience(
                        observation=obs, action=action, reward=reward, done=0
                    )
                trajectory.append(exp)

            done, reward, info = next_done, next_reward, next_info

            if len(self.replay_buffer) > 1000 and self.total_steps % 4 == 0:
                with timer(f"Update network"):
                    self.update_network()

            ep_steps += 1
            self.total_steps += 1

        self.replay_buffer.add(trajectory)

        with self.summary_writer.as_default():
            tf.summary.scalar("ep_rewards", ep_rewards, step=self.total_steps)
            tf.summary.scalar("ep_steps", ep_steps, step=self.total_steps)

        env.close()
        info = {"rewards": ep_rewards, "steps": ep_steps}
        return info

    def update_network(self):

        stats = collections.defaultdict(list)
        with timer(f"reanalyze({self.batch_size})"):
            (init_obs, init_action, observations, actions, rewards, masks) = (
                self.replay_buffer.sample_batch(
                    batch_size=self.batch_size,
                    unroll_steps=self.unroll_steps,
                )
            )
            B, T, H, W, C = observations.shape

            target_rewards = tf.reshape(
                self.network.scalar_to_dist(
                    tf.reshape(rewards, (B * T,)), mode="reward"
                ),
                [B, T, -1],
            )
            _, target_policies, target_values, target_states, _ = mcts.search_batch(
                observations=tf.reshape(observations, [B * T, H, W, C]),
                action_space=self.action_space,
                network=self.network,
                num_simulations=self.num_simulations,
                gamma=self.gamma,
            )

        target_states = tf.reshape(
            target_states,
            [B, T] + target_states.shape[1:],
        )
        target_policies = tf.reshape(target_policies, [B, T, -1])
        target_values = tf.reshape(
            self.network.scalar_to_dist(target_values, mode="value"), [B, T, -1]
        )

        with tf.GradientTape() as tape:
            loss = 0.0
            init_state = self.network.encode(init_obs, training=True)
            next_state = self.network.predict_transition(
                init_state, actions=init_action, training=True
            )
            for i in range(self.unroll_steps):
                state = next_state
                _, _, _, policy_t, value_t, reward_t, _, _ = (
                    self.network.predict_policy_value_reward(state, training=True)
                )

                target_policy_t = target_policies[:, i, :]
                target_value_t = target_values[:, i, :]
                target_reward_t = target_rewards[:, i, :]
                target_state_t = target_states[:, i, :]
                mask_t = masks[:, i : i + 1]

                loss_p = -tf.reduce_mean(
                    tf.reduce_sum(
                        target_policy_t * tf.math.log(policy_t + 1e-8), axis=-1
                    )
                    * mask_t
                )
                loss_v = -tf.reduce_mean(
                    tf.reduce_sum(target_value_t * tf.math.log(value_t + 1e-8), axis=-1)
                    # * mask_t
                )

                loss_r = -tf.reduce_mean(
                    tf.reduce_sum(
                        target_reward_t * tf.math.log(reward_t + 1e-8), axis=-1
                    )
                    # * mask_t
                )

                proj = self.network.p2_network(
                    self.network.p1_network(state, training=True), training=True
                )
                proj_normed = proj / (
                    tf.norm(proj, ord=2, axis=-1, keepdims=True) + 1e-12
                )

                target_proj = self.network.p1_network(target_state_t, training=False)
                target_proj_normed = target_proj / (
                    tf.norm(target_proj, ord=2, axis=-1, keepdims=True) + 1e-12
                )

                loss_g = -tf.reduce_mean(
                    tf.reduce_sum(
                        proj_normed * tf.stop_gradient(target_proj_normed), axis=-1
                    )
                    # * mask_t
                )

                loss_entropy = -1.0 * -tf.reduce_mean(
                    tf.reduce_sum(policy_t * tf.math.log(policy_t + 1e-8), axis=-1)
                )

                loss_t = (
                    self.lambda_r * loss_r
                    + self.lambda_p * loss_p
                    + self.lambda_v * loss_v
                    + self.lambda_g * loss_g
                    # + 5e-2 * loss_entropy
                ) / self.unroll_steps

                loss += loss_t

                next_state = self.network.predict_transition(
                    state, actions=actions[:, i], training=True
                )
                next_state = 0.5 * next_state + 0.5 * tf.stop_gradient(next_state) # fmt: skip
                if i == 0:
                    stats[f"loss_{i}"].append(loss_t)
                    stats[f"loss_r_{i}"].append(loss_r)
                    stats[f"loss_p_{i}"].append(loss_p)
                    stats[f"loss_v_{i}"].append(loss_v)
                    stats[f"loss_g_{i}"].append(loss_g)
                    stats[f"loss_entropy"].append(loss_entropy)
                    stats["state_init_mu"].append(tf.reduce_mean(init_state))
                    stats["rewardsum_gt"].append(tf.reduce_sum(rewards[:, 0]))

        grads = tape.gradient(loss, self.network.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, clip_norm=5)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))

        with self.summary_writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, tf.reduce_mean(value), step=self.total_steps)
            tf.summary.scalar("grad_norm", grad_norm, step=self.total_steps)

    def test_play(self, tag: int | None = None, monitor_dir: Path | None = None):
        if monitor_dir:
            env = gym.wrappers.RecordVideo(
                gym.make(self.env_id, render_mode="rgb_array"),
                video_folder=monitor_dir,
                name_prefix=tag,
            )
        else:
            env = gym.make(self.env_id)

        frames = collections.deque(maxlen=4)
        frame, _ = env.reset()
        for _ in range(4):
            frames.append(process_frame(frame))

        ep_rewards = 0
        steps = 0
        done = False
        while not done:
            obs = np.stack(frames, axis=2)[np.newaxis, ...]
            action, policy, value, root_state, root_reward = mcts.search(
                observation=obs,
                action_space=self.action_space,
                network=self.network,
                num_simulations=self.num_simulations,
                gamma=self.gamma,
            )
            next_frame, reward, done, _, info = env.step(action)
            ep_rewards += reward
            print(
                f"{steps}, r: {reward}, r_pred:{root_reward[0]:.1f}, a:{action}, policy:{[round(p, 1) for p in policy.numpy()]}, v:{value:.1f}"
            )
            frames.append(process_frame(next_frame))
            steps += 1

        print(f"Test score: {ep_rewards}")
        env.close()
        return ep_rewards


def train(
    resume_step=None,
    max_steps=100_000,
    env_id="BreakoutDeterministic-v4",
    log_dir="log",
):
    if resume_step is None:
        if Path(log_dir).exists():
            shutil.rmtree(log_dir)
        agent = EfficientZeroV2(env_id=env_id, log_dir=log_dir)
    else:
        agent = EfficientZeroV2(env_id=env_id, log_dir=log_dir)
        agent.total_steps += resume_step
        agent.load(load_dir="checkpoints")

    n = 0
    while max_steps >= agent.total_steps:
        info = agent.rollout()
        print("=" * 20)
        print(f"Episode: {n}")
        print(f"Total_steps: {agent.total_steps}")
        print("info: ", info)
        print("=" * 20)
        n += 1
        if n % 10 == 0:
            agent.save(save_dir="checkpoints/")


def test(
    load_dir: str,
    env_id="BreakoutDeterministic-v4",
):
    MONITOR_DIR = Path(__file__).parent / "mp4"
    if MONITOR_DIR.exists():
        shutil.rmtree(MONITOR_DIR)

    agent = EfficientZeroV2(env_id=env_id, log_dir=None)

    agent.load(load_dir=load_dir)
    for i in range(1, 3):
        score = agent.test_play(tag=f"{i}", monitor_dir=MONITOR_DIR)
        print("----" * 10)
        print(f"{i}: Score {score}")
        print("----" * 10)


if __name__ == "__main__":
    # train(resume_step=None)
    test(load_dir="checkpoints_bkup1")
