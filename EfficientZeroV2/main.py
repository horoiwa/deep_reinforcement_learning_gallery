from pathlib import Path
import shutil
import collections
import functools
import contextlib

import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
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
    image = np.array(image).astype(np.float32)  #: (96, 96)
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

        self.replay_buffer = ReplayBuffer(maxlen=1_000_000)
        self.batch_size = 16
        self.gamma = 0.997
        self.replay_ratio = 1
        self.unroll_steps = 5
        self.td_steps = 5
        self.num_simulations = 16
        self.update_interval, self.target_update_interval = 100, 400
        self.lambda_r, self.lambda_p, self.lambda_v, self.lambda_g = 1.0, 1.0, 0.25, 2.0

        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=0.2, weight_decay=0.0001, momentum=0.9
        )

        self.setup()
        self.summary_writer = tf.summary.create_file_writer(str(log_dir))
        self.total_steps = 0

    def setup(self):
        env = gym.make(self.env_id)

        frame, info = env.reset()
        frames = collections.deque(maxlen=self.n_frames)
        for _ in range(self.n_frames):
            frames.append(process_frame(frame))
        observations = np.stack(frames, axis=2)[np.newaxis, ...]

        states = self.network.encode(observations, training=False)
        p, v, r = self.network.predict_policy_value_reward(states, training=False)
        next_states = self.network.predict_transition(
            states, actions=np.array([[2]]), training=False
        )
        proj = self.network.p1_network(states, training=False)
        pred = self.network.p2_network(proj, training=False)

    def rollout(self):
        env = gym.make(self.env_id)

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
            action, _, _, _ = mcts.search(
                observation=obs,
                action_space=self.action_space,
                network=self.network,
                num_simulations=self.num_simulations,
                gamma=self.gamma,
            )
            next_frame, next_reward, next_done, next_info = env.step(action)

            print("Step: ", ep_steps)
            print(f"best_action: {action}")
            print(f"reward: {reward}")
            print(f"lives: {lives}")

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

            if self.total_steps > 1000:
                if self.total_steps % self.update_interval == 0:
                    num_updates = int(self.update_interval * self.replay_ratio)
                    self.update_network(num_updates=num_updates)
                if self.total_steps % self.target_update_interval == 0:
                    self.update_target_network()

            ep_steps += 1
            self.total_steps += 1

        self.replay_buffer.add(trajectory)
        with self.summary_writer.as_default():
            tf.summary.scalar("ep_rewards", ep_rewards, step=self.total_steps)
            tf.summary.scalar("ep_steps", ep_steps, step=self.total_steps)
        if True:
            self.update_network(num_updates=1)

        info = {"rewards": ep_rewards, "steps": ep_steps}
        return info

    def update_network(self, num_updates: int):
        stats = collections.defaultdict(list)
        for i in range(num_updates):
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

            with timer(f"reanalyze({self.batch_size})"):
                _, target_policies, target_values, target_states = mcts.search_batch(
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
                init_state = self.network.encode(init_obs, training=True)
                next_state = self.network.predict_transition(
                    init_state, actions=init_action, training=True
                )
                for i in range(self.unroll_steps):
                    state = next_state
                    policy_t, value_t, reward_t = (
                        self.network.predict_policy_value_reward(state, training=True)
                    )

                    target_policy_t = target_policies[:, i, :]
                    target_value_t = target_values[:, i, :]
                    target_reward_t = target_rewards[:, i, :]
                    target_state_t = target_states[:, i, :]

                    loss_p = -tf.reduce_sum(target_policy_t * policy_t, axis=-1)
                    loss_v = -tf.reduce_sum(target_value_t * value_t, axis=-1)
                    loss_r = -tf.reduce_sum(target_reward_t * reward_t, axis=-1)

                    projection = self.network.p2_network(
                        self.network.p1_network(state), training=True
                    )
                    target_projection = self.network.p1_network(
                        target_state_t, training=False
                    )
                    import pdb; pdb.set_trace()  # fmt: skip
                    loss_g = tf.reduce_mean(
                        (state_proj - tf.stop_gradient(target_state_proj)) ** 2
                    )

                    _loss_t = (
                        self.lambda_r * loss_r
                        + self.lambda_p * loss_p
                        + self.lambda_v * loss_v
                        + self.lambda_g * loss_g
                    )
                    loss_t = tf.reduce_mean(_loss_t) / self.unroll_steps
                    loss += loss_t

                    next_state = self.network.predict_transition(
                        state, actions=actions[:, i], training=True
                    )
                    next_state = 0.5 * next_state + 0.5 * tf.stop_gradient(next_state) # fmt: skip

                    stats[f"loss_r_{i}"].append(loss_r)
                    stats[f"loss_p_{i}"].append(loss_p)
                    stats[f"loss_v_{i}"].append(loss_v)
                    stats[f"loss_g_{i}"].append(loss_g)

            grads = tape.gradient(loss, self.network.trainable_variables)
            grads, global_norm = tf.clip_by_global_norm(grads, clip_norm=5)
            self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))

        with self.summary_writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, tf.reduce_mean(value), step=self.total_steps)


def main(max_steps=100_000, env_id="BreakoutDeterministic-v4", log_dir="logs"):
    agent = EfficientZeroV2(env_id=env_id, log_dir=log_dir)

    n = 0
    while max_steps >= agent.total_steps:
        info = agent.rollout()
        print("=" * 20)
        print(f"Episode: {n}")
        print(f"Total_steps: {agent.total_steps}")
        print("info: ", info)
        print("=" * 20)
        n += 1


def test():
    pass


if __name__ == "__main__":
    main()
