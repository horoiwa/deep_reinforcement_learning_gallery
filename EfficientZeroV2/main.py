from pathlib import Path
import shutil
import collections
import functools

import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
from PIL import Image

import mcts
from buffers import ReplayBuffer, Experience
from networks import EFZeroNetwork


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
        self.batch_size = 256
        self.gamma = 0.997
        self.replay_ratio = 1
        self.unroll_steps = 5
        self.td_steps = 5
        self.num_simulations = 16
        self.update_interval, self.target_update_interval = 100, 400
        self.lambda_1, self.lambda_2, self.lambda_3, self.lambda_4 = 1.0, 1.0, 0.25, 2.0

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
        states = np.stack(frames, axis=2)[np.newaxis, ...]
        (
            z,
            policy_prob,
            value_prob,
            reward_prob,
            z_next,
            projection,
            target_projection,
        ) = self.network(states, actions=np.array([[2]]))
        import pdb; pdb.set_trace()  # fmt: skip

    def init_tree(self) -> mcts.GumbelMCTS:

        tree = mcts.GumbelMCTS(
            pv_network=self.pv_network,
            action_space=self.action_space,
            gamma=self.gamma,
            num_simulations=self.num_simulations,
        )
        return tree

    def rollout(self):
        env = gym.make(self.env_id)

        frame, info = env.reset()
        lives = info["lives"]
        frames = collections.deque(maxlen=self.n_frames)
        for _ in range(self.n_frames):
            frames.append(process_frame(frame))

        done = False
        ep_rewards, ep_steps = 0, 0
        trajectory = []
        while not done:
            state = np.stack(frames, axis=2)[np.newaxis, ...]
            action = mcts.search(
                root_state=state,
                num_simulations=self.num_simulations,
                network=self.network,
            )
            next_frame, raw_reward, done, info = env.step(action)

            ep_rewards += raw_reward
            reward = np.clip(raw_reward, -1, 1)
            frames.append(process_frame(next_frame))

            if done:
                exp = Experience(state=state, action=action, reward=reward, done=1)
                trajectory.append(exp)
                break
            else:
                #: life loss as episode ends
                if info["lives"] != lives:
                    lives = info["lives"]
                    exp = Experience(state=state, action=action, reward=reward, done=1)
                else:
                    exp = Experience(state=state, action=action, reward=reward, done=0)
                trajectory.append(exp)

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

        info = {"rewards": ep_rewards, "steps": ep_steps}
        return info

    def update_network(self, num_updates: int):
        batchs = [
            self.replay_buffer.sample_batch(
                batch_size=self.batch_size,
                unroll_steps=self.unroll_steps,
                td_steps=self.td_steps,
                gamma=self.gamma,
            )
            in _
            for _ in range(num_updates)
        ]
        for batch in batchs:
            # reanalyze
            # update
            pass

    def update_target_network(self):
        pass


def main(max_steps=100_000, env_id="BreakoutDeterministic-v4", log_dir="logs"):
    agent = EfficientZeroV2(env_id=env_id, log_dir=log_dir)

    n = 0
    while max_steps >= agent.total_steps:
        info = agent.rollout()
        print("-" * 20)
        print(f"n: {n}, total_steps: {agent.total_steps}")
        print("info: ", info)
        n += 1


def test():
    pass


if __name__ == "__main__":
    main()
