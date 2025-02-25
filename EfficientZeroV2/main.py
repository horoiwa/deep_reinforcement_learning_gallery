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
from networks import PolicyValueNetwork, TransitionModel, RewardModel


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

        self.pv_network = PolicyValueNetwork()
        self.transition_model = TransitionModel()
        self.reward_model = RewardModel()

        self.replay_buffer = ReplayBuffer()

        self.setup()
        self.total_steps = 0
        self.summary_writer = tf.summary.create_file_writer(str(log_dir))

    def setup(self):
        pass

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
            # action = mcts.search(state, self.pv_network)
            action = env.action_space.sample()
            next_frame, reward, done, info = env.step(action)

            ep_rewards += reward
            reward = np.clip(reward, -1, 1)
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
                self.update_network()

            ep_steps += 1
            self.total_steps += 1

        self.replay_buffer.add(trajectory)
        with self.summary_writer.as_default():
            tf.summary.scalar("ep_rewards", ep_rewards, step=self.total_steps)
            tf.summary.scalar("ep_steps", ep_steps, step=self.total_steps)

        info = {"rewards": ep_rewards, "steps": ep_steps}
        return info


def main(max_steps=100_000, env_id="BreakoutNoFrameskip-v4", log_dir="logs"):
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
