from pathlib import Path
import shutil
import collections

import gym
import tensorflow as tf
import numpy as np

from buffers import PrioritizedReplayBuffer, Experience
from networks import BBFNetwork
import utils


class BBFAgent:
    def __init__(self, env_id: str, max_steps: int, logdir: Path | None):

        self.env_id = env_id
        self.action_space = gym.make(self.env_id).action_space.n
        self.summary_writer = (
            tf.summary.create_file_writer(str(logdir)) if logdir else None
        )
        self.network = BBFNetwork(action_space=self.action_space)
        self.target_network = BBFNetwork(action_space=self.action_space)
        self.replay_buffer = PrioritizedReplayBuffer(maxlen=max_steps)
        self.optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0005)
        self.gamma = 0.997
        self.batch_size = 32
        self.update_period = 1
        self.num_updates = 2
        self.gamma = 0.99

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
        return 0.8

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
            import pdb; pdb.set_trace()  # fmt: skip
            next_frame, reward, done, info = env.step(action)
            ep_rewards += reward
            frames.append(utils.preprocess_frame(next_frame))

            if done:
                exp = Experience(
                    state=state, action=action, reward=reward, is_done=done
                )
                self.replay_buffer.push(exp)
                break
            else:
                if info["lives"] != lives:
                    lives = info["ale.lives"]
                    #: life loss as episode ends
                    exp = Experience(
                        state=state, action=action, reward=reward, is_done=True
                    )
                else:
                    exp = Experience(
                        state=state, action=action, reward=reward, is_done=done
                    )
                self.replay_buffer.add(exp)

            #: Network update
            if self.steps % self.update_period == 0:
                for i in range(self.num_updates):
                    self.update_network()

            #: Target network update
            if self.steps % self.target_update_period == 0:
                self.update_target_network()

            ep_steps += 1
            self.global_steps += 1

        return ep_rewards, ep_steps

    def update_network(self):
        pass

    def update_target_network(self):
        pass

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
