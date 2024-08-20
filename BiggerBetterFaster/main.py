from pathlib import Path
import shutil

import gym
import tensorflow as tf

from buffers import PrioritizedReplayBuffer
from networks import BBFNetwork


class BBFAgent:
    def __init__(self, env_id: str, logdir: Path | None):

        self.env_id = env_id
        self.action_space = gym.make(self.env_id).action_space.shape[0]
        self.summary_writer = (
            tf.summary.create_file_writer(str(logdir)) if logdir else None
        )
        self.network = BBFNetwork(action_space=self.action_space, target=False)
        self.target_network = BBFNetwork(action_space=self.action_space, target=True)
        self.replay_buffer = PrioritizedReplayBuffer(maxlen=None)
        self.optimizer = tf.keras.optimizers.AdamW(lr=0.0005)
        self.gamma = 0.997
        self.batch_size = 32
        self.update_period = 1
        self.num_updates = 2
        self.gamma = 0.99

        self.setup()
        self.global_steps = 0

    def setup(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def rollout(self):
        pass

    def update_network(self):
        pass

    def update_target_network(self):
        pass

    def test_play(self):
        pass


def train(env_id="BreakoutDeterministic-v4", max_steps=100_000):
    """
    Note:
        if you failed to "pip install gym[box2d]", try "pip install box2d"
    """

    LOGDIR = Path(__file__).parent / "log"
    if LOGDIR.exists():
        shutil.rmtree(LOGDIR)

    agent = BBFAgent(env_id=env_id, logdir=LOGDIR)

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
