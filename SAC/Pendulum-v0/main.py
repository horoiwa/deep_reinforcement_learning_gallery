from pathlib import Path
import shutil

import tensorflow as tf
import gym

from models import Actor, Critic
from buffer import ReplayBuffer


class SAC:

    def __init__(self, env_id):

        self.env = gym.make("Pendulum-v0")

        self.replay_buffer = ReplayBuffer()

        self.actor = Actor()

        self.critic = Critic()

        self.global_steps = 0

    def play_episode(self):

        episode_reward = 0

        local_steps = 0

        done = False

        state = self.env.reset()

        while not done:

            action = self.actor.sample_action(state)

            next_state, reward, done, _ = self.env.step(action)

            exp = (state, action, reward, next_state, done)

            self.replay_buffer.add_experience(exp)

            state = next_state

            episode_reward += reward

            local_steps += 1

            self.global_steps += 1

        return episode_reward, local_steps

    def update(self):
        # Delayed Policy update
        if self.global_steps % self.CRITIC_UPDATE_PERIOD == 0:
            self.update_critic()
            if self.global_steps % self.POLICY_UPDATE_PERIOD == 0:
                self.update_policy()

    def update_critic(self):
        pass

    def update_policy(self):
        pass

    def save_model(self):
        pass


def learn(n_episodes):

    LOGDIR = Path(__file__).parent / "log"
    if LOGDIR.exists():
        shutil.rmtree(LOGDIR)

    summary_writer = tf.summary.create_file_writer(str(LOGDIR))

    sac = SAC(env_id="Pendulum-v0")

    episode_rewards = []

    for n in range(n_episodes):

        episode_reward = sac.play_episode()

        episode_rewards.append(episode_reward)

        with summary_writer.as_default:
            tf.summary.scalar("episode_reward", episode_reward, step=n)

    self.save_model()


def testplay(n, monitor_dir):
    MONITOR_DIR = Path(__file__).parent / "mp4"
    if MONITOR_DIR.exists():
        shutil.rmtree(MONITOR_DIR)


if __name__ == '__main__':
    learn(10)
    testplay(3, MONITOR_DIR)
