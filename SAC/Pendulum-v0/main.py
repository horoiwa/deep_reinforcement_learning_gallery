from pathlib import Path
import shutil

import numpy as np
import tensorflow as tf
import gym
from gym import wrappers

from models import Actor, Critic
from buffer import ReplayBuffer, Experience


class SAC:

    ACTION_SPACE = 1

    ACTION_SCALE = 2

    MAX_EXPERIENCES = 100000

    MIN_EXPERIENCES = 2500

    UPDATE_PERIOD = 4

    GAMMA = 0.99

    TAU = 0.005

    BATCH_SIZE = 256

    def __init__(self, env_id, action_space, action_scale):

        self.env_id = env_id

        self.action_space = action_space

        self.action_scale = action_scale

        self.env = gym.make(self.env_id)

        self.replay_buffer = ReplayBuffer(max_len=self.MAX_EXPERIENCES)

        self.actor = Actor(action_space=self.action_space,
                           action_scale=self.action_scale)

        self.target_actor = Actor(action_space=self.action_space,
                                  action_scale=self.action_scale)

        self.critic_1 = Critic()

        self.critic_2 = Critic()

        self.target_critic_1 = Critic()

        self.target_critic_2 = Critic()

        self.target_critic = Critic()

        self.alpha = 1.0

        self.global_steps = 0

        self._initialize_weights()

    def _initialize_weights(self):
        """1度callすることでネットワークの重みを初期化
        """

        env = gym.make(self.env_id)

        dummy_state = env.reset()
        dummy_state = (dummy_state[np.newaxis, ...]).astype(np.float32)

        dummy_action = np.random.normal(0, 0.1, size=self.action_space)
        dummy_action = (dummy_action[np.newaxis, ...]).astype(np.float32)

        self.critic_1(dummy_state, dummy_action)
        self.target_critic_1(dummy_state, dummy_action)
        self.target_critic_1.set_weights(self.critic_1.get_weights())

        self.critic_2(dummy_state, dummy_action)
        self.target_critic_2(dummy_state, dummy_action)
        self.target_critic_2.set_weights(self.critic_2.get_weights())

    def play_episode(self):

        episode_reward = 0

        local_steps = 0

        done = False

        state = self.env.reset()

        while not done:

            action = self.actor.sample_action(state)

            next_state, reward, done, _ = self.env.step(action)

            exp = Experience(state, action, reward, next_state, done)

            self.replay_buffer.push(exp)

            state = next_state

            episode_reward += reward

            local_steps += 1

            self.global_steps += 1

            if self.global_steps % self.UPDATE_PERIOD == 0:
                self.update_critic()
                self.update_policy()

        return episode_reward, local_steps

    def update_critic(self):
        pass

    def update_policy(self):
        pass

    def save_model(self):

        self.actor.save_weights("checkpoints/actor")

        self.critic_1.save_weights("checkpoints/critic")

        self.critic_2.save_weights("checkpoints/critic")

    def load_model(self):

        self.actor.load_weights("checkpoints/actor")

        self.target_actor.load_weights("checkpoints/actor")

        self.critic_1.load_weights("checkpoints/critic")

        self.critic_2.load_weights("checkpoints/critic")

        self.target_critic_1.load_weights("checkpoints/critic")

        self.target_critic_2.load_weights("checkpoints/critic")

    def testplay(self, n=1, monitordir=None):

        if monitordir:
            env = wrappers.Monitor(gym.make(self.env_id),
                                   monitordir, force=True,
                                   video_callable=(lambda ep: True))
        else:
            env = gym.make(self.env_id)

        total_rewards = []

        for _ in range(n):

            state = env.reset()

            done = False

            total_reward = 0

            while not done:

                action = self.actor.sample_action(state)

                next_state, reward, done, _ = env.step(action)

                total_reward += reward

                if done:
                    break
                else:
                    state = next_state

            total_rewards.append(total_reward)
            print()
            print(total_reward)
            print()

        return total_rewards


def main(n_episodes, n_testplay=1):

    LOGDIR = Path(__file__).parent / "log"
    if LOGDIR.exists():
        shutil.rmtree(LOGDIR)

    summary_writer = tf.summary.create_file_writer(str(LOGDIR))

    agent = SAC(env_id="Pendulum-v0", action_space=1, action_scale=2)

    episode_rewards = []

    for n in range(n_episodes):

        episode_reward, episode_steps = agent.play_episode()

        episode_rewards.append(episode_reward)

        with summary_writer.as_default():
            tf.summary.scalar("episode_reward", episode_reward, step=n)
            tf.summary.scalar("episode_steps", episode_steps, step=n)
    else:
        agent.save_model()

    if n_testplay:
        MONITOR_DIR = Path(__file__).parent / "mp4"
        if MONITOR_DIR.exists():
            shutil.rmtree(MONITOR_DIR)

        agent = SAC(env_id="Pendulum-v0", action_space=1, action_scale=2)
        agent.load_model()
        agent.testplay(n=n_testplay, monitordir=MONITOR_DIR)


if __name__ == '__main__':
    main(n_episodes=1, n_testplay=None)
