from dataclasses import dataclass
import collections
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from PIL import Image
import gym
from gym import wrappers
import matplotlib.pyplot as plt

from buffer import ReplayBuffer
from models import ActorNetwork, CriticNetwork


@dataclass
class Experience:

    state: np.ndarray

    action: int

    reward: float

    next_state: np.ndarray

    done: bool


class DDPGAgent:

    MAX_EXPERIENCES = 30000

    MIN_EXPERIENCES = 300

    ENV_ID = "Pendulum-v0"

    ACTION_SPACE = 1

    OBSERVATION_SPACE = 3

    UPDATE_PERIOD = 4

    START_EPISODES = 20

    TAU = 0.02

    GAMMA = 0.99

    BATCH_SIZE = 32

    def __init__(self):

        self.env = gym.make(self.ENV_ID)

        self.env.max_episode_steps = 1000

        self.actor_network = ActorNetwork(action_space=self.ACTION_SPACE)

        self.target_actor_network = ActorNetwork(action_space=self.ACTION_SPACE)

        self.critic_network = CriticNetwork()

        self.target_critic_network = CriticNetwork()

        self.stdev = 0.2

        self.buffer = ReplayBuffer(max_experiences=self.MAX_EXPERIENCES)

        self.global_steps = 0

        self.hiscore = None

        self._build_networks()

    def _build_networks(self):
        """パラメータの初期化
        """

        dummy_state = np.random.normal(0, 0.1, size=self.OBSERVATION_SPACE)
        dummy_state = (dummy_state[np.newaxis, ...]).astype(np.float32)

        dummy_action = np.random.normal(0, 0.1, size=self.ACTION_SPACE)
        dummy_action = (dummy_action[np.newaxis, ...]).astype(np.float32)

        self.actor_network.call(dummy_state)
        self.target_actor_network.call(dummy_state)
        self.target_actor_network.set_weights(self.actor_network.get_weights())

        self.critic_network.call(dummy_state, dummy_action, training=False)
        self.target_critic_network.call(dummy_state, dummy_action, training=False)
        self.target_critic_network.set_weights(self.critic_network.get_weights())

    def play(self, n_episodes):

        total_rewards = []

        recent_scores = collections.deque(maxlen=10)

        for n in range(n_episodes):


            if n <= self.START_EPISODES:
                total_reward, localsteps = self.play_episode(random=True)
            else:
                total_reward, localsteps = self.play_episode()

            total_rewards.append(total_reward)

            recent_scores.append(total_reward)

            recent_average_score = sum(recent_scores) / len(recent_scores)

            print(f"Episode {n}: {total_reward}")
            print(f"Local steps {localsteps}")
            print(f"Experiences {len(self.buffer)}")
            print(f"Global step {self.global_steps}")
            print(f"Noise stdev {self.stdev}")
            print(f"recent average score {recent_average_score}")
            print()

            if (self.hiscore is None) or (recent_average_score > self.hiscore):
                self.hiscore = recent_average_score
                print(f"HISCORE Updated: {self.hiscore}")
                self.save_model()

        return total_rewards

    def play_episode(self, random=False):

        total_reward = 0

        steps = 0

        done = False

        state = self.env.reset()

        while not done:

            if random:
                action = np.random.uniform(-2, 2, size=self.ACTION_SPACE)
            else:
                action = self.actor_network.sample_action(state, noise=self.stdev)

            next_state, reward, done, _ = self.env.step(action)

            exp = Experience(state, action, reward, next_state, done)

            self.buffer.add_experience(exp)

            state = next_state

            total_reward += reward

            steps += 1

            self.global_steps += 1

            if self.global_steps % self.UPDATE_PERIOD == 0:
                self.update_network(self.BATCH_SIZE)
                self.update_target_network()

        return total_reward, steps

    def update_network(self, batch_size):

        if len(self.buffer) < self.MIN_EXPERIENCES:
            return

        (states, actions, rewards,
         next_states, dones) = self.buffer.get_minibatch(batch_size)

        next_actions = self.target_actor_network(next_states)

        next_qvalues = self.target_critic_network(next_states, next_actions).numpy().flatten()

        #: Compute taeget values and update CriticNetwork
        target_values = np.vstack(
            [reward + self.GAMMA * next_qvalue if not done else reward
             for reward, done, next_qvalue
             in zip(rewards, dones, next_qvalues)]).astype(np.float32)

        with tf.GradientTape() as tape:
            qvalues = self.critic_network(states, actions)
            loss = tf.reduce_mean(tf.square(target_values - qvalues))

        variables = self.critic_network.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.critic_network.optimizer.apply_gradients(zip(gradients, variables))

        #: Update ActorNetwork
        with tf.GradientTape() as tape:
            J = -1 * tf.reduce_mean(self.critic_network(states, self.actor_network(states)))

        variables = self.actor_network.trainable_variables
        gradients = tape.gradient(J, variables)
        self.actor_network.optimizer.apply_gradients(zip(gradients, variables))

    def update_target_network(self):

        # soft-target update Actor
        target_actor_weights = self.target_actor_network.get_weights()
        actor_weights = self.actor_network.get_weights()

        assert len(target_actor_weights) == len(actor_weights)

        self.target_actor_network.set_weights(
            (1 - self.TAU) * np.array(target_actor_weights)
            + (self.TAU) * np.array(actor_weights))

        # soft-target update Critic
        target_critic_weights = self.target_critic_network.get_weights()
        critic_weights = self.critic_network.get_weights()

        assert len(target_critic_weights) == len(critic_weights)

        self.target_critic_network.set_weights(
            (1 - self.TAU) * np.array(target_critic_weights)
            + (self.TAU) * np.array(critic_weights))

    def save_model(self):

        self.actor_network.save_weights("checkpoints/actor")

        self.critic_network.save_weights("checkpoints/critic")

    def load_model(self):

        self.actor_network.load_weights("checkpoints/actor")

        self.target_actor_network.load_weights("checkpoints/actor")

        self.critic_network.load_weights("checkpoints/critic")

        self.target_critic_network.load_weights("checkpoints/critic")

    def test_play(self, n, monitordir, load_model=False):

        if load_model:
            self.load_model()

        if monitordir:
            env = wrappers.Monitor(gym.make(self.ENV_ID),
                                   monitordir, force=True,
                                   video_callable=(lambda ep: ep % 1 == 0))
        else:
            env = gym.make(self.ENV_ID)

        for i in range(n):

            total_reward = 0

            steps = 0

            done = False

            state = env.reset()

            while not done:

                action = self.actor_network.sample_action(state, noise=False)

                next_state, reward, done, _ = env.step(action)

                state = next_state

                total_reward += reward

                steps += 1

            print()
            print(f"Test Play {i}: {total_reward}")
            print(f"Steps:", steps)
            print()


def main():
    N_EPISODES = 150
    agent = DDPGAgent()
    history = agent.play(n_episodes=N_EPISODES)

    print(history)
    plt.plot(range(len(history)), history)
    plt.xlabel("Episodes")
    plt.ylabel("Total Rewards")
    plt.savefig("history/log.png")

    agent.test_play(n=8, monitordir="history", load_model=True)


if __name__ == "__main__":
    main()
