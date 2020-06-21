from dataclasses import dataclass
import collections

import numpy as np
from PIL import Image
import gym
from gym import wrappers

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

    MIN_EXPERIENCES = 1000

    ENV_ID = 'BipedalWaker-v3'

    ACTION_SPACE = [(-1., 1.), (-1, 1), (-1, 1), (-1, 1)]

    NUM_FRAMES = 4

    UPDATE_PERIOD = 100

    TAU = 0.01

    BATCH_SIZE = 64

    def __init__(self):

        self.env = gym.make(self.ENV_ID)

        self.actor_network = ActorNetwork(action_space=self.ACTION_SPACE)

        self.target_actor_network = ActorNetwork(action_space=self.ACTION_SPACE)

        self.critic_network = CriticNetwork(action_space=self.ACTION_SPACE)

        self.target_critic_network = CriticNetwork(action_space=self.ACTION_SPACE)

        self.buffer = ReplayBuffer(max_experiences=self.MAX_EXPERIENCES)

        self.global_steps = 0

        self.hiscore = 0

    def play(self, n_episodes):

        total_rewards = []

        recent_scores = collections.deque(maxlen=10)

        for n in range(n_episodes):

            total_reward, localsteps = self.play_episode()

            total_rewards.append(total_reward)

            recent_scores.append(total_reward)

            recent_average_score = sum(recent_scores) / len(recent_scores)

            print(f"Episode {n}: {total_reward}")
            print(f"Local steps {localsteps}")
            print(f"Experiences {len(self.buffer)}")
            print(f"Global step {self.global_steps}")
            print(f"recent average score {recent_average_score}")
            print()

            if recent_average_score > self.hiscore:
                self.hiscore = recent_average_score
                print(f"HISCORE Updated: {self.hiscore}")
                #self.save_model()

        return total_rewards

    def play_episode(self):

        total_reward = 0

        steps = 0

        done = False

        state = self.env.reset()

        while not done:

            action = self.actor_network.sample_action(state, noise=True)

            next_state, reward, done, _ = self.env.step(action)

            exp = Experience(state, action, reward, next_state, done)

            self.buffer.add_experience(exp)

            state = next_state

            total_reward += reward

            steps += 1

            self.global_steps += 1

            if self.global_steps % self.UPDATE_PERIOD == 0:
                self.update_network()
                self.update_target_network()

        return total_reward, steps

    def update_network(self, batch_size):
        pass

    def update_target_network(self):
        pass


def main():
    N_EPISODES = 30
    agent = DDPGAgent()
    history = agent.play(n_episodes=N_EPISODES)
    print(history)



if __name__ == "__main__":
    main()
