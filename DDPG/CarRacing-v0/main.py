from dataclasses import dataclass
import collections

import numpy as np
from PIL import Image
import gym
from gym import wrappers

from buffer import ReplayBuffer


@dataclass
class Experience:

    state: np.ndarray

    action: int

    reward: float

    next_state: np.ndarray

    done: bool


def preprocess(frame):

    frame = Image.fromarray(frame)
    frame = frame.convert("L")
    frame = frame.crop((0, 0, 96, 84))
    frame = frame.resize((84, 84))
    frame = np.array(frame, dtype=np.float32)
    frame = frame / 255

    return frame


class DDPGAgent:

    MAX_EXPERIENCES = 30000

    MIN_EXPERIENCES = 1000

    ENV_ID = 'CarRacing-v0'

    EPSILON_ANEALING = 50000

    NUM_FRAMES = 4

    def __init__(self):

        self.actor_network = None

        self.target_actor_network = None

        self.critic_network = None

        self.target_critic_network = None

        self.buffer = ReplayBuffer(max_experiences=self.MAX_EXPERIENCES)

        self.global_steps = 0

        self.hiscore = 0

    def play(self, n_episodes):

        total_rewards = []

        recent_scores = collections.deque(maxlen=5)

        for n in range(n_episodes):

            self.epsilon = 1.0 - min(0.95, self.global_steps * 0.95 / self.EPSILON_ANEALING)

            total_reward, localsteps = self.play_episode()

            total_rewards.append(total_reward)

            recent_scores.append(total_reward)

            recent_average_score = sum(recent_scores) / len(recent_scores)

            print(f"Episode {n}: {total_reward}")
            print(f"Local steps {localsteps}")
            print(f"Experiences {len(self.replay_buffer)}")
            print(f"Current epsilon {self.epsilon}")
            print(f"Current beta {self.beta}")
            print(f"Current maxp {self.replay_buffer.max_priority}")
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

        frames = collections.deque(maxlen=self.NUM_FRAMES)

        frame = self.env.reset()
        for _ in range(self.NUM_FRAMES):
            frames.append(preprocess(frame))

        for _ in range(random.randint(45, 55)):
            frame, reward, done, info = self.env.step(1)
            frames.append(preprocess(frame))

        lives = info["ale.lives"]

        state = np.stack(frames, axis=2)[np.newaxis, ...]

        while not done:

            action = self.sample_action(state)

            frame, reward, done, info = self.env.step(action)

            #: reward clipping
            reward = 1 if reward else 0

            frames.append(preprocess(frame))

            next_state = np.stack(frames, axis=2)[np.newaxis, ...]

            if info["ale.lives"] != lives:
                lives = info["ale.lives"]
                exp = Experience(state, action, reward, next_state, True)
            else:
                exp = Experience(state, action, reward, next_state, done)

            self.replay_buffer.add_experience(exp)

            state = next_state

            total_reward += reward

            steps += 1

            self.global_steps += 1

            if self.global_steps % self.UPDATE_PERIOD == 0:
                self.update_qnetwork()

            if self.global_steps % self.COPY_PERIOD == 0:
                print("==Update target newwork==")
                self.target_network.set_weights(self.q_network.get_weights())

        return total_reward, steps


def main():
    N_EPISODES = 30
    agent = DDPGAgent()
    history = agent.play(n_episodes=N_EPISODES)
    print(history)



if __name__ == "__main__":
    main()
