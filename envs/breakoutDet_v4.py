from pathlib import Path

import gym
from gym import wrappers


if __name__ == "__main__":

    ENV_NAME = "Breakout-v4"
    env = gym.make(ENV_NAME)
    monitor_dir = Path(__file__).parent / "breakoutDet-v4"
    env = wrappers.Monitor(env, monitor_dir, force=True)

    print("Environment:", ENV_NAME)
    print("Observation space:", env.observation_space)
    print("Action space", env.action_space)
    print("")

    """
    Actions
    0: Do nothing
    1: Fire (start/restart game)
    2: move right
    3: move left
    """

    done = False
    observation = env.reset()
    step = 0

    observation, reward, done, info = env.step(1)
    lives = info["ale.lives"]

    for _ in range(10):
        env.step(1)

    for _ in range(100):
        observation, reward, done, info = env.step(3)

    for step in range(10000):

        env.step(0)
        observation, reward, done, info = env.step(3)

        if step % 100 == 0:
            print(f"====STEP{step}====")
            print("obs", observation.shape)
            print("reward", reward)
            print("done", done)
            print("info", info)
            print()

        if done or reward == -1:
            print(f"====STEP{step}====")
            print("GAMEOVER!")
            print("obs", observation.shape)
            print("reward", reward)
            print("done", done)
            print("info", info)
            print()
            break

        if lives != info["ale.lives"]:
            print(f"====STEP{step}====")
            print("Lost a ball!")
            print("obs", observation.shape)
            print("reward", reward)
            print("done", done)
            print("info", info)
            print()

            lives = info["ale.lives"]
            observation, reward, done, info = env.step(1)
