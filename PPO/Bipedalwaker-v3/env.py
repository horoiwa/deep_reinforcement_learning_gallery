import numpy as np
import ray
import gym

import warnings
warnings.filterwarnings("ignore")


@ray.remote
class Agent:

    def __init__(self, env_id, max_timesteps):

        self.env = gym.make(env_id)

        self.max_timesteps = max_timesteps

        self.trajectory = {"s": [], "a": [], "r": [], "s2": [], "done": []}

    def reset(self):

        self.state = self.env.reset()

        self.count = 0

        return self.state

    def step(self, action):

        self.count += 1

        state = self.state

        next_state, reward, done, _ = self.env.step(action)

        if done:
            next_state = self.env.reset()
            #: bipedalwalkerの転倒時ペナルティ-100はreward_scalingを狂わせるため大幅緩和
            reward = -1
        elif self.count == self.max_timesteps:
            done = True
            next_state = self.env.reset()
            self.count = 0

        self.trajectory["s"].append(state)
        self.trajectory["a"].append(action)
        self.trajectory["r"].append(reward)
        self.trajectory["s2"].append(next_state)
        self.trajectory["done"].append(done)

        self.state = next_state

        return next_state

    def get_trajectory(self):

        trajectory = self.trajectory

        trajectory["s"] = np.array(trajectory["s"], dtype=np.float32)
        trajectory["a"] = np.array(trajectory["a"], dtype=np.float32)
        trajectory["r"] = np.array(trajectory["r"], dtype=np.float32).reshape(-1, 1)
        trajectory["s2"] = np.array(trajectory["s2"], dtype=np.float32)
        trajectory["done"] = np.array(trajectory["done"], dtype=np.float32).reshape(-1, 1)

        self.trajectory = {"s": [], "a": [], "r": [], "s2": [], "done": []}

        return trajectory


class VecEnv:

    def __init__(self, env_id, n_envs, max_timesteps=100000):

        ray.init()

        self.env_id = env_id

        self.n_envs = n_envs

        self.agents = [Agent.remote(self.env_id, max_timesteps) for _ in range(self.n_envs)]

    def step(self, actions):

        next_states = ray.get(
            [agent.step.remote(action) for agent, action in zip(self.agents, actions)])

        return np.array(next_states)

    def reset(self):

        states = ray.get([agent.reset.remote() for agent in self.agents])

        return np.array(states)

    def get_trajectories(self):

        trajectories = ray.get([agent.get_trajectory.remote() for agent in self.agents])

        return trajectories

    def __len__(self):

        return self.n_envs

    def __del__(self):
        ray.shutdown()
