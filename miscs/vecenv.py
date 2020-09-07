import numpy as np
import ray
import gym

from collections import namedtuple



@ray.remote
class Agent:

    def __init__(self, env_id):

        self.env = gym.make(env_id)

        self.trajectory = {"s": [], "a": [], "r": [], "s2": [], "done": []}

    def reset(self):

        self.state = self.env.reset()

        return self.state

    def step(self, action):

        state = self.state

        next_state, reward, done, _ = self.env.step(action)

        if done:
            next_state = self.env.reset()

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

    def __init__(self, env_id, n_envs):

        ray.init()

        self.env_id = env_id

        self.n_envs = n_envs

        self.agents = [Agent.remote(self.env_id) for _ in range(self.n_envs)]

    def step(self, actions):

        states = ray.get(
            [agent.step.remote(action) for agent, action in zip(self.agents, actions)])

        return np.array(states)

    def reset(self):

        states = ray.get([agent.reset.remote() for agent in self.agents])

        return np.array(states)

    def get_trajectories(self):

        trajectories = ray.get([agent.get_trajectory.remote() for agent in self.agents])

        return trajectories

    def __len__(self):

        return self.n_envs


class MasterAgent:

    def __init__(self, env_id, n_envs):

        self.env_id = env_id

        self.vecenv = VecEnv(env_id=self.env_id, n_envs=n_envs)

        self.policy = None

        self.value = None

    def run(self, steps):

        states = self.vecenv.reset()

        for n in range(steps):

            actions = self.policy(states)

            states = self.vecenv.step(actions)

            if n % 1024:
                trajectories = self.vecenv.get_trajectories()
                self.update_policy(trajectories)

    def update_policy(self, trajectories):
        raise NotImplementedError()


if __name__ == "__main__":
    vecenv = VecEnv(env_id="Pendulum-v0", n_envs=4)
    actions = np.array([[0], [1], [1], [0]])
    for i in range(10):
        next_states = vecenv.step(actions)

    trajectories = vecenv.get_trajectories()
    print(trajectories[0])
