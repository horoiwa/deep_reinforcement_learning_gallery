"""A2C的な動き
1stepごとにworkerに指示を出し、結果を蓄積する
Nプロセス×５ステップ分のミニバッチができたらモデルのアップデートをする
"""

import functools

import numpy as np
import gym
from multiprocessing import Pipe, Process


def envfunc_proto(env_id):
    env = gym.make("CartPole-v1")
    env.seed = env_id
    return env


class SubProcVecEnv:

    def __init__(self, env_funcs):

        self.closed = False

        self.n_envs = len(env_funcs)

        pipes = [Pipe() for _ in range(self.n_envs)]

        self.conns = [pipe[0] for pipe in pipes]

        self.worker_conns = [pipe[1] for pipe in pipes]

        self.workers = [Process(target=self.worker, args=(worker_conn, env_func))
                        for (worker_conn, env_func)
                        in zip(self.worker_conns, env_funcs)]

        for worker in self.workers:
            worker.daemon = True
            worker.start()

        for conn in self.conns:
            conn.send(("connect_test", None))
            print(conn.recv())

    def step(self, actions):
        for conn, action in zip(self.conns, actions):
            conn.send(('step', action))

        steps = [remote.recv() for remote in self.remotes]

        next_states = [step.next_state for step in steps]

        rewards = [step.reward for step in steps]

        dones = [step.done for step in steps]

        infos = [step.info for step in steps]

    def reset(self):
        for conn in self.conns:
            conn.send(('reset', None))

        states = [conn.recv() for conn in self.conns]

        return np.stack(states)

    def close(self):
        if self.closed:
            return

        for conn in self.conns:
            conn.send(('close', None))

        for worker in self.workers:
            worker.join()

        self.closed = True

    @staticmethod
    def worker(conn, env_func):

        env = env_func()

        while True:
            cmd, data = conn.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    ob = env.reset()
                conn.send((ob, reward, done, info))

            elif cmd == 'reset':
                obs = env.reset()
                conn.send(obs)

            elif cmd == 'close':
                conn.close()
                break

            elif cmd == "connect_test":
                conn.send(f"Connection OK: worker{env.seed}")

            else:
                raise NotImplementedError()


class MasterAgent:

    def __init__(self):
        pass


def main():

    N_PROC = 4

    vecenv = SubProcVecEnv(
        [functools.partial(envfunc_proto, env_id=i) for i in range(N_PROC)])

    vecenv.close()



if __name__ == "__main__":
    main()

