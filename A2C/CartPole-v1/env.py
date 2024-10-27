import collections
from dataclasses import dataclass

import numpy as np
from multiprocessing import Pipe, Process


@dataclass
class Step:

    reward: float

    next_state: np.ndarray

    done: bool

    info: dict


def workerfunc(conn, env_func):

    env = env_func()

    while True:

        cmd, action = conn.recv()

        if cmd == 'step':
            next_state, reward, done, info = env.step(action)

            if done:
                next_state = env.reset()

            conn.send(Step(reward, next_state, done, info))

        elif cmd == 'reset':
            next_state = env.reset()
            conn.send(next_state)

        elif cmd == 'close':
            conn.close()
            break

        elif cmd == "connect_test":
            conn.send(f"Connection OK: worker{env.seed}")

        else:
            raise NotImplementedError()


class SubProcVecEnv:

    def __init__(self, env_funcs):

        self.closed = False

        self.n_envs = len(env_funcs)

        pipes = [Pipe() for _ in range(self.n_envs)]

        self.conns = [pipe[0] for pipe in pipes]

        self.worker_conns = [pipe[1] for pipe in pipes]

        self.workers = [Process(target=workerfunc, args=(worker_conn, env_func))
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

        steps = [conn.recv() for conn in self.conns]

        rewards = [step.reward for step in steps]

        next_states = [step.next_state for step in steps]

        dones = [step.done for step in steps]

        infos = [step.info for step in steps]

        return rewards, next_states, dones, infos

    def reset(self):
        for conn in self.conns:
            conn.send(('reset', None))

        states = [conn.recv() for conn in self.conns]

        return states

    def close(self):
        if self.closed:
            return

        for conn in self.conns:
            conn.send(('close', None))

        for worker in self.workers:
            worker.join()

        self.closed = True
