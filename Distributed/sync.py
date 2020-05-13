"""A2C的な動き
1stepごとにworkerに指示を出し、結果を蓄積する
Nプロセス×５ステップ分のミニバッチができたらモデルのアップデートをする
"""

import functools

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

        self.procs = [Process(target=self.worker, args=(worker_conn, env_func))
                      for (worker_conn, env_func)
                      in zip(self.worker_conns, env_funcs)]

        for p in self.procs:
            p.daemon = True
            p.start()

        self.conns[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.conns[0].recv()

        self.conns[0].send(('get_id', None))
        self.env_id = self.conns[0].recv()

        for conn in self.conns:
            conn.send(("connect_test", None))
            print(conn.recv())

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
                ob = env.reset()
                conn.send(ob)
            elif cmd == 'reset_task':
                ob = env.reset_task()
                conn.send(ob)
            elif cmd == 'close':
                conn.close()
                break
            elif cmd == 'get_spaces':
                conn.send((env.action_space, env.observation_space))
            elif cmd == 'get_id':
                conn.send(env.spec.id)
            elif cmd == "connect_test":
                conn.send(f"Connection OK: worker{env.seed}")
            else:
                raise NotImplementedError


class MasterAgent:

    def __init__(self):
        pass


def main():
    N_PROC = 4

    vec_env = SubProcVecEnv(
        [functools.partial(envfunc_proto, env_id=i) for i in range(N_PROC)])



if __name__ == "__main__":
    main()

