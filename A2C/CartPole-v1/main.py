import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gym
from multiprocessing import Process, Pipe


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()

    #: ここで初めて環境を作成する
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.action_space, env.observation_space))
        elif cmd == 'get_id':
            remote.send(env.spec.id)
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents
    (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)



class SubprocVecEnv():

    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.closed = False

        nenvs = len(env_fns)

        """Pipe() は　（pipe1, pipe2）
            糸電話の端と端

            pipes = [Pipe() for _ in range(nenvs)]

            #: ホスト側パイプ
            self.remotes = [pipe[0] for pipe in pipes]

            #: リモート側パイプ
            self.work_remotes = [pipe[1] for pipe in pipes]
        """
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]

        for p in self.ps:
            p.daemon = True
            p.start()

        #:こちらのメモリからは必要なくなったので閉じる
        for remote in self.work_remotes:
            remote.close()

        #ただの確認だなこれは
        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()

        self.remotes[0].send(('get_id', None))
        self.env_id = self.remotes[0].recv()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    @property
    def num_envs(self):
        return len(self.remotes)


def make_env(env_id):
    def wrapper():
        env = gym.make("CartPole-v1")
        env.seed = env_id
        return env
    return wrapper


def main():
    create_env = (lambda: gym.make("CartPole-v1"))
    env = SubprocVecEnv([make_env(i) for i in range(4)])



def test():
    pass


def test1():
    a, b = zip(*[Pipe() for _ in range(3)])
    print(a, b)

    arr = [(i, i*10) for i in range(3)]
    c, d = zip(*arr)
    print(*arr)
    print(c) # [0, 1, 2]
    print(d) # [0, 10, 20]


if __name__ == "__main__":
    #main()
    test()
