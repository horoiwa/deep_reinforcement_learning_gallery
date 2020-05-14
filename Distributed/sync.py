"""A2C的な動き
1stepごとにworkerに指示を出し、結果を蓄積する
Nプロセス×５ステップ分のミニバッチができたらモデルのアップデートをする
"""

import functools
import collections
from dataclasses import dataclass
import random

import numpy as np
from PIL import Image
from multiprocessing import Pipe, Process
import gym


def envfunc_proto(env_id):
    env = gym.make("BreakoutDeterministic-v4")
    env.seed = env_id
    return env


def preprocess(frame):

    frame = Image.fromarray(frame)
    frame = frame.convert("L")
    frame = frame.crop((0, 20, 160, 210))
    frame = frame.resize((84, 84))
    frame = np.array(frame, dtype=np.float32)

    return frame


@dataclass
class Step:

    reward: float

    next_state: np.ndarray

    done: bool

    info: dict


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

        steps = [conn.recv() for conn in self.conns]

        rewards = [step.reward for step in steps]

        next_states = np.stack([step.next_state for step in steps])

        dones = [step.done for step in steps]

        infos = [step.info for step in steps]


        return rewards, next_states, dones, infos

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

        NUM_FRAMES = 4

        env = env_func()

        frames = collections.deque(maxlen=NUM_FRAMES)

        lives = 5

        while True:
            cmd, action = conn.recv()
            if cmd == 'step':
                frame, reward, done, info = env.step(action)
                frame = preprocess(frame)
                frames.append(frame)

                if done:
                    frame = env.reset()
                    frame = preprocess(frame)
                    for _ in range(NUM_FRAMES):
                        frames.append(frame)

                    for _ in range(random.randint(0, 10)):
                        frame, _, _, _ = env.step(1)
                        frame = preprocess(frame)
                        frames.append(frame)

                elif info["ale.lives"] != lives:
                    lives = info["ale.lives"]
                    done = True

                next_state = np.stack(frames, axis=2)
                conn.send(Step(reward, next_state, done, info))

            elif cmd == 'reset':
                frame = env.reset()

                frame = preprocess(frame)
                for _ in range(NUM_FRAMES):
                    frames.append(frame)

                state = np.stack(frames, axis=2)
                conn.send(state)

            elif cmd == 'close':
                conn.close()
                break

            elif cmd == "connect_test":
                conn.send(f"Connection OK: worker{env.seed}")

            else:
                raise NotImplementedError()


def main():

    N_PROC = 3

    TRAJECTORY_SIZE = 5

    vecenv = SubProcVecEnv(
        [functools.partial(envfunc_proto, env_id=i) for i in range(N_PROC)])

    self_state = vecenv.reset()
    print(self_state.shape)

    mb_states, mb_actions, mb_rewards, mb_dones = [], [], [], []

    for i in range(TRAJECTORY_SIZE):
        state = self_state

        #actions = model.sample_action(state)
        actions = [1, 0, 1]
        rewards, next_states, dones, infos = vecenv.step(actions)

        mb_states.append(state)
        mb_actions.append(actions)
        mb_rewards.append(rewards)
        mb_dones.append(dones)

        self_state = next_states

    mb_states = np.array(mb_states).swapaxes(0, 1)
    mb_actions = np.array(mb_actions).T
    mb_rewards = np.array(mb_rewards).T
    mb_dones = np.array(mb_dones).T

    print(mb_states.shape)

    vecenv.close()



if __name__ == "__main__":
    main()

