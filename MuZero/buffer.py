from dataclasses import dataclass

import numpy as np


class Sample:

    observation: np.ndarray
    actions: list
    rewards: list
    mcts_policies: list
    done: bool


class EpisodeBuffer:

    def __init__(self):

        self.buffer = []

    def add(self, transition):
        pass

    def to_segments(self):
        self.buffer = []
        pass

