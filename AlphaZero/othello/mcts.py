import collections

import numpy as np
import othello


class MCTS:

    def __init__(self):

        #: visit count
        self.N = collections.defaultdict(lambda: [0] * othello.ACTION_SPACE)

        self.Q = collections.defaultdict(lambda: [0] * othello.ACTION_SPACE)

    def search(self, state, network):
        pass

    def mcts_policy(self, state):
        pass
