import collections

import numpy as np
import othello


class MCTS:

    def __init__(self, network):

        self.network = network

        #: prior probability
        self.P = {}

        #: visit count
        self.N = collections.defaultdict(lambda: [0] * othello.ACTION_SPACE)

        #: cumsum of each evaluation of Q(s, a)
        self.Q_cum = collections.defaultdict(lambda: [0] * othello.ACTION_SPACE)

    def search(root):
        pass

    def evaluate(self, state):
        pass

    def expand(self, state):
        pass

    def policy(self, state):
        pass
