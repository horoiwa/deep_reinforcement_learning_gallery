import math
import random
import json

import numpy as np
import tensorflow as tf


class AtariMCTS:
    """ Single player MCTS """

    def __init__(self, action_space,
                 pv_network, dynamics_network, gamma,
                 dirichlet_alpha, c_puct=1.25, epsilon=0.25):

        self.action_space = action_space

        self.pv_network = pv_network

        self.dynamics_network = dynamics_network

        self.dirichlet_alpha = dirichlet_alpha

        self.gamma = gamma

        self.c_puct = c_puct

        self.eps = epsilon

        #: prior probability
        self.P = {}

        #: visit count
        self.N = {}

        #: state action value
        self.Q = {}

        #: Immediate reward
        self.R = {}

        #: next states
        self.S = {}

        self.q_min, self.q_max = np.inf, -np.inf

    def _normalize(self, q):

        if self.q_max > self.q_min:
            return (q - self.q_min) / (self.q_max - self.q_min)
        else:
            return q

    def search(self, root_state, num_simulations, T):

        #: EagerTensor.ref() is hashable
        s = root_state.ref()

        if s not in self.P:
            root_value = self.expand(root_state)

        #: Adding Dirichlet noise to the prior probabilities in the root node
        if self.dirichlet_alpha is not None:
            dirichlet_noise = np.random.dirichlet(
                alpha=[self.dirichlet_alpha]*self.action_space)
            self.P[s] = [(1 - self.eps) * prob + self.eps * noise
                         for prob, noise in zip(self.P[s], dirichlet_noise)]

        #: MCTS simulation
        for _ in range(num_simulations):

            U = [self.c_puct * self.P[s][a] * math.sqrt(sum(self.N[s])) / (1 + self.N[s][a])
                 for a in range(self.action_space)]

            Q = [self._normalize(q) if n != 0 else 0.
                 for q, n in zip(self.Q[s], self.N[s])]

            ucb_scores = np.array([u + q for u, q in zip(U, Q)])

            #: np.argmaxでは同値maxで偏るため
            a = random.choice(np.where(ucb_scores == ucb_scores.max())[0])

            next_state = self.S[s][a]

            #: cumulative reward with n-step bootstrapping
            G = self.R[s][a] + self.gamma * self.evaluate(next_state)

            self.Q[s][a] = (self.N[s][a] * self.Q[s][a] + G) / (self.N[s][a] + 1)
            self.N[s][a] = self.N[s][a] + 1

            self.q_min = min(self.q_min, self.Q[s][a])
            self.q_max = max(self.q_max, self.Q[s][a])

        visit_counts = np.array(self.N[s], dtype=np.float32)

        mcts_policy = visit_counts ** (1 / T) / (visit_counts ** (1 / T)).sum()

        action = np.random.choice(range(self.action_space), p=mcts_policy)

        self.pruning(root_state, action)

        return mcts_policy, action, root_value

    def expand(self, state):

        s = state.ref()

        nn_policy, nn_value = self.pv_network.predict(state)
        next_states, rewards = self.dynamics_network.predict_all(state)

        self.P[s] = nn_policy.tolist()
        self.N[s] = [0] * self.action_space
        self.Q[s] = [0] * self.action_space
        self.R[s] = [r for r in rewards.numpy().flatten()]

        #: cache valid actions and next state to save computation
        #: instead of [i, ...], [i:i+1, ...] to keep the dimension
        self.S[s] = [next_states[i:i+1, ...] for i in range(self.action_space)]

        return nn_value

    def evaluate(self, state):

        s = state.ref()

        if s not in self.P:
            #: 未展開ノードへの到達時
            value = self.expand(state)
            return value

        else:
            #: 子ノードをUCBスコアに従って選択
            U = [self.c_puct * self.P[s][a] * math.sqrt(sum(self.N[s])) / (1 + self.N[s][a])
                 for a in range(self.action_space)]

            Q = [self._normalize(q) if n != 0 else 0.
                 for q, n in zip(self.Q[s], self.N[s])]

            ucb_scores = np.array([u + q for u, q in zip(U, Q)])

            #: np.argmaxでは同値maxで偏るため
            a = random.choice(np.where(ucb_scores == ucb_scores.max())[0])

            next_state = self.S[s][a]

            G = self.R[s][a] + self.gamma * self.evaluate(next_state)

            self.Q[s][a] = (self.N[s][a] * self.Q[s][a] + G) / (self.N[s][a] + 1)
            self.N[s][a] = self.N[s][a] + 1

            self.q_min = min(self.q_min, self.Q[s][a])
            self.q_max = max(self.q_max, self.Q[s][a])

            return G

    def pruning(self, state, action):

        s = state.ref()

        redundunt_states = [s]

        for a in range(self.action_space):

            if a != action:
                redundunt_states += self._recurrent_search(self.S[s][a].ref())

        for rs in redundunt_states:
            del self.P[rs],  self.N[rs], self.Q[rs], self.R[rs], self.S[rs]

    def _recurrent_search(self, s):

        if s in self.S:
            redundunt_states = [s]
            for a in range(self.action_space):
                redundunt_states += self._recurrent_search(self.S[s][a].ref())
            return redundunt_states
        else:
            return []
