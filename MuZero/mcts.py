import math
import random
import json

import numpy as np
import tensorflow as tf


class AtariMCTS:
    """ Single player MCTS """

    def __init__(self, n_mcts_simulation, action_space,
                 pv_network, dynamics_network,
                 dirichlet_alpha, c_puct=1.25, epsilon=0.25):

        self.n_mcts_simulation = n_mcts_simulation

        self.action_space = action_space

        self.pv_network = pv_network

        self.dynamics_network = dynamics_network

        self.dirichlet_alpha = dirichlet_alpha

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

        #: cache next states to save computation
        self.next_states = {}

        self.q_min, self.q_max = np.inf, -np.inf

    def search(self, root_state, num_simulations):
        """
        Args:
            root_state (Tensor)
            num_simulations (int)
        """

        #: Tensor.ref() is hashable
        s = root_state.ref()

        if s not in self.P:
            _ = self._expand(root_state)

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

            Q = [(q - self.q_min) / (self.q_max - self.q_min)
                 if n != 0 else 0.
                 for q, n in zip(self.Q[s], self.N[s])]

            scores = np.array([u + q for u, q in zip(U, Q)])

            #: np.argmaxでは同値maxで偏るため
            a = random.choice(np.where(scores == scores.max())[0])

            next_state = self.next_states[s][a]

            G = self.R[s][a] + self.gamma * self._evaluate(next_state)

            self.Q[s][a] = (self.N[s][a] * self.Q[s][a] + G) / (self.N[s][a] + 1)

            self.N[s][a] = self.N[s][a] + 1

            self.q_min = min(self.q_min, self.Q[s][a])
            self.q_max = max(self.q_max, self.Q[s][a])

        mcts_policy = [n / sum(self.N[s]) for n in self.N[s]]

        return mcts_policy

    def _expand(self, state):

        s = state.ref()

        with tf.device("/cpu:0"):
            nn_policy, nn_value = self.pv_network.predict(state)
            next_states, rewards = self.dynamics_network.predict_all(state)

        self.P[s] = nn_policy.tolist()
        self.N[s] = [0] * self.action_space
        self.Q[s] = [0] * self.action_space
        self.R[s] = [r for r in rewards.numpy()]

        #: cache valid actions and next state to save computation
        #: instead of [i, ...], [i:i+1, ...] to keep the dimension
        self.next_states[s] = [
            next_states[i:i+1, ...] for i in range(self.action_space)]

        return nn_value

    def _evaluate(self, state, current_player):

        s = self.state_to_str(state, current_player)

        if othello.is_done(state, current_player):
            #: ゲーム終了
            reward_first, reward_second = othello.get_result(state)
            reward = reward_first if current_player == 1 else reward_second
            return reward

        elif s not in self.P:
            #: ゲーム終了していないリーフノードの場合は展開
            nn_value = self._expand(state, current_player)
            return nn_value

        else:
            #: 子ノードをevaluate
            U = [self.c_puct * self.P[s][a] * math.sqrt(sum(self.N[s])) / (1 + self.N[s][a])
                 for a in range(othello.ACTION_SPACE)]
            Q = [q / n if n != 0 else q for q, n in zip(self.W[s], self.N[s])]

            assert len(U) == len(Q) == othello.ACTION_SPACE

            valid_actions = othello.get_valid_actions(state, current_player)

            scores = [u + q for u, q in zip(U, Q)]
            scores = np.array([score if action in valid_actions else -np.inf
                               for action, score in enumerate(scores)])

            best_action = random.choice(np.where(scores == scores.max())[0])

            next_state = self.next_states[s][best_action]

            v = -self._evaluate(next_state, -current_player)

            self.W[s][best_action] += v
            self.N[s][best_action] += 1

            return v
