from AlphaZero.othello import ACTION_SPACE
import math
import json

import numpy as np
import othello


class MCTS:

    def __init__(self, network, alpha, c_puct=1.0, epsilon=0.25):

        self.network = network

        self.alpha = alpha

        self.c_puct = c_puct

        self.eps = epsilon

        #: prior probability
        self.P = {}

        #: visit count
        self.N = {}

        #: cumsum of each evaluation of Q(s, a)
        self.Q_sum = {}

        #: cache valid actions and next states to save computation
        self.valid_actions = {}
        self.next_states = {}

    def search(self, root_state, current_player, num_simulations):

        #: str is hashable
        s = json.dumps(root_state)

        if s not in self.P:
            _ = self._expand(root_state, current_player)

        #: Adding Dirichlet noise to the prior probabilities in the root node
        dirichlet_noise = np.random.dirichlet(alpha=[self.alpha]*othello.ACTION_SPACE)
        self.P[s] = [(1 - self.eps) * prob + self.eps * noise
                     for prob, noise in zip(self.P[s], dirichlet_noise)]

        valid_actions = self.valid_actions[s]

        for _ in num_simulations:

            U = [self.c_puct * self.P[s][a] * math.sqrt(sum(self.N[s])) / (1 + self.N[s][a])
                 for a in range(othello.ACTION_SPACE)]
            Q = [q / n if n != 0 else q for q, n in zip(self.Q_sum[s], self.N[s])]

            score = [score if (action in valid_actions) else -np.inf
                     for action, score in enumerate(U + Q)]
            selected_action = np.argmax(np.array(score))

            next_state = self.next_states[s][selected_action]

            self.Q_sum[s][selected_action] += -self._evaluate(next_state, -current_player)

        mcts_policy = None

        return mcts_policy

    def _expand(self, state, current_player):

        s = json.dumps(state)

        nn_policy, nn_value = self.network.predict(
            othello.encode_state(state, current_player)).numpy()[0].tolist()

        self.P[s] = nn_policy
        self.N[s] = [0] * othello.ACTION_SPACE
        self.Q_sum[s] = [0] * othello.ACTION_SPACE

        valid_actions = othello.get_valid_actions(state, current_player)

        #: cache valid actions and next state to save computation
        self.valid_actions[s] = valid_actions
        self.next_states[s] = {
            a: othello.get_next_state(state, a, current_player)
            if (a in valid_actions) else None
            for a in range(othello.ACTION_SPACE)}

        return nn_value

    def _evaluate(self, state, current_player):

        s = json.dumps(state)

        #: ゲーム終了判定

        if s not in self.P:
            #: leaf node
            nn_value = self._expand(state, current_player)
            return nn_value

