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

        nn_policy, _ = self.network.predict(
            othello.encode_state(root_state, current_player))

        #: adding Dirichlet noise to the prior probabilities in the root node
        dirichlet_noise = np.random.dirichlet(alpha=[self.alpha]*othello.ACTION_SPACE)
        nn_policy = (1 - self.eps) * nn_policy.numpy()[0] + self.eps * dirichlet_noise

        #: str is hashable
        s = json.dumps(root_state)

        self.P[s] = nn_policy.tolist()
        self.N[s] = [0] * othello.ACTION_SPACE
        self.Q_sum[s] = [0] * othello.ACTION_SPACE

        valid_actions = othello.get_valid_actions(root_state, current_player)

        #: cache valid actions and next state to save computation
        self.valid_actions[s] = valid_actions
        self.next_states[s] = {
            a: othello.get_next_state(root_state, a, current_player)
            if (a in valid_actions) else None
            for a in range(othello.ACTION_SPACE)}

        for _ in num_simulations:

            U = [self.c_puct * self.P[s][a] * math.sqrt(sum(self.N[s])) / (1 + self.N[s][a])
                 for a in range(othello.ACTION_SPACE)]
            Q = [q / n if n != 0 else q for q, n in zip(self.Q_sum[s], self.N[s])]
            import pdb; pdb.set_trace()

            score = [score if (action in valid_actions) else -np.inf
                     for action, score in enumerate(U + Q)]
            selected_action = np.argmax(np.array(score))

            next_state = self.next_states[s][selected_action]

            self.Q_sum[s][selected_action] += -self._evaluate(next_state, -current_player)

    def _evaluate(self, state):
        pass

    def mcts_policy(self, tau):
        pass
