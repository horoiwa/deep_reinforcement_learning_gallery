import math
import random
import json

import numpy as np
import tensorflow as tf

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

        #: W is total action-value and Q is mean action-value
        self.W = {}

        #: cache next states to save computation
        self.next_states = {}

        #: string is hashable
        self.state_to_str = (
            lambda state, player: json.dumps(state) + str(player)
            )

    def search(self, root_state, current_player, num_simulations):

        s = self.state_to_str(root_state, current_player)

        if s not in self.P:
            _ = self._expand(root_state, current_player)

        valid_actions = othello.get_valid_actions(root_state, current_player)

        #: Adding Dirichlet noise to the prior probabilities in the root node
        if self.alpha is not None:
            dirichlet_noise = np.random.dirichlet(alpha=[self.alpha]*len(valid_actions))
            for a, noise in zip(valid_actions, dirichlet_noise):
                self.P[s][a] = (1 - self.eps) * self.P[s][a] + self.eps * noise

        #: MCTS simulation
        for _ in range(num_simulations):

            U = [self.c_puct * self.P[s][a] * math.sqrt(sum(self.N[s])) / (1 + self.N[s][a])
                 for a in range(othello.ACTION_SPACE)]
            Q = [w / n if n != 0 else 0 for w, n in zip(self.W[s], self.N[s])]

            assert len(U) == len(Q) == othello.ACTION_SPACE

            scores = [u + q for u, q in zip(U, Q)]

            #: Mask invalid actions
            scores = np.array([score if action in valid_actions else -np.inf
                               for action, score in enumerate(scores)])

            #: np.argmaxでは同値maxで偏るため
            action = random.choice(np.where(scores == scores.max())[0])

            next_state = self.next_states[s][action]

            v = -self._evaluate(next_state, -current_player)

            self.W[s][action] += v

            self.N[s][action] += 1

        mcts_policy = [n / sum(self.N[s]) for n in self.N[s]]

        return mcts_policy

    def _expand(self, state, current_player):

        s = self.state_to_str(state, current_player)

        with tf.device("/cpu:0"):
            nn_policy, nn_value = self.network.predict(
                othello.encode_state(state, current_player))

        nn_policy, nn_value = nn_policy.numpy().tolist()[0], nn_value.numpy()[0][0]

        self.P[s] = nn_policy
        self.N[s] = [0] * othello.ACTION_SPACE
        self.W[s] = [0] * othello.ACTION_SPACE

        valid_actions = othello.get_valid_actions(state, current_player)

        #: cache valid actions and next state to save computation
        self.next_states[s] = [
            othello.step(state, action, current_player)[0]
            if (action in valid_actions) else None
            for action in range(othello.ACTION_SPACE)]

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
