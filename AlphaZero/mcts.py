import math
import random
import json
import time

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

    def search(self, root_state, current_player, num_simulations):

        #: json string is hashable
        s = json.dumps(root_state)

        if s not in self.P:
            _ = self._expand(root_state, current_player)

        #: Adding Dirichlet noise to the prior probabilities in the root node
        dirichlet_noise = np.random.dirichlet(alpha=[self.alpha]*othello.ACTION_SPACE)
        self.P[s] = [(1 - self.eps) * prob + self.eps * noise
                     for prob, noise in zip(self.P[s], dirichlet_noise)]

        valid_actions = othello.get_valid_actions(root_state, current_player)

        #: MCTS simulation
        for _ in range(num_simulations):

            U = [self.c_puct * self.P[s][a] * math.sqrt(sum(self.N[s])) / (1 + self.N[s][a])
                 for a in range(othello.ACTION_SPACE)]
            Q = [w / n if n != 0 else 0 for w, n in zip(self.W[s], self.N[s])]

            score = np.array([score if (action in valid_actions) else -np.inf
                              for action, score in enumerate(U + Q)])

            #: np.argmaxでは同値maxで偏るため
            best_action = random.choice(np.where(score == score.max())[0])

            next_state = self.next_states[s][best_action]

            v = -self._evaluate(next_state, -current_player)

            self.W[s][best_action] += v

            self.N[s][best_action] += 1

        mcts_policy = [n / sum(self.N[s]) for n in self.N[s]]

        return mcts_policy

    def _expand(self, state, current_player):

        s = json.dumps(state)

        #: gpu -> 0.05sec, cpu _> 0.06 - 0.1
        with tf.device("/cpu:0"):
            nn_policy, nn_value = self.network.predict(
                othello.encode_state(state, current_player)[np.newaxis, ...])

        nn_policy, nn_value = nn_policy.numpy().tolist()[0], nn_value.numpy()[0][0]

        self.P[s] = nn_policy
        self.N[s] = [0] * othello.ACTION_SPACE
        self.W[s] = [0] * othello.ACTION_SPACE

        valid_actions = othello.get_valid_actions(state, current_player)

        #: cache valid actions and next state to save computation
        self.next_states[s] = [
            othello.step(state, action, current_player)[0]
            if action in valid_actions else None
            for action in range(othello.ACTION_SPACE)]

        return nn_value

    def _evaluate(self, state, current_player):

        s = json.dumps(state)

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

            valid_actions = othello.get_valid_actions(state, current_player)
            score = np.array([score if (action in valid_actions) else -np.inf
                              for action, score in enumerate(U + Q)])

            best_action = random.choice(np.where(score == score.max())[0])

            next_state = self.next_states[s][best_action]

            v = -self._evaluate(next_state, -current_player)

            self.W[s][best_action] += v
            self.N[s][best_action] += 1

            return v
