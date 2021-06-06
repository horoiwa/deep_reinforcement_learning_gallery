from AlphaZero.othello.network import AlphaZeroNetwork
import collections

import tensorflow as tf
import numpy as np

from network import AlaphaZeroNetwork
from mcts import MCTS
from buffer import ReplayBuffer
import othello


def selfplay(network, num_mcts_search, dirichlet_alpha):

    record = []

    GameStep = collections.namedtuple(
        'GameStep', ['state', 'action', 'player', 'result'])

    state = othello.get_initial_state()

    current_player = 1

    i = 0

    while True:

        mcts = MCTS(network=network, root=state)

        mcts.search(num_search=num_mcts_search)

        # For the first 30 moves of each game, the temperature is set to τ = 1
        if i <= 30:
            mcts_policy = mcts.get_policy(tau=1.0, alpha=dirichlet_alpha)
        else:
            mcts_policy = mcts.get_policy(tau=0)

        if mcts_policy is not None:
            #: select action according to mcts policy(action probability)
            action = np.random.choice(othello.ACTION_SPACE, p=mcts_policy)

            record.append(GameStep(state, mcts_policy, current_player, None))

            state = othello.get_next_state(state, action, current_player)

            current_player = -current_player

        else:
            break  # game end (合法手なし)

        i += 1

    #: win: 1, lose: -1, draw: 0
    result_first, result_second = othello.get_results(state)

    #: backup
    for step in reversed(record):
        step.result = result_first if step.player == 1 else result_second

    return record


def train(n_episodes=1000000, buffer_size=30000,
          batch_size=32, n_minibatchs=64,
          num_mcts_search=800,
          update_period=25000,
          n_play_for_network_evaluation=400,
          win_ratio_margin=0.55,
          dirichlet_alpha=0.15,
          c=1e-4, lr=0.05):

    network = AlaphaZeroNetwork(action_space=othello.ACTION_SPACE)

    optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=0.9)

    replay = ReplayBuffer(buffer_size=buffer_size)

    n = 0

    while n <= n_episodes:

        #: collect samples by selfplay
        for _ in range(update_period):
            game_record = selfplay(network, num_mcts_search, dirichlet_alpha)
            replay.add_record(game_record)
            n += 1

        #: update network
        current_weights = network.get_weights()
        minibatchs = [replay.get_minibatch(batch_size=batch_size)
                      for _ in range(n_minibatchs)]

        new_weights = network.get_weights()

        #: current network vs. new network


if __name__ == "__main__":
    train()
