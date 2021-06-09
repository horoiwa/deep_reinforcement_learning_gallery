import collections
from dataclasses import dataclass
import time
import random

import tensorflow as tf
import numpy as np

from network import AlphaZeroNetwork
from mcts import MCTS
from buffer import ReplayBuffer
import othello


@dataclass
class Sample:

    state: list
    mcts_policy: list
    player: int
    result: int


def selfplay(network, num_mcts_simulations, dirichlet_alpha):

    record = []

    mcts = MCTS(network=network, alpha=dirichlet_alpha)

    state = othello.get_initial_state()

    current_player = 1

    i = 0
    done = False
    while not done:

        #: 200 simulations: GTX 1650 -> 4.6sec, 1CPU -> sec
        mcts_policy = mcts.search(root_state=state,
                                  current_player=current_player,
                                  num_simulations=num_mcts_simulations)

        if i <= 30:
            # For the first 30 moves of each game, the temperature is set to τ = 1;
            # this selects moves proportionally to their visit count in MCTS
            action = np.random.choice(range(othello.ACTION_SPACE), p=mcts_policy)
        else:
            action = np.argmax(mcts_policy)

        record.append(Sample(state, mcts_policy, current_player, None))

        next_state, done = othello.step(state, action, current_player)

        state = next_state

        current_player = -current_player

        i += 1

        print(i, action, mcts_policy)

    #: win: 1, lose: -1, draw: 0
    result_first, result_second = othello.get_result(state)

    #: backup
    for sample in reversed(record):
        sample.result = result_first if sample.player == 1 else result_second

    return record


def main(n_episodes=1000000, buffer_size=30000,
         batch_size=32, n_minibatchs=64,
         num_mcts_simulations=200,
         update_period=25000,
         n_play_for_network_evaluation=400,
         win_ratio_margin=0.55,
         dirichlet_alpha=0.15,
         lr=0.05):

    network = AlphaZeroNetwork(action_space=othello.ACTION_SPACE)
    #: initialize network parameters
    dummy_state = othello.encode_state(othello.get_initial_state(), 1)[np.newaxis, ...]
    network.call(dummy_state)

    replay = ReplayBuffer(buffer_size=buffer_size)

    optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=0.9)

    n = 0

    while n <= n_episodes:

        #: collect samples by selfplay
        for _ in range(update_period):
            #: 自己対戦でデータ収集
            record = selfplay(network, num_mcts_simulations, dirichlet_alpha)
            replay.add_record(record)
            n += 1

        #: update network
        old_weights = network.get_weights()
        minibatchs = [replay.get_minibatch(batch_size=batch_size)
                      for _ in range(n_minibatchs)]

        for minibacth in minibatchs:
            pass

        new_weights = network.get_weights()

        #: old network vs. updated network
        win_ratio = None
        if win_ratio < win_ratio_margin:
            print("old parameter win")
            network.set_weights(old_weights)
        else:
            print("new paramter win")


if __name__ == "__main__":
    main()
