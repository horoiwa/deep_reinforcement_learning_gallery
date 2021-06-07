import collections

import tensorflow as tf
import numpy as np

from network import AlphaZeroNetwork
from mcts import MCTS
from buffer import ReplayBuffer
import othello


def selfplay(network, num_mcts_simulations, dirichlet_alpha):

    record = []

    GameStep = collections.namedtuple(
        'GameStep', ['state', 'action', 'player', 'result'])

    state = othello.get_initial_state()

    current_player = 1

    i = 0
    done = False
    while not done:

        mcts = MCTS(network=network, alpha=dirichlet_alpha)

        mcts.search(root_state=state, current_player=current_player,
                    num_simulations=num_mcts_simulations)

        # For the first 30 moves of each game, the temperature is set to τ = 1
        if i <= 30:
            mcts_policy = mcts.get_policy(tau=1.0)
        else:
            mcts_policy = mcts.get_policy(tau=0.)

        #: select action according to mcts policy(action probability)
        action = np.random.choice(othello.ACTION_SPACE, p=mcts_policy)

        record.append(GameStep(state, mcts_policy, current_player, None))

        next_state, done = othello.step(state, action, current_player)

        state = next_state

        current_player = -current_player

        i += 1

    #: win: 1, lose: -1, draw: 0
    result_first, result_second = othello.get_results(state)

    #: backup
    for step in reversed(record):
        step.result = result_first if step.player == 1 else result_second

    return record


def main(n_episodes=1000000, buffer_size=30000,
         batch_size=32, n_minibatchs=64,
         num_mcts_simulations=800,
         update_period=25000,
         n_play_for_network_evaluation=400,
         win_ratio_margin=0.55,
         dirichlet_alpha=0.15,
         lr=0.05):

    network = AlphaZeroNetwork(action_space=othello.ACTION_SPACE)

    optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=0.9)

    replay = ReplayBuffer(buffer_size=buffer_size)

    n = 0

    while n <= n_episodes:

        #: collect samples by selfplay
        for _ in range(update_period):
            game_record = selfplay(network, num_mcts_simulations, dirichlet_alpha)
            replay.add_record(game_record)
            n += 1

        #: update network
        current_weights = network.get_weights()
        minibatchs = [replay.get_minibatch(batch_size=batch_size)
                      for _ in range(n_minibatchs)]

        new_weights = network.get_weights()

        #: current network vs. new network


if __name__ == "__main__":
    main()
