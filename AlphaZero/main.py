import collections
from dataclasses import dataclass
import time
import random

import tensorflow as tf
import numpy as np
import ray
from tensorflow.python.eager.context import num_gpus

from network import AlphaZeroNetwork
from mcts import MCTS
from buffer import ReplayBuffer
import othello


@dataclass
class Sample:

    state: list
    mcts_policy: list
    player: int
    reward: int


@ray.remote(num_cpus=1, num_gpus=0)
def selfplay(weights, num_mcts_simulations, dirichlet_alpha):

    record = []

    state = othello.get_initial_state()

    network = AlphaZeroNetwork(action_space=othello.ACTION_SPACE)

    network.call(othello.encode_state(state, 1)[np.newaxis, ...])

    network.set_weights(weights)

    mcts = MCTS(network=network, alpha=dirichlet_alpha)

    current_player = 1

    done = False

    i = 0

    while not done:

        #: 200 simulations: GTX 1650 -> 4.6sec, 1CPU -> 8.8sec
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

    #: win: 1, lose: -1, draw: 0
    reward_first, reward_second = othello.get_result(state)

    for sample in reversed(record):
        sample.reward = reward_first if sample.player == 1 else reward_second

    return record


def main(num_workers, n_episodes=1000000, buffer_size=30000,
         batch_size=32, n_minibatchs=64,
         num_mcts_simulations=100,
         update_period=250, dirichlet_alpha=0.15,
         lr=0.05, c=1e-4):

    ray.init(num_cpus=num_workers+2, num_gpus=1)

    network = AlphaZeroNetwork(action_space=othello.ACTION_SPACE)

    #: initialize network parameters
    dummy_state = othello.encode_state(othello.get_initial_state(), 1)[np.newaxis, ...]

    network.call(dummy_state)

    current_weights = ray.put(network.get_weights())

    #optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=0.9)
    optimizer = tf.keras.optimizers.Adam(lr=0.00025)

    replay = ReplayBuffer(buffer_size=buffer_size)

    #: 並列Selfplay
    work_in_progresses = [
        selfplay.remote(current_weights, num_mcts_simulations, dirichlet_alpha)
        for _ in range(num_workers)]

    n = 0

    while n <= n_episodes:

        #for _ in range(update_period):
        for _ in range(3):
            #: selfplayが終わったプロセスを一つ取得
            record, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
            replay.add(ray.get(record[0]))
            work_in_progresses.extend([
                selfplay.remote(network, num_mcts_simulations, dirichlet_alpha)
            ])
            n += 1

        minibatchs = [replay.get_minibatch(batch_size=batch_size)
                      for _ in range(n_minibatchs)]

        #: Update network
        for (states, mcts_policy, rewards) in minibatchs:

            with tf.GradientTape() as tape:
                import pdb; pdb.set_trace()
                p_pred, v_pred = network(states, training=True)
                value_loss = tf.square(rewards, v_pred)

                policy_loss = -mcts_policy * tf.math.log(p_pred + 0.0001)
                policy_loss = tf.reduce_sum(
                    policy_loss, axis=1, keepdims=True)

                l2_weight = None

                loss = value_loss + policy_loss + c * l2_weight

            grads = tape.gardient(loss, network.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, network.trainable_variables))

        current_weights = ray.put(network.get_weights())



if __name__ == "__main__":
    main(num_workers=3)
