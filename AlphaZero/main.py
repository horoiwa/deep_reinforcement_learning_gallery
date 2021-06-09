import collections
from dataclasses import dataclass
import time
import random

import tensorflow as tf
import numpy as np
import ray
from tqdm import tqdm

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

    network.predict(othello.encode_state(state, 1))

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


@ray.remote(num_cpus=1, num_gpus=0)
def testplay(current_weights, num_mcts_simulations, dirichlet_alpha, n_testplay=3):
    """石の数の差がスコア"""

    scores = []

    network = AlphaZeroNetwork(action_space=othello.ACTION_SPACE)

    dummy_state = othello.get_initial_state()

    network.predict(othello.encode_state(dummy_state, 1))

    network.set_weights(current_weights)
    t = time.time()

    for n in range(n_testplay):

        alphazero = random.choice([1, -1])

        state = othello.get_initial_state()

        current_player = 1

        done = False

        while not done:

            mcts = MCTS(network=network, alpha=dirichlet_alpha)


            if current_player == alphazero:
                mcts_policy = mcts.search(root_state=state,
                                          current_player=current_player,
                                          num_simulations=num_mcts_simulations)
                action = np.argmax(mcts_policy)
            else:
                action = othello.greedy_action(state, current_player, epsilon=0.2)

            print(action)
            next_state, done = othello.step(state, action, current_player)

            state = next_state

            current_player = -1 * current_player

        stone_first, stone_second = othello.count_stone(state)

        stone_diff = stone_first - stone_second
        score = stone_diff if alphazero == 1 else -stone_diff
        print(score)
        scores.append(score)

    print(time.time() - t)
    average_score = sum(scores) / n_testplay

    return average_score


def main(num_cpus, n_episodes=10000, buffer_size=10000,
         batch_size=32, n_minibatchs=64,
         num_mcts_simulations=50,
         update_period=100, test_period=300,
         dirichlet_alpha=0.15,
         lr=0.05, c=1e-4):

    ray.init(num_cpus=num_cpus, num_gpus=1)

    network = AlphaZeroNetwork(action_space=othello.ACTION_SPACE)

    #: initialize network parameters
    dummy_state = othello.encode_state(othello.get_initial_state(), 1)

    network.predict(dummy_state)

    current_weights = ray.put(network.get_weights())

    #optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=0.9)
    optimizer = tf.keras.optimizers.Adam(lr=0.00025)

    replay = ReplayBuffer(buffer_size=buffer_size)

    #: 並列Selfplay
    work_in_progresses = [
        selfplay.remote(current_weights, num_mcts_simulations, dirichlet_alpha)
        for _ in range(num_cpus - 2)]

    test_in_progress = testplay.remote(
        current_weights, num_mcts_simulations, dirichlet_alpha)

    n = 0

    while n <= n_episodes:

        for _ in tqdm(range(update_period)):
            #: selfplayが終わったプロセスを一つ取得
            record, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
            replay.add_record(ray.get(record[0]))
            work_in_progresses.extend([
                selfplay.remote(current_weights, num_mcts_simulations, dirichlet_alpha)
            ])
            n += 1

        minibatchs = [replay.get_minibatch(batch_size=batch_size)
                      for _ in range(n_minibatchs)]

        #: Update network
        for (states, mcts_policy, rewards) in minibatchs:

            with tf.GradientTape() as tape:

                p_pred, v_pred = network(states, training=True)
                value_loss = tf.square(rewards - v_pred)

                policy_loss = -mcts_policy * tf.math.log(p_pred + 0.0001)
                policy_loss = tf.reduce_sum(
                    policy_loss, axis=1, keepdims=True)

                loss = tf.reduce_mean(value_loss + policy_loss)

            grads = tape.gradient(loss, network.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, network.trainable_variables))

        current_weights = ray.put(network.get_weights())

        if n % test_period == 0:
            print(f"{n - test_period}: TEST")
            test_score = ray.get(test_in_progress)
            print(f"TEST SCORE: {test_score}")
            test_in_progress = testplay.remote(
                current_weights, num_mcts_simulations, dirichlet_alpha)


if __name__ == "__main__":
    main(num_cpus=5)
