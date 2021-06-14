from dataclasses import dataclass
import time
import random
from pathlib import Path
import shutil

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

        if i <= 10:
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
def testplay(current_weights, num_mcts_simulations,
             dirichlet_alpha=None, n_testplay=24):

    t = time.time()

    scores = []

    network = AlphaZeroNetwork(action_space=othello.ACTION_SPACE)

    dummy_state = othello.get_initial_state()

    network.predict(othello.encode_state(dummy_state, 1))

    network.set_weights(current_weights)

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
                action = othello.greedy_action(state, current_player, epsilon=0.1)

            next_state, done = othello.step(state, action, current_player)

            state = next_state

            current_player = -1 * current_player

        stone_first, stone_second = othello.count_stone(state)

        stone_diff = stone_first - stone_second
        score = stone_diff if alphazero == 1 else -stone_diff
        scores.append(score)

    average_score = sum(scores) / n_testplay

    win_ratio = sum([1 if score > 0 else 0 for score in scores]) / len(scores)

    elapsed = time.time() - t

    return average_score, win_ratio, elapsed


def main(num_cpus, n_episodes=50000, buffer_size=150000,
         batch_size=64, epochs_per_update=5,
         num_mcts_simulations=30,
         update_period=400, test_period=400, save_period=1000,
         dirichlet_alpha=0.15):

    ray.init(num_cpus=num_cpus, num_gpus=1)

    logdir = Path(__file__).parent / "log"
    if logdir.exists():
        shutil.rmtree(logdir)
    summary_writer = tf.summary.create_file_writer(str(logdir))

    network = AlphaZeroNetwork(action_space=othello.ACTION_SPACE)

    #: initialize network parameters
    dummy_state = othello.encode_state(othello.get_initial_state(), 1)

    network.predict(dummy_state)

    current_weights = ray.put(network.get_weights())

    #optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=0.9)
    optimizer = tf.keras.optimizers.Adam(lr=0.0005)

    replay = ReplayBuffer(buffer_size=buffer_size)

    #: 並列Selfplay
    work_in_progresses = [
        selfplay.remote(current_weights, num_mcts_simulations, dirichlet_alpha)
        for _ in range(num_cpus - 2)]

    test_in_progress = testplay.remote(
        current_weights, num_mcts_simulations)

    t = time.time()

    n = 0

    while n <= n_episodes:

        for _ in tqdm(range(update_period)):
            #: selfplayが終わったプロセスを一つ取得
            finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
            replay.add_record(ray.get(finished[0]))
            work_in_progresses.extend([
                selfplay.remote(current_weights, num_mcts_simulations, dirichlet_alpha)
            ])
            n += 1

        #: Update network
        if len(replay) >= 20000:

            stats_vloss = []
            stats_ploss = []

            num_iters = epochs_per_update * (len(replay) // batch_size)
            for _ in range(num_iters):

                states, mcts_policy, rewards = replay.get_minibatch(batch_size=batch_size)

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

                stats_ploss.append(policy_loss.numpy().mean())
                stats_vloss.append(value_loss.numpy().mean())

            elapsed_time = time.time() - t

            t = time.time()

            vloss_mean = sum(stats_vloss) / len(stats_vloss)
            ploss_mean = sum(stats_ploss) / len(stats_ploss)

            with summary_writer.as_default():
                tf.summary.scalar("v_loss", vloss_mean, step=n)
                tf.summary.scalar("p_loss", ploss_mean, step=n)
                tf.summary.scalar("buffer_size", len(replay), step=n)
                tf.summary.scalar("Elapsed time", elapsed_time, step=n)

            current_weights = ray.put(network.get_weights())

        if n % test_period == 0:
            print(f"{n - test_period}: TEST")
            test_score, win_ratio, elapsed_time = ray.get(test_in_progress)
            print(f"SCORE: {test_score}, {win_ratio}, Elapsed: {elapsed_time}")
            test_in_progress = testplay.remote(
                current_weights, num_mcts_simulations)

            with summary_writer.as_default():
                tf.summary.scalar("test_score", test_score, step=n-test_period)
                tf.summary.scalar("test_winratio", win_ratio, step=n-test_period)

        if n % save_period == 0:
            network.save_weights("checkpoints/network")


if __name__ == "__main__":
    main(num_cpus=23)
