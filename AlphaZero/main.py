from dataclasses import dataclass
import time
import random
from pathlib import Path
import shutil

import tensorflow as tf
import numpy as np
import ray
from tqdm import tqdm

from network import SimpleCNN as AlphaZeroNetwork
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

        mcts_policy = mcts.search(root_state=state,
                                  current_player=current_player,
                                  num_simulations=num_mcts_simulations)

        if i <= 10:
            # For the first 30 moves of each game, the temperature is set to τ = 1;
            # this selects moves proportionally to their visit count in MCTS
            action = np.random.choice(
                range(othello.ACTION_SPACE), p=mcts_policy)
        else:
            action = random.choice(
                np.where(np.array(mcts_policy) == max(mcts_policy))[0])

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

    win_count = 0

    network = AlphaZeroNetwork(action_space=othello.ACTION_SPACE)

    dummy_state = othello.get_initial_state()

    network.predict(othello.encode_state(dummy_state, 1))

    network.set_weights(current_weights)

    for n in range(n_testplay):

        alphazero = random.choice([1, -1])

        mcts = MCTS(network=network, alpha=dirichlet_alpha)

        state = othello.get_initial_state()

        current_player = 1

        done = False

        while not done:

            if current_player == alphazero:
                mcts_policy = mcts.search(root_state=state,
                                          current_player=current_player,
                                          num_simulations=num_mcts_simulations)
                action = np.argmax(mcts_policy)
            else:
                action = othello.greedy_action(state, current_player, epsilon=0.3)

            next_state, done = othello.step(state, action, current_player)

            state = next_state

            current_player = -1 * current_player

        reward_first, reward_second = othello.get_result(state)

        reward = reward_first if alphazero == 1 else reward_second
        result = "win" if reward == 1 else "lose" if reward == -1 else "draw"

        if reward > 0:
            win_count += 1

        stone_first, stone_second = othello.count_stone(state)

        if alphazero == 1:
            stone_az, stone_tester = stone_first, stone_second
            color = "black"
        else:
            stone_az, stone_tester = stone_second, stone_first
            color = "white"

        message = f"AlphaZero ({color}) {result}: {stone_az} vs {stone_tester}"

        othello.save_img(state, "img", f"test_{n}.png", message)

    elapsed = time.time() - t

    return win_count, win_count / n_testplay, elapsed


def main(num_cpus, n_episodes=30000, buffer_size=40000,
         batch_size=64, epochs_per_update=5,
         num_mcts_simulations=50,
         update_period=300, test_period=300,
         n_testplay=20,
         save_period=3000,
         dirichlet_alpha=0.35):

    ray.init(num_cpus=num_cpus, num_gpus=1, local_mode=False)

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
        current_weights, num_mcts_simulations, n_testplay=n_testplay)

    n_updates = 0
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
        #if len(replay) >= 2000:

            num_iters = epochs_per_update * (len(replay) // batch_size)
            for i in range(num_iters):

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

                n_updates += 1

                if i % 100 == 0:
                    with summary_writer.as_default():
                        tf.summary.scalar("v_loss", value_loss.numpy().mean(), step=n_updates)
                        tf.summary.scalar("p_loss", policy_loss.numpy().mean(), step=n_updates)

            current_weights = ray.put(network.get_weights())

        if n % test_period == 0:
            print(f"{n - test_period}: TEST")
            win_count, win_ratio, elapsed_time = ray.get(test_in_progress)
            print(f"SCORE: {win_count}, {win_ratio}, Elapsed: {elapsed_time}")
            test_in_progress = testplay.remote(
                current_weights, num_mcts_simulations, n_testplay=n_testplay)

            with summary_writer.as_default():
                tf.summary.scalar("win_count", win_count, step=n-test_period)
                tf.summary.scalar("win_ratio", win_ratio, step=n-test_period)
                tf.summary.scalar("buffer_size", len(replay), step=n)

        if n % save_period == 0:
            network.save_weights("checkpoints/network")


if __name__ == "__main__":
    main(num_cpus=23)
