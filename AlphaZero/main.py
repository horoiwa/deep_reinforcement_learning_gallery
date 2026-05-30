"""
AlphaZero: 自己対戦強化学習 with 動的盤面サイズ対応

=============================================================================
【基本的な使用方法】
=============================================================================

1. デフォルト設定（6×6盤面, CPU 23, エピソード 10000）で実行:
   $ python main.py

2. 4×4盤面で小規模学習（データ収集のみ、モデル保存なし）:
   $ python main.py --board_size 4 --episodes 5000 --num_cpus 8

2b. 4×4盤面でテスト学習（モデル保存あり）:
   $ python main.py --board_size 4 --episodes 22000 --num_cpus 8

3. 8×8盤面で大規模学習:
   $ python main.py --board_size 8 --episodes 5000 --num_cpus 16

=============================================================================
【引数一覧】
=============================================================================

  --num_cpus CPUS
    使用するCPU数 (default: 23)
    マシンスペックに合わせて調整。通常は物理コア数 - 2 程度が目安

  --board_size SIZE
    盤面サイズ (正方形のみ対応, default: 6)
    4, 6, 8 など偶数の正方形盤面をサポート
    小さいほど学習が高速だが複雑度が低い
    --board-size でも指定可能

  --episodes EPISODES
    総エピソード数 (default: 10000)
    学習の進行度を決定

  --buffer_size SIZE
    リプレイバッファサイズ (default: 40000)
    大きいほど多様な経験を保有するが、メモリ使用量増加

  --batch_size BATCH_SIZE
    ミニバッチサイズ (default: 64)
    グラデイエント計算単位。小さいほどメモリ効率的

  --mcts_simulations SIMS
    MCTSシミュレーション数 (default: 50)
    多いほど探索が深くなるが計算量増加

  --load_checkpoint PATH
    復元するチェックポイントパス (default: None)
    例: checkpoints/run_YYYYMMDD_HHMMSS/network_step_600/network.weights.h5
    指定時は停止したモデルから追加学習を再開

  --resume_logging
    既存のTensorBoardログを保持 (default: False)
    このフラグがない場合、ログディレクトリは初期化される
    追加学習時に前回の学習曲線を保持したい場合に使用

=============================================================================
【推奨ハイパーパラメータ】
=============================================================================

【小規模（デバッグ用, 5-10分）】
$ python main.py --board_size 4 --episodes 100 --buffer_size 2000 \\
    --batch_size 16 --mcts_simulations 20 --num_cpus 4

【中規模（テスト, 1-2時間）】4×4で十分なモデル数を保存して解析したい場合
$ python main.py --board_size 4 --episodes 50000 --buffer_size 10000 \\
    --batch_size 32 --mcts_simulations 30 --num_cpus 8
  ※ 学習開始: buffer_size // 2 = 5000 サンプル
  ※ モデル保存: 学習開始後、300エピソード間隔で蓄積保存

【本格学習（6時間以上）】
$ python main.py --board_size 6 --episodes 10000 --buffer_size 40000 \\
    --batch_size 64 --mcts_simulations 50 --num_cpus 23

=============================================================================
【学習の再開方法】
=============================================================================

1. 初回実行（4×4でモデル保存までテスト）:
   $ python main.py --board_size 4 --episodes 50000 --buffer_size 10000 --num_cpus 8
   ↓
   (学習進行中に Ctrl+C で中断)
   ↓
   学習開始: buffer_size // 2 = 5000 サンプル
   checkpoints/run_YYYYMMDD_HHMMSS/network_step_*/network.weights.h5 以降に自動保存される

2. 途中から再開（新規ログで再開）:
   $ python main.py --board_size 4 --episodes 50000 --buffer_size 10000 --num_cpus 8 \\
       --load_checkpoint checkpoints/run_YYYYMMDD_HHMMSS/network_step_*/network.weights.h5
   ※ ログはリセットされ、TensorBoard上では新規グラフから開始

3. 既存ログを保持しながら再開（推奨）:
   $ python main.py --board_size 4 --episodes 50000 --buffer_size 10000 --num_cpus 8 \\
       --load_checkpoint checkpoints/run_YYYYMMDD_HHMMSS/network_step_*/network.weights.h5 --resume_logging
   ※ TensorBoard上で前回のグラフに追加プロットされる

=============================================================================
【TensorBoard でモニタリング】
=============================================================================

別ターミナルで:
$ tensorboard --logdir=./log

ブラウザで http://localhost:6006 を開く

記録されるメトリクス:
  - v_loss     : Value Head 損失（100ステップごと）
  - p_loss     : Policy Head 損失（100ステップごと）
  - win_count  : テスト勝利数（初期モデル、およびチェックポイント保存ごと）
  - win_ratio  : 勝率（初期モデル、およびチェックポイント保存ごと）
  - buffer_size: リプレイバッファサイズ（初期モデル、およびチェックポイント保存ごと）

=============================================================================
【ファイル構造】
=============================================================================

AlphaZero/
├── main.py              ← このファイル
├── network.py           ← ニューラルネットワーク（AlphaZeroResNet）
├── othello.py           ← ゲームロジック（動的盤面対応）
├── mcts.py              ← モンテカルロ木探索
├── buffer.py            ← リプレイバッファ
├── checkpoints/         ← モデルの歴代保存先（実行ごとのrunディレクトリ配下）
│   └── run_YYYYMMDD_HHMMSS/
│       ├── network_step_600/
│       │   └── network.weights.h5
│       └── ...
├── log/                 ← TensorBoardログ
│   └── events.out.tfevents.xxx
└── img/                 ← テストゲーム結果の画像保存先

=============================================================================
"""

from dataclasses import dataclass
import time
import random
from pathlib import Path
import shutil
import argparse

import tensorflow as tf
import numpy as np
import ray
from tqdm import tqdm

from network import AlphaZeroResNet
from mcts import MCTS
from buffer import ReplayBuffer
import othello


@dataclass
class Sample:

    state: list
    mcts_policy: list
    player: int
    reward: int


def resolve_weights_path(path):
    checkpoint_path = Path(path)
    candidates = [checkpoint_path]

    if checkpoint_path.is_dir():
        candidates.insert(0, checkpoint_path / "network.weights.h5")
    elif checkpoint_path.suffix != ".h5":
        candidates.insert(0, checkpoint_path.with_name(checkpoint_path.name + ".weights.h5"))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


@ray.remote(num_cpus=1, num_gpus=0)
def selfplay(weights, num_mcts_simulations, dirichlet_alpha, board_size):

    othello.set_board_size(board_size, board_size)

    record = []

    state = othello.get_initial_state()

    network = AlphaZeroResNet(action_space=othello.ACTION_SPACE)

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
             dirichlet_alpha=None, n_testplay=24, board_size=6):

    othello.set_board_size(board_size, board_size)

    t = time.time()

    win_count = 0

    network = AlphaZeroResNet(action_space=othello.ACTION_SPACE)

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


def main(num_cpus, n_episodes=10000, buffer_size=40000,
         batch_size=64, epochs_per_update=5,
         num_mcts_simulations=50,
         update_period=300,
         n_testplay=20,
         save_period=300,
         dirichlet_alpha=0.35,
         board_size=6,
         load_checkpoint=None,
         resume_logging=False):
    
    if board_size < 4 or board_size % 2 != 0:
        raise ValueError(f"board_size は4以上の偶数を指定してください: {board_size}")

    # 盤面サイズを設定（正方形のみ）
    othello.set_board_size(board_size, board_size)
    print(f"盤面サイズ: {board_size}×{board_size}, アクション数: {othello.ACTION_SPACE}")
    
    # 学習開始のしきい値を動的に計算（6×6基準: buffer_size=40000 → threshold=20000）
    # つまり、バッファサイズの50%を学習開始のしきい値とする
    learn_threshold = buffer_size // 2
    print(f"学習開始しきい値: {learn_threshold} サンプル（バッファサイズの{100*learn_threshold//buffer_size}%）")

    ray.init(num_cpus=num_cpus, local_mode=False)

    logdir = Path(__file__).parent / "log"
    if logdir.exists() and not resume_logging:
        shutil.rmtree(logdir)
    summary_writer = tf.summary.create_file_writer(str(logdir))

    checkpoint_root = Path(__file__).parent / "checkpoints" / time.strftime("run_%Y%m%d_%H%M%S")
    suffix = 2
    while checkpoint_root.exists():
        checkpoint_root = Path(__file__).parent / "checkpoints" / f"{time.strftime('run_%Y%m%d_%H%M%S')}_{suffix}"
        suffix += 1
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    print(f"チェックポイント保存先: {checkpoint_root}")

    network = AlphaZeroResNet(action_space=othello.ACTION_SPACE)

    #: initialize network parameters
    dummy_state = othello.encode_state(othello.get_initial_state(), 1)

    network.predict(dummy_state)
    
    # チェックポイントから復元
    if load_checkpoint:
        checkpoint_path = resolve_weights_path(load_checkpoint)
        if checkpoint_path.exists():
            try:
                network.load_weights(str(checkpoint_path))
                print(f"✓ チェックポイント読み込み成功: {checkpoint_path}")
            except Exception as e:
                print(f"✗ チェックポイント読み込みエラー: {e}")
                print("新規にネットワークを初期化します")
        else:
            print(f"⚠ チェックポイントが見つかりません: {checkpoint_path}")
            print("新規にネットワークを初期化します")

    current_weights = ray.put(network.get_weights())

    #optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    replay = ReplayBuffer(buffer_size=buffer_size)

    print("初期モデル性能評価")
    win_count, win_ratio, elapsed_time = ray.get(testplay.remote(
        current_weights, num_mcts_simulations,
        n_testplay=n_testplay, board_size=board_size))
    print(f"SCORE: {win_count}, {win_ratio}, Elapsed: {elapsed_time}")
    with summary_writer.as_default():
        tf.summary.scalar("win_count", win_count, step=0)
        tf.summary.scalar("win_ratio", win_ratio, step=0)
        tf.summary.scalar("buffer_size", len(replay), step=0)

    #: 並列Selfplay
    work_in_progresses = [
        selfplay.remote(current_weights, num_mcts_simulations,
                        dirichlet_alpha, board_size)
        for _ in range(num_cpus - 2)]

    n_updates = 0
    n = 0
    while n < n_episodes:

        games_to_collect = min(update_period, n_episodes - n)
        for _ in tqdm(range(games_to_collect)):
            #: selfplayが終わったプロセスを一つ取得
            finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
            replay.add_record(ray.get(finished[0]))
            work_in_progresses.extend([
                selfplay.remote(current_weights, num_mcts_simulations,
                                dirichlet_alpha, board_size)
            ])
            n += 1

        #: Update network
        if len(replay) >= learn_threshold:

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

        #: 学習開始後、save_periodごとにモデルを保存し、同じ重みを性能評価
        if n % save_period == 0 and len(replay) >= learn_threshold:
            checkpoint_dir = checkpoint_root / f"network_step_{n}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            network.save_weights(str(checkpoint_dir / "network.weights.h5"))
            print(f"✓ チェックポイント保存: step_{n}")

            print(f"{n}: TEST")
            win_count, win_ratio, elapsed_time = ray.get(testplay.remote(
                current_weights, num_mcts_simulations,
                n_testplay=n_testplay, board_size=board_size))
            print(f"SCORE: {win_count}, {win_ratio}, Elapsed: {elapsed_time}")
            with summary_writer.as_default():
                tf.summary.scalar("win_count", win_count, step=n)
                tf.summary.scalar("win_ratio", win_ratio, step=n)
                tf.summary.scalar("buffer_size", len(replay), step=n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaZero: 自己対戦強化学習")
    parser.add_argument("--num_cpus", type=int, default=23, help="使用するCPU数 (default: 23)")
    parser.add_argument("--board_size", "--board-size", type=int, default=6, help="盤面サイズ（正方形のみ） (default: 6)")
    parser.add_argument("--episodes", type=int, default=10000, help="エピソード数 (default: 10000)")
    parser.add_argument("--buffer_size", type=int, default=40000, help="リプレイバッファサイズ (default: 40000)")
    parser.add_argument("--batch_size", type=int, default=64, help="バッチサイズ (default: 64)")
    parser.add_argument("--mcts_simulations", type=int, default=50, help="MCTS シミュレーション数 (default: 50)")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="読み込むチェックポイントパス")
    parser.add_argument("--resume_logging", action="store_true", help="既存のログを保持して追加学習 (default: False)")
    
    args = parser.parse_args()
    
    main(
        num_cpus=args.num_cpus,
        n_episodes=args.episodes,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        board_size=args.board_size,
        num_mcts_simulations=args.mcts_simulations,
        load_checkpoint=args.load_checkpoint,
        resume_logging=args.resume_logging
    )
