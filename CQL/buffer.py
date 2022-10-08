from pathlib import Path
import random
import numpy as np

from dopamine.replay_memory import circular_replay_buffer


class OfflineReplayBuffer:

    def __init__(self, dataset_dir, capacity=500000, num_ckpt_files=50, batch_size=32):
        """
        CQL論文の設定である NUM_CKPT_FILE=50, capacity 0.1%/1.0% ではメモリに乗らないので、
        データセットサイズを0.1%まで落とす代わりに試行錯誤が豊富に含まれる最初の5ファイルを使用

        Args:
            dataset_dir (str): path to dqn-replay-dataset (50M遷移 = 1M * 50 files)
            num_ckpt_files (int)): 50分割されたデータセットの何番目まで使うか(CQL論文を再現するなら50）
            capacity (int): set 500,000 for 1% dataset, 5,000,000 for 10% dataset, 50,000,000 for full dataset.

        Note:
            #: Download dataset in advance
            mkdir dqn-replay-dataset && cd ./dqn-replay-dataset
            gsutil -m cp -R gs://atari-replay-datasets/dqn/BreakOut .
        """

        self.buffers = []

        self.dataset_dir = Path(dataset_dir)

        self.num_ckpt_files = num_ckpt_files

        self.capacity = capacity

        self.batch_size = batch_size

        self._load_dataset()

    def _load_dataset(self):

        assert self.dataset_dir.exists()

        capacity_per_buffer = self.capacity // self.num_ckpt_files

        for i in range(0, self.num_ckpt_files):

            buffer = circular_replay_buffer.OutOfGraphReplayBuffer(
                replay_capacity=capacity_per_buffer,
                observation_shape=(84, 84),
                stack_size=4,
                batch_size=self.batch_size,
                update_horizon=1,
                gamma=0.99)

            buffer.load(self.dataset_dir, suffix=f"{i}")

            self.buffers.append(buffer)

    def sample_minibatch(self):

        idx = random.randint(0, self.num_ckpt_files-1)

        (states, actions, rewards, next_states,
         _, _, dones, _) = self.buffers[idx].sample_transition_batch()

        states = states.astype(np.float32)
        next_states = next_states.astype(np.float32)
        dones = dones.astype(np.float32)

        return (states, actions, rewards, next_states, dones)


if __name__ == '__main__':
    dataset_dir = "dqn-replay-dataset/Breakout/1/replay_logs"
    buffer = OfflineReplayBuffer(dataset_dir=dataset_dir)
    buffer.sample_minibatch()
