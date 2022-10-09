from pathlib import Path
import random
import numpy as np

from dopamine.replay_memory import circular_replay_buffer


class OfflineReplayBuffer:

    def __init__(self, dataset_dir, num_buffers=50,
                 capacity_of_each_buffer=100000, batch_size=32):
        """
        Args:
            dataset_dir (str): path to dqn-replay-dataset (50M遷移 = 1M * 50 files)
            num_buffer(int): 50分割されたデータセットの何番目まで使うか, set 50 for CQL paper
            maxlen(int): 各bufferあたりのcapacity. set 10,000 for 1% dataset, 100,000 for 10% dataset, max 1,000,000.

        Note:
            #: Download dataset in advance
            mkdir dqn-replay-dataset && cd ./dqn-replay-dataset
            gsutil -m cp -R gs://atari-replay-datasets/dqn/BreakOut .
        """

        self.buffers = []

        self.dataset_dir = Path(dataset_dir)

        self.num_buffers = num_buffers

        self.capacity_of_each_buffer = capacity_of_each_buffer

        self.batch_size = batch_size

        self.reload_dataset()

    def reload_dataset(self):

        assert self.dataset_dir.exists()

        for buffer in self.buffers:
            del buffer

        for i in range(0, self.num_buffers):

            buffer = circular_replay_buffer.OutOfGraphReplayBuffer(
                replay_capacity=self.capacity_of_each_buffer,
                observation_shape=(84, 84),
                stack_size=4,
                batch_size=self.batch_size,
                update_horizon=1,
                gamma=0.99)

            buffer.load(self.dataset_dir, suffix=f"{i}")
            self.buffers.append(buffer)

    def sample_minibatch(self):

        idx = random.randint(0, len(self.buffers)-1)

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
