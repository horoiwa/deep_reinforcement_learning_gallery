from pathlib import Path

import tensorflow as tf
import gym

from network import DualQNetwork, GaussianPolicy, ValueNetwork


def load_dataset(dataset_path: str, batch_size: int):

    def deserialize(serialized_transition):

        transition = tf.io.parse_single_example(
            serialized_transition,
            features={
                'state': tf.io.FixedLenFeature([], tf.string),
                'action': tf.io.FixedLenFeature([], tf.string),
                'reward': tf.io.FixedLenFeature([], tf.float32),
                'next_state': tf.io.FixedLenFeature([], tf.string),
                'done': tf.io.FixedLenFeature([], tf.float32),
            }
        )

        a = tf.io.decode_raw(transition["action"], tf.float32)
        r = transition["reward"]
        d = transition["done"]
        s = tf.io.decode_raw(transition["state"], tf.float32)
        s2 = tf.io.decode_raw(transition["next_state"], tf.float32)

        return s, a, r, s2, d

    dataset = (
        tf.data.TFRecordDataset(filenames=dataset_path, num_parallel_reads=tf.data.AUTOTUNE)
               .shuffle(1024*1024, reshuffle_each_iteration=True)
               .repeat()
               .map(deserialize, num_parallel_calls=tf.data.AUTOTUNE)
               .batch(batch_size, drop_remainder=True)
               .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


class IQLAgent:

    def __init__(self, env_id: str):

        self.env_id = env_id
        self.action_space = gym.make(self.env_id).action_space.shape[0]

        self.expectile = 0.8
        self.temperature = 0.1
        self.tau = 0.005
        self.gamma = 0.99

        self.policy = GaussianPolicy(action_space=self.action_space)
        self.valuenet = ValueNetwork()
        self.qnet = DualQNetwork()
        self.target_qnet = DualQNetwork()

    def setup(self):
        """ Initialize network weights """

        env = gym.make(self.env_id)

        dummy_state = env.reset()
        dummy_state = (dummy_state[np.newaxis, ...]).astype(np.float32)

        dummy_action = np.random.normal(0, 0.1, size=self.action_space)
        dummy_action = (dummy_action[np.newaxis, ...]).astype(np.float32)

        self.policy(dummy_state)

        self.qnet(dummy_state, dummy_action)
        self.target_qnet(dummy_state, dummy_action)
        self.target_qnet.set_weights(self.qnet.get_weights())

        self.valuenet(dummy_state)

    def save(self, save_dir="checkpoints/"):
        save_dir = Path(save_dir)

        self.policy.save_weights(str(save_dir / "policy"))
        self.qnet.save_weights(str(save_dir / "qnet"))
        self.valuenet.save_weights(str(save_dir / "valuenet"))

    def load(self, load_dir="checkpoints/"):
        load_dir = Path(load_dir)

        self.policy.load_weights(str(load_dir / "policy"))
        self.qnet.load_weights(str(load_dir / "qnet"))
        self.target_qnet.load_weights(str(load_dir / "qnet"))
        self.valuenet.load_weights(str(load_dir / "valuenet"))

    def update_value(self, states, actions):
        pass

    def update_policy(self):
        pass

    def update_q(self):
        pass

    def sync_target_weight(self):
        pass

    def test_play(self, monitor_dir, tag):
        total_rewards = []

        env = wrappers.RecordVideo(
            gym.make(self.env_id),
            video_folder=monitor_dir,
            step_trigger=lambda i: True,
            name_prefix=tag
        )

        state = env.reset()

        done = False

        total_reward = 0

        while not done:

            action = self.policy.sample_action(np.atleast_2d(state))

            action = action.numpy()[0]

            next_state, reward, done, _ = env.step(action)

            total_reward += reward

            state = next_state

        total_rewards.append(total_reward)

        print(f"{name}", total_reward)


def main():

    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"   # Needed only for ubuntu

    LOGDIR = Path(__file__).parent / "log"
    if LOGDIR.exists():
        shutil.rmtree(LOGDIR)

    MONITOR_DIR = Path(__file__).parent / "mp4"
    if MONITOR_DIR.exists():
        shutil.rmtree(MONITOR_DIR)

    summary_writer = tf.summary.create_file_writer(str(logdir))

    agent = IQLAgent()

    tf_dataset = load_dataset(dataset_path="bipedalwalker.tfrecord", batch_size=32)

    for n, minibatch in enumerate(tf_dataset):
        states, actions, rewards, next_states, dones = minibatch

        agent.update_value(states, actions)
        agent.update_policy(states, actions)
        agent.update_q(states, actions, rewards, dones)
        agent.sync_target_weight()

        if n % 1000 == 0:
            agent.test_play(tag=f"{n}", monitor_dir=MONITOR_DIR)

    agent.save("checkpoints/")


if __name__ == '__main__':
    main()
