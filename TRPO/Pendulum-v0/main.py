from dataclasses import dataclass
import collections
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import numpy as np
import tensorflow as tf
import tensorflow_probability
import gym
from gym import wrappers
import matplotlib.pyplot as plt

from buffer import ReplayBuffer
from models import PolicyNetwork, ValueNetwork
from util import compute_logprob, compute_kl, cg, restore_shape


class TRPOAgent:

    TRAJECTORY_SIZE = 1024

    VF_BATCHSIZE = 64

    MAX_KL = 0.01

    GAMMA = 0.99

    GAE_LAMBDA = 0.98

    ENV_ID = "Pendulum-v0"

    OBS_SPACE = 3

    ACTION_SPACE = 1

    def __init__(self):

        self.policy = PolicyNetwork(action_space=self.ACTION_SPACE)

        self.value_network = ValueNetwork()

        self.env = gym.make(self.ENV_ID)

        self.global_steps = 0

        self.history = []

        self.hiscore = None

    def play(self, n_iters):

        self.epi_reward = 0

        self.epi_steps = 0

        self.state = self.env.reset()

        for _ in range(n_iters):

            trajectory = self.generate_trajectory()

            trajectory = self.compute_advantage(trajectory)

            self.update_policy(trajectory)

            self.update_vf(trajectory)

        return self.history

    def generate_trajectory(self):
        """generate trajectory on current policy
        """

        trajectory = {"s": np.zeros((self.TRAJECTORY_SIZE, self.OBS_SPACE), dtype=np.float32),
                      "a": np.zeros((self.TRAJECTORY_SIZE, self.ACTION_SPACE), dtype=np.float32),
                      "r": np.zeros((self.TRAJECTORY_SIZE, 1), dtype=np.float32),
                      "s2": np.zeros((self.TRAJECTORY_SIZE, self.OBS_SPACE), dtype=np.float32),
                      "done": np.zeros((self.TRAJECTORY_SIZE, 1), dtype=np.float32)}

        state = self.state

        for i in range(self.TRAJECTORY_SIZE):

            action = self.policy.sample_action(state)

            next_state, reward, done, _ = self.env.step(action)

            trajectory["s"][i] = state

            trajectory["a"][i] = action

            trajectory["r"][i] = reward

            trajectory["s2"][i] = next_state

            trajectory["done"][i] = done

            self.epi_reward += reward

            self.epi_steps += 1

            self.global_steps += 1

            if done:
                state = self.env.reset()

                self.history.append(self.epi_reward)

                recent_score = sum(self.history[-10:]) / 10

                print("===="*5)
                print("Episode:", len(self.history))
                print("Episode reward:", self.epi_reward)
                print("Global steps:", self.global_steps)

                if len(self.history) > 100 and (self.hiscore is None or recent_score > self.hiscore):
                    print("*HISCORE UPDATED:", recent_score)
                    self.save_model()
                    self.hiscore = recent_score

                self.epi_reward = 0

                self.epi_steps = 0

            else:
                state = next_state

        self.state = state

        return trajectory

    def compute_advantage(self, trajectory):
        """Compute

        Args:
            trajectory ([type]): [description]
        """

        trajectory["vpred"] = self.value_network(trajectory["s"]).numpy()

        trajectory["vpred_next"] = self.value_network(trajectory["s2"]).numpy()

        is_nonterminals = 1 - trajectory["done"]

        deltas = trajectory["r"] + self.GAMMA * is_nonterminals * trajectory["vpred_next"] - trajectory["vpred"]

        advantages = np.zeros_like(deltas, dtype=np.float32)

        lastgae = 0
        for i in reversed(range(len(deltas))):
            lastgae = deltas[i] + self.GAMMA * self.GAE_LAMBDA * is_nonterminals[i] * lastgae
            advantages[i] = lastgae

        trajectory["adv"] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        #trajectory["adv"] = advantages

        trajectory["vftarget"] = trajectory["adv"] + trajectory["vpred"]

        return trajectory

    def update_policy(self, trajectory):

        def flattengrads(grads):
            flatgrads_list = [tf.reshape(grad, shape=[1, -1]) for grad in grads]
            flatgrads = tf.concat(flatgrads_list, axis=1)
            return flatgrads

        actions = tf.convert_to_tensor(trajectory["a"], dtype=tf.float32)
        states = tf.convert_to_tensor(trajectory["s"], dtype=tf.float32)
        advantages = tf.convert_to_tensor(trajectory["adv"], dtype=tf.float32)

        old_means, old_stdevs = self.policy(states)
        old_logp = compute_logprob(old_means, old_stdevs, actions)

        with tf.GradientTape() as tape:
            new_means, new_stdevs = self.policy(states)
            new_logp = compute_logprob(new_means, new_stdevs, actions)

            loss = tf.exp(new_logp - old_logp) * advantages
            loss = tf.reduce_mean(loss)

        g = tape.gradient(loss, self.policy.trainable_variables)
        g = tf.transpose(flattengrads(g))

        @tf.function
        def hvp_func(vector):
            """Compute hessian-vector product
            """
            with tf.GradientTape() as t2:
                with tf.GradientTape() as t1:
                    new_means, new_stdevs = self.policy(states)
                    kl = compute_kl(old_means, old_stdevs, new_means, new_stdevs)
                    meankl = tf.reduce_mean(kl)

                kl_grads = t1.gradient(meankl, self.policy.trainable_variables)
                kl_grads = flattengrads(kl_grads)
                grads_vector_product = tf.matmul(kl_grads, vector)

            hvp = t2.gradient(grads_vector_product, self.policy.trainable_variables)
            hvp = tf.transpose(flattengrads(hvp))

            return hvp + vector * 1e-2 #: 共役勾配法の安定化のために微小量を加える

        step_direction = cg(hvp_func, g)

        shs = tf.matmul(tf.transpose(step_direction), hvp_func(step_direction))
        lm = tf.sqrt(2 * self.MAX_KL / shs)
        fullstep = lm * step_direction

        expected_improve = tf.matmul(tf.transpose(g), fullstep)
        fullstep = restore_shape(fullstep, self.policy.trainable_variables)

        params_old = [var.numpy() for var in self.policy.trainable_variables]
        old_loss = loss

        for stepsize in [0.5 ** i for i in range(10)]:
            params_new = [p + step * stepsize for p, step in zip(params_old, fullstep)]
            self.policy.set_weights(params_new)

            new_means, new_stdevs = self.policy(states)
            new_logp = compute_logprob(new_means, new_stdevs, actions)

            new_loss = tf.reduce_mean(tf.exp(new_logp - old_logp) * advantages)
            improve = new_loss - old_loss

            kl = compute_kl(old_means, old_stdevs, new_means, new_stdevs)
            mean_kl = tf.reduce_mean(kl)

            print(f"Expected: {expected_improve} Actual: {improve}")
            print(f"KL {mean_kl}")

            if mean_kl > self.MAX_KL * 1.5:
                print("violated KL constraint. shrinking step.")
            elif improve < 0:
                print("surrogate didn't improve. shrinking step.")
            else:
                print("Stepsize OK!")
                break
        else:
            print("更新に失敗")
            self.policy.set_weights(params_old)

    def update_vf(self, trajectory):

        for _ in range(self.TRAJECTORY_SIZE // self.VF_BATCHSIZE):

            indx = np.random.choice(self.TRAJECTORY_SIZE, self.VF_BATCHSIZE, replace=True)

            with tf.GradientTape() as tape:
                vpred = self.value_network(trajectory["s"][indx])
                vtarget = trajectory["vftarget"][indx]
                loss = tf.reduce_mean(tf.square(vtarget - vpred))

            variables = self.value_network.trainable_variables
            grads = tape.gradient(loss, variables)
            self.value_network.optimizer.apply_gradients(zip(grads, variables))

    def save_model(self):

        self.policy.save_weights("checkpoints/actor")

        self.value_network.save_weights("checkpoints/critic")

        print()
        print("Model Saved")
        print()

    def load_model(self):

        self.policy.load_weights("checkpoints/actor")

        self.value_network.load_weights("checkpoints/critic")

    def test_play(self, n, monitordir, load_model=False):

        if load_model:
            self.load_model()

        if monitordir:
            env = wrappers.Monitor(gym.make(self.ENV_ID),
                                   monitordir, force=True,
                                   video_callable=(lambda ep: ep % 1 == 0))
        else:
            env = gym.make(self.ENV_ID)

        for i in range(n):

            total_reward = 0

            steps = 0

            done = False

            state = env.reset()

            while not done:

                action = self.policy.sample_action(state)

                next_state, reward, done, _ = env.step(action)

                state = next_state

                total_reward += reward

                steps += 1

            print()
            print(f"Test Play {i}: {total_reward}")
            print(f"Steps:", steps)
            print()


def main():

    agent = TRPOAgent()

    history = agent.play(n_iters=200)

    print(history)

    plt.plot(range(len(history)), history)
    plt.xlabel("Episodes")
    plt.ylabel("Total Rewards")
    plt.savefig("history/log.png")

    agent.test_play(n=3, monitordir="history", load_model=True)


if __name__ == "__main__":
    main()
