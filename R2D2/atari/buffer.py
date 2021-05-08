import collections
import random
import math

import numpy as np


Transition = collections.namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done",
                   "c", "h", "prev_action", "is_terminal"])

Segment = collections.namedtuple(
    "Segment", ["states", "actions", "rewards", "dones",
                "c_init", "h_init", "prev_action_init", "last_state"])


class EpisodeBuffer:

    def __init__(self, burnin_length, unroll_length, nstep, gamma):

        self.transitions = []

        self.burnin_len = burnin_length
        self.unroll_len = unroll_length

        self.nstep = nstep
        self.gamma = gamma
        self.tmp_buffer = collections.deque(maxlen=self.nstep)

        self.reward_clipping_func = \
            (lambda r: math.copysign(1, r) * (math.sqrt(abs(r) + 1) - 1) + 0.001 * r)

    def __len__(self):
        return len(self.transitions)

    def add(self, transition):
        """
        Args:
            transition (Tuple): (s, a, r, s2, done, c, h)
        """

        transition = Transition(*transition)
        is_terminal = transition.is_terminal
        self.tmp_buffer.append(transition)

        if len(self.tmp_buffer) == self.nstep:
            indices = [0] if not is_terminal else range(self.nstep)
            for idx in indices:
                nstep_return = 0
                has_done = False

                for i, target_idx in enumerate(range(self.nstep)[idx:]):
                    transition = self.tmp_buffer[target_idx]
                    reward, done = transition.reward, transition.done
                    reward = self.reward_clipping_func(reward)
                    nstep_return += self.gamma ** i * (1 - done) * reward
                    if done:
                        has_done = True
                        break

                nstep_transition = Transition(
                    state=self.tmp_buffer[idx].state,
                    action=self.tmp_buffer[idx].action,
                    reward=nstep_return,
                    next_state=self.tmp_buffer[-1].next_state,
                    done=has_done,
                    c=self.tmp_buffer[idx].c,
                    h=self.tmp_buffer[idx].h,
                    prev_action=self.tmp_buffer[idx].prev_action,
                    is_terminal=None)

                self.transitions.append(nstep_transition)

        if is_terminal:
            self.tmp_buffer = None

    def pull_segments(self):

        # this method shold be called after episode end
        assert self.tmp_buffer is None
        assert np.all([t.done for t in self.transitions[-self.nstep:]])
        assert not self.transitions[-self.nstep-1].done

        segments = []

        for t in range(self.burnin_len, len(self.transitions), self.unroll_len):

            if (t + self.unroll_len) > len(self.transitions):
                #: エピソード終端の長さ修正
                total_len = self.burnin_len + self.unroll_len
                timesteps = self.transitions[-total_len:]
            else:
                timesteps = self.transitions[t-self.burnin_len:t+self.unroll_len]

            segment = Segment(
                states=[t.state for t in timesteps],
                actions=[t.action for t in timesteps],
                rewards=[t.reward for t in timesteps][self.burnin_len:],
                dones=[t.done for t in timesteps][self.burnin_len:],
                c_init=timesteps[0].c,
                h_init=timesteps[0].h,
                prev_action_init=timesteps[0].prev_action,
                last_state=timesteps[-1].next_state
                )
            segments.append(segment)

        return segments


class SegmentReplayBuffer:

    def __init__(self, buffer_size):

        self.buffer_size = buffer_size
        self.priorities = SumTree(capacity=self.buffer_size)
        self.segment_buffer = [None] * self.buffer_size

        self.beta = 0.6  # importance sampling exponent

        self.count = 0
        self.full = False

    def __len__(self):
        return len(self.segment_buffer) if self.full else self.count

    def add(self, priorities: list, segments: list):
        assert len(priorities) == len(segments)

        for priority, segment in zip(priorities, segments):

            self.priorities[self.count] = priority
            self.segment_buffer[self.count] = segment

            self.count += 1
            if self.count == self.buffer_size:
                self.count = 0
                self.full = True

    def update_priority(self, sampled_indices, priorities):
        assert len(sampled_indices) == len(priorities)

        for idx, priority in zip(sampled_indices, priorities):
            self.priorities[idx] = priority

    def sample_minibatch(self, batch_size):

        sampled_indices = [self.priorities.sample() for _ in range(batch_size)]

        #: Compute importance sampling weights
        weights = []
        current_size = len(self.segment_buffer) if self.full else self.count
        for idx in sampled_indices:
            prob = self.priorities[idx] / self.priorities.sum()
            weight = (prob * current_size)**(-self.beta)
            weights.append(weight)
        weights = np.array(weights) / max(weights)

        sampled_segments = [self.segment_buffer[idx] for idx in sampled_indices]

        return sampled_indices, weights, sampled_segments


class SumTree:
    """ See https://github.com/ray-project/ray/blob/master/rllib/execution/segment_tree.py
    """

    def __init__(self, capacity: int):
        #: 2のべき乗チェック
        assert capacity & (capacity - 1) == 0
        self.capacity = capacity
        self.values = [0 for _ in range(2 * capacity)]

    def __str__(self):
        return str(self.values[self.capacity:])

    def __setitem__(self, idx, val):
        idx = idx + self.capacity
        self.values[idx] = val

        current_idx = idx // 2
        while current_idx >= 1:
            idx_lchild = 2 * current_idx
            idx_rchild = 2 * current_idx + 1
            self.values[current_idx] = self.values[idx_lchild] + self.values[idx_rchild]
            current_idx //= 2

    def __getitem__(self, idx):
        idx = idx + self.capacity
        return self.values[idx]

    def sum(self):
        return self.values[1]

    def sample(self):
        z = random.uniform(0, self.sum())
        try:
            assert 0 <= z <= self.sum(), z
        except AssertionError:
            print(z)
            import pdb; pdb.set_trace()

        current_idx = 1
        while current_idx < self.capacity:

            idx_lchild = 2 * current_idx
            idx_rchild = 2 * current_idx + 1

            #: 左子ノードよりzが大きい場合は右子ノードへ
            if z > self.values[idx_lchild]:
                current_idx = idx_rchild
                z = z -self.values[idx_lchild]
            else:
                current_idx = idx_lchild

        #: 見かけ上のインデックスにもどす
        idx = current_idx - self.capacity
        return idx
