import collections


Transition = collections.namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done", "c", "h"])

Segment = collections.namedtuple(
    "Segment", ["states", "actions", "rewards", "dones", "c_init", "h_init", "last_state"])


class EpisodeBuffer:

    def __init__(self, burnin_length, unroll_length):

        self.transitions = []
        self.burnin_len = burnin_length
        self.unroll_len = unroll_length

    def __len__(self):
        return len(self.transitions)

    def put(self, transition):
        #: transition: (s, a, r, s2, done, c, h)
        self.transitions.append(Transition(*transition))

    def pull(self):
        segments = []
        for i in range(self.burnin_len, len(self.transitions), self.unroll_len):

            if (i + self.unroll_len) > len(self.transitions):
                #: エピソード終端の長さ修正
                total_len = self.burnin_len + self.unroll_len
                timesteps = self.transitions[-total_len:]
            else:
                timesteps = self.transitions[i-self.burnin_len:i+self.unroll_len]

            segment = Segment(
                states=[t.state for t in timesteps],
                actions=[t.action for t in timesteps],
                rewards=[t.reward for t in timesteps],
                dones=[t.done for t in timesteps],
                c_init=timesteps[0].c,
                h_init=timesteps[0].h,
                last_state=timesteps[-1].next_state
                )
            segments.append(segment)

        return segments
