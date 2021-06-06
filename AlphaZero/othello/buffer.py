import collections


class ReplayBuffer:

    def __init__(self, buffer_size):

        self.buffer = collections.deque(maxlen=buffer_size)

    def get_minibatch(self, batch_size):
        return None

    def add_record(self, record):
        for game_step in record:
            self.buffer.append(game_step)
