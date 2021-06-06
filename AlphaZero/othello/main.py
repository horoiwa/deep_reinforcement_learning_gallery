import collections

import tensorflow as tf

from network import AlaphaZeroNetwork
from mcts import MCTS
import othello


def selfplay(network):
    pass


def main(buffer_size=10000, lr=0.05):

    replay = collections.deque(maxlen=buffer_size)
    optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=0.9)
    network = None



if __name__ == "__main__":
    main()
