from typing import List, Optional
from dataclasses import dataclass

import numpy as np
import tensorflow as tf


def search(
    raw_state: np.ndarray,
    action_space: int,
    network,
    num_simulations: int,
    gamma: float,
) -> tuple[int, float, float]:

    best_actions, _, _ = search_batch(
        raw_states=[raw_state],
        action_space=action_space,
        network=network,
        num_simulations=num_simulations,
        gamma=gamma,
    )
    return best_actions[0]


def search_batch(
    raw_states: list[np.ndarray],
    action_space: int,
    network,
    num_simulations: int,
    gamma: float,
) -> tuple[list[int], list[float], list[float]]:
    B: int = len(raw_states)

    raw_states = tf.concat(raw_states, axis=0)
    states = network.representation_network(raw_states, training=False)
    policy_logits, _, _, _, values = network.policy_value_network.predict(
        states, training=False
    )

    #: Setup root nodes
    root_nodes = []
    for i, root_node in enumerate(range(B)):
        root_node = Node(
            state=states[i : i + 1],
            prior=1.0,
            value=values[i][0].numpy(),
            reward=0,
            parent=None,
            children=None,
            depth=0,
            visit_count=0,
        )
        root_node.expand(policy_logits=policy_logits[i].numpy().flatten())
        root_nodes.append(root_node)


@dataclass
class Node:

    state: np.ndarray | None

    prior: float
    value: float | None
    reward: float | None

    parent: Optional["Node"]
    children: Optional[List["Node"]]
    depth: int
    visit_count: int

    def expand(self, policy_logits: np.ndarray):
        self.children = [
            Node(
                state=None,
                prior=logit,
                value=None,
                reward=None,
                parent=self,
                children=None,
                depth=self.depth + 1,
                visit_count=0,
            )
            for logit in policy_logits
        ]
        self.visit_count += 1


class GumbelMCTS:
    def __init__(
        self,
        state: np.ndarray,
        action_space: int,
        network,
        num_simulations: int,
        gamma: float,
        temperature: float = 1.0,
    ):
        self.action_space = action_space
        self.network = network
        self.num_simulations = num_simulations
        self.gamma = gamma
        self.temperature = temperature

        self.root_node = Node(
            state=state,
            reward=0,
            action_space=self.action_space,
            parent=None,
            children=None,
            depth=0,
            prior=1.0,
            visit_count=1,
        )

        self.target_node = None

    def run(self):
        if self.root_node.children is None:
            pass

        else:
            pass
            raise NotImplementedError()

    def expand(self, node: Node):
        node
        pass

    def get_trajectory(self):
        pass
