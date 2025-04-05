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
    temperature: float = 1.0,
) -> tuple[int, float, float]:

    best_actions, _, _ = search_batch(
        raw_states=[raw_state],
        action_space=action_space,
        network=network,
        num_simulations=num_simulations,
        gamma=gamma,
        temperature=temperature,
    )
    return best_actions[0]


def search_batch(
    raw_states: list[np.ndarray],
    action_space: int,
    network,
    num_simulations: int,
    gamma: float,
    temperature: float = 1.0,
) -> tuple[list[int], list[float], list[float]]:

    batch_size: int = len(raw_states)
    raw_states = tf.concat(raw_states, axis=0)
    states = network.representation_network(raw_states, training=False)
    policy_logits, _, _, _, values = network.policy_value_network.predict(
        states, training=False
    )

    trees = []
    for i in range(batch_size):
        tree = GumbelMCTS(
            tid=i,
            action_space=action_space,
            num_simulations=num_simulations,
            gamma=gamma,
            temperature=temperature,
        )
        root_state, root_value = states[i : i + 1], float(values[i][0].numpy())
        root_policy_logits = policy_logits[i].numpy().flatten()
        tree.initialize(
            root_state=root_state,
            root_value=root_value,
            root_policy_logits=root_policy_logits,
        )
        trees.append(tree)

    for n in range(num_simulations):
        nodes_to_evaluates: list[Node] = [tree.search() for tree in trees]
    import pdb; pdb.set_trace()  # fmt: skip


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
        tid: int,
        action_space: int,
        num_simulations: int,
        gamma: float,
        temperature: float,
    ):
        self.tid = tid
        self.root_node = None
        self.action_space = action_space
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.gamma = gamma

        self.search_based_values = []
        self.simulation_count = 0

    def initialize(
        self, root_state: np.ndarray, root_value: float, root_policy_logits: np.ndarray
    ):
        assert len(root_state.shape) == 4 and root_state.shape[0] == 1
        assert type(root_value) == float
        assert (
            len(root_policy_logits.shape) == 1
            and root_policy_logits.shape[0] == self.action_space
        )
        self.root_node = Node(
            state=root_state,
            prior=1.0,
            value=root_value,
            reward=0,
            parent=None,
            children=None,
            depth=0,
            visit_count=0,
        )
        self.root_node.expand(policy_logits=root_policy_logits)
        self.search_based_values.append([self.root_node.value])

    def search(self, n: int):
        gumble_noise = np.random.gumbel(0, 1, size=self.action_space) * self.temperature
        self.simulation_count += 1
