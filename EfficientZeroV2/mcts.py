from typing import List, Optional
from dataclasses import dataclass

import numpy as np


def search(
    root_state: np.ndarray,
    action_space: int,
    network,
    num_simulations: int,
    gamma: float,
) -> tuple[int, float, float]:

    best_actions, _, _ = search_batch(
        root_states=[root_state],
        action_space=action_space,
        network=network,
        num_simulations=num_simulations,
        gamma=gamma,
    )
    return best_actions[0]


def search_batch(
    root_states: list[np.ndarray],
    action_space: int,
    network,
    num_simulations: int,
    gamma: float,
) -> tuple[list[int], list[float], list[float]]:
    B: int = len(root_states)
    for i in range(num_simulations):
        trees = [
            GumbelMCTS(
                action_space=action_space,
                network=network,
                num_simulations=num_simulations,
                gamma=gamma,
                temperature=1.0,
            )
            for _ in range(B)
        ]
        import pdb; pdb.set_trace()  # fmt: skip
        pass
    root_nodes = [Node(state=state, parent=None, children=[]) for state in root_states]
    # return best_actions, values, policies


@dataclass
class Node:

    state: np.ndarray
    action_space: int
    parent: Optional["Node"]
    children: List["Node"]
    visit_count: int

    def expand(self):
        pass


class GumbelMCTS:
    def __init__(
        self,
        pv_network,
        action_space: int,
        gamma: float,
        num_simulations: int,
        temperature: float = 1.0,
    ):
        self.pv_network = pv_network
        self.action_space = action_space
        self.gamma = gamma
        self.temperature = temperature
        self.num_simulations = num_simulations
