from typing import List, Optional
from dataclasses import dataclass

import numpy as np


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

    def search_batch(self, root_states: list[np.ndarray]) -> int:
        root_nodes = [
            Node(state=state, parent=None, children=[]) for state in root_states
        ]
        import pdb; pdb.set_trace()  # fmt: skip
        return best_actions, values, policies
