from typing import List, Optional, Generator, Callable
from dataclasses import dataclass, field

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
    """GPU効率のため複数のMCTSを疑似並列実行"""

    trees = []
    for i in range(batch_size):
        tree = GumbelMCTS(
            action_space=action_space,
            num_simulations=num_simulations,
            gamma=gamma,
            temperature=temperature,
        )
        root_state, root_value = states[i : i + 1], float(values[i][0].numpy())
        root_policy_logits = policy_logits[i].numpy().flatten()
        tree.setup(
            root_state=root_state,
            root_value=root_value,
            root_policy_logits=root_policy_logits,
        )
        trees.append(tree)

    for _ in range(num_simulations):
        simulations: list[Generator] = [tree.run_simulation() for tree in trees]

        prev_states, prev_actions = [], []
        for simulation in simulations:
            prev_state, prev_action = next(simulation)
            prev_states.append(prev_state)
            prev_actions.append([prev_action])

        prev_states = tf.concat(prev_states, axis=0)
        prev_actions = tf.convert_to_tensor(prev_actions, dtype=tf.float32)
        next_states = network.transition_network(
            prev_states, prev_actions, training=False
        )
        policy_logits, _, _, _, values = network.policy_value_network.predict(
            next_states, training=False
        )
        _, rewards = network.reward_network.predict(next_states, training=False)

        for i, simulation in enumerate(simulations):
            simulation.send(
                (
                    next_states[i : i + 1],
                    policy_logits[i].numpy().flatten(),
                    values[i][0].numpy(),
                    rewards[i][0].numpy(),
                )
            )

    selected_actions, target_policies, target_values = zip(
        *[tree.get_result() for tree in trees]
    )

    import pdb; pdb.set_trace()  # fmt: skip
    return (
        selected_actions,
        target_policies,
        target_values,
    )


class ValueStats(list):

    def normalize(self, value: float) -> float:
        if len(self) == 0:
            return np.clip(value, 0.0, 1.0)

        _min, _max = min(self), max(self)
        if _max > _min:
            value = np.clip(value, _min, _max)
            value = (value - _min) / max(_max - _min, 0.01)
        value = np.clip(value, 0.0, 1.0)
        return value


@dataclass
class Node:

    prior: float
    parent: Optional["Node"]
    prev_action: int | None
    depth: int
    gamma: float
    visit_count: int = 0

    state: np.ndarray | None = None
    priors: np.ndarray | None = None
    value: float | None = None
    reward: float | None = None
    children: Optional[List["Node"]] = None
    estimated_values: list[float] = field(default_factory=list)
    visible: bool = True

    def __repr__(self):
        return f"Node(prior={self.prior}, prev_action={self.prev_action}, visit_count={self.visit_count}, value={self.value}, reward={self.reward}, depth={self.depth}, visible={self.visible})"

    @property
    def is_expanded(self):
        return self.children is not None

    @property
    def estimated_value(self) -> np.ndarray:
        if self.estimated_values:
            return np.mean(self.estimated_values)
        else:
            return None

    def get_transformed_qsa(
        self,
        scaler: Callable[[float], float],
        c_visit: float = 50.0,
        c_scale: float = 0.1,
    ) -> float:
        if self.estimated_value is None:
            return 0.0
        else:
            scaled_qsa: float = scaler(self.estimated_value)
            max_child_visit_count = max(child.visit_count for child in self.children)
            transformed_qsa = (c_visit + max_child_visit_count) * c_scale * scaled_qsa
            return transformed_qsa

    def get_score(
        self,
        is_phase_zero: bool,
        scaler: Callable[[float], float],
        c_visit: float = 50.0,
        c_scale: float = 0.1,
    ) -> float:
        if is_phase_zero:
            return self.prior
        else:
            return self.prior + self.get_transformed_qsa(
                scaler=scaler, c_visit=c_visit, c_scale=c_scale
            )

    def search(self, action: int | None = None):
        self.visit_count += 1
        if action is None:
            """
            直感的には改善方策 π (a) が示唆する選択確率と、
            これまでの訪問回数 N(a) に基づく選択確率との差が最も大きい行動、
            つまり「まだ十分に訪問されていないが、改善方策上は有望な行動」を選択
            Note:
                No use of the improved policy (Gumbel MCTS) for simplification
            """
            policy = tf.math.softmax(self.priors).numpy()
            visit_counts = [child.visit_count for child in self.children]
            scores = [
                policy[i] - visit_counts[i] / (1 + sum(visit_counts))
                for i in range(len(self.children))
            ]
            action = np.argmax(scores)

        selected_node: Node = self.children[action]
        if selected_node.is_expanded:
            return selected_node.search(action=None)
        else:
            prev_state = self.state
            prev_action = action
            return selected_node, prev_state, prev_action

    def expand(
        self,
        state: np.ndarray,
        priors: np.ndarray,
        value: float = None,
        reward: float = None,
    ):
        self.state = state
        self.priors = priors
        self.value = value
        self.reward = reward
        self.children = [
            Node(
                prior=prior,
                prev_action=i,
                parent=self,
                depth=self.depth + 1,
                gamma=self.gamma,
            )
            for i, prior in enumerate(priors)
        ]
        self.visit_count += 1

    def backprop(self, vstats: ValueStats) -> float:
        value = self.reward + self.gamma * self.value
        self.estimated_values.append(value)
        vstats.append(value)

        node = self.parent
        while node.depth > 0:
            value = node.reward + self.gamma * value
            node.estimated_values.append(value)
            vstats.append(value)
            node = node.parent
        return value


class GumbelMCTS:
    def __init__(
        self,
        action_space: int,
        num_simulations: int,
        gamma: float,
        temperature: float,
    ):
        self.root_node = None

        self.action_space = action_space
        self.num_simulations = num_simulations
        assert (num_simulations & (num_simulations - 1)) == 0
        self.temperature = temperature
        self.gamma = gamma

        self.simulation_count = 0
        self.simulation_count_to_next_phase = self.num_simulations // 2
        self.is_phase_zero = True
        self.estimated_values = []
        self.vstats = ValueStats()

    def setup(
        self, root_state: np.ndarray, root_value: float, root_policy_logits: np.ndarray
    ):
        assert len(root_state.shape) == 4 and root_state.shape[0] == 1
        assert type(root_value) == float
        assert (
            len(root_policy_logits.shape) == 1
            and root_policy_logits.shape[0] == self.action_space
        )
        gumble_noises = (
            np.random.gumbel(0, 1, size=self.action_space) * self.temperature
        )
        scores = np.array(
            [score + noise for score, noise in zip(root_policy_logits, gumble_noises)],
            dtype=np.float32,
        )

        # ルートノードを作成
        self.root_node = Node(
            prior=1.0,
            parent=None,
            prev_action=None,
            depth=0,
            gamma=self.gamma,
        )
        self.root_node.expand(
            state=root_state,
            priors=scores,
            value=root_value,
            reward=0.0,
        )

    def run_simulation(self):

        # Select Action
        candidates: list[Node] = sorted(
            [
                child_node
                for child_node in self.root_node.children
                if child_node.visible
            ],
            key=lambda x: x.get_score(
                is_phase_zero=self.is_phase_zero, scaler=self.vstats.normalize
            ),
            reverse=True,
        )
        min_visit_candidate = min(candidates, key=lambda x: x.visit_count)
        action: int = min_visit_candidate.prev_action
        print("--------------------------------")
        print(f"selected action: {action}")

        # Retrieve leaf node
        leaf_node, prev_state, prev_action = self.root_node.search(action=action)

        # Expand leaf node
        received_values: tuple = yield (prev_state, prev_action)
        next_state, policy_logits, value, reward = received_values

        leaf_node.expand(
            state=next_state, priors=policy_logits, value=value, reward=reward
        )

        # backprop
        estimated_value = leaf_node.backprop(vstats=self.vstats)
        self.estimated_values.append(estimated_value)
        self.simulation_count += 1

        # Sequential Halving
        if self.simulation_count == self.simulation_count_to_next_phase:
            num_remaining_simulations = self.num_simulations - self.simulation_count

            if num_remaining_simulations > 2:
                self.is_phase_zero = False
                self.simulation_count_to_next_phase += num_remaining_simulations // 2

                child_nodes: list[Node] = sorted(
                    [
                        child_node
                        for child_node in self.root_node.children
                        if child_node.visible
                    ],
                    key=lambda x: x.get_score(
                        is_phase_zero=self.is_phase_zero, scaler=self.vstats.normalize
                    ),
                    reverse=True,
                )
                n_child_nodes = len(child_nodes)
                if n_child_nodes > 1:
                    m = n_child_nodes // 2
                    for child_node in child_nodes[-m:]:
                        child_node.visible = False
        yield

    def get_result(self):
        assert self.simulation_count == self.num_simulations
        import pdb; pdb.set_trace()  # fmt: skip
