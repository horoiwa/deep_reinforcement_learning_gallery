from typing import List, Optional, Generator
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
    for tree in trees:
        tree.get_result()

    import pdb; pdb.set_trace()  # fmt: skip
    selected_actions, mcts_policies, search_based_values = [
        tree.get_mcts_result() for tree in trees
    ]

    return (
        selected_actions,
        mcts_policies,
        search_based_values,
    )


class ValueStats(list):

    def normalize(self, value: float) -> float:

        if len(self) == 0:
            return value

        _min, _max = min(self), max(self)
        if _max > _min:
            if value >= _max:
                value = _max
            elif value <= _min:
                value = _min
            value = (value - _min) / (_max - _min)
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
    def policy(self) -> np.ndarray:
        return tf.math.softmax(self.priors).numpy()

    @property
    def estimated_value(self) -> np.ndarray:
        return np.mean(self.estimated_values)

    def get_v_mix(self) -> float:
        pi = self.policy
        pi_sum, pi_qsa_sum = 0.0, 0.0
        for i, child_node in enumerate(self.children):
            if child_node.is_expanded:
                pi_sum += pi[i]
                pi_qsa_sum += pi[i] * (
                    child_node.reward + self.gamma * child_node.value
                )
        if pi_sum < 1e-6:
            v_mix = self.estimated_value
        else:
            visit_sum = sum(child_node.visit_count for child_node in self.children)
            v_mix = (1.0 / (1.0 + visit_sum)) * (
                self.estimated_value + visit_sum * pi_qsa_sum / pi_sum
            )
        return v_mix

    def get_completed_Qs(self) -> list[float]:
        """
        v_mix: https://openreview.net/pdf?id=bERaNdoegnO#page=16
        """
        v_mix: float = self.get_v_mix()

        completed_Qs = []
        for child_node in self.children:
            if child_node.is_expanded:
                # 子ノードが展開済みの場合は1step-TD
                completed_Q = child_node.reward + self.gamma * child_node.value
            else:
                # 子ノードが展開済みでない場合は親nodeのv_mix
                completed_Q = v_mix

            completed_Qs.append(completed_Q)

        return completed_Qs

    def get_transformed_completed_Qs(
        self, vstats: ValueStats, c_visit: float = 50.0, c_scale: float = 0.1
    ) -> np.ndarray:
        completed_Qs: list[float] = self.get_completed_Qs()
        normalized_completed_Qs = np.array([vstats.normalize(q) for q in completed_Qs])

        max_child_visit_count = max(child.visit_count for child in self.children)

        transformed_completed_Qs = (
            (c_visit + max_child_visit_count) * c_scale * normalized_completed_Qs
        )
        return transformed_completed_Qs

    def search(self, action: int | None = None, vstats: ValueStats | None = None):
        self.visit_count += 1
        if action is None:
            """
            直感的には改善方策 π (a) が示唆する選択確率と、
            これまでの訪問回数 N(a) に基づく選択確率との差が最も大きい行動、
            つまり「まだ十分に訪問されていないが、改善方策上は有望な行動」を選択
            """
            assert vstats is not None
            policy = tf.math.softmax(
                self.priors + self.get_transformed_completed_Qs(vstats=vstats)
            ).numpy()

            visit_counts = [child.visit_count for child in self.children]
            scores = [
                policy[i] - visit_counts[i] / (1 + sum(visit_counts))
                for i in range(len(self.children))
            ]
            action = np.argmax(scores)

        selected_node: Node = self.children[action]
        if selected_node.is_expanded:
            return selected_node.search(action=None, vstats=vstats)
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

    def backprop(self, value_stats: ValueStats) -> float:
        value = self.reward + self.gamma * self.value
        self.estimated_values.append(value)

        node = self.parent
        while node.depth > 0:
            value = node.reward + self.gamma * value
            node.estimated_values.append(value)
            value_stats.append(value)
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
        self.estimated_values = []
        self.value_stats: list = ValueStats()

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
            [v for v in self.root_node.children if v.visible],
            key=lambda x: x.prior,
            reverse=True,
        )
        min_visit_candidate = min(candidates, key=lambda x: x.visit_count)
        action: int = min_visit_candidate.prev_action
        print("--------------------------------")
        print(f"selected action: {action}")

        # Retrieve leaf node
        leaf_node, prev_state, prev_action = self.root_node.search(
            action=action, vstats=self.value_stats
        )

        # Expand leaf node
        received_values: tuple = yield (prev_state, prev_action)
        next_state, policy_logits, value, reward = received_values

        leaf_node.expand(
            state=next_state, priors=policy_logits, value=value, reward=reward
        )

        # backprop
        estimated_value = leaf_node.backprop(value_stats=self.value_stats)
        self.estimated_values.append(estimated_value)
        self.simulation_count += 1

        if self.simulation_count in [8, 4, 2]:
            import pdb; pdb.set_trace()  # fmt: skip

        yield

    def get_result(self):
        import pdb; pdb.set_trace()  # fmt: skip
