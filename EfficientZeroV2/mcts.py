from typing import List, Optional, Generator, Callable
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf


def search(
    observation: np.ndarray,
    action_space: int,
    network,
    num_simulations: int,
    gamma: float,
    temperature: float = 1.0,
    debug: bool = False,
) -> tuple[int, float, float]:

    best_actions, policies, values = search_batch(
        observations=[observation],
        action_space=action_space,
        network=network,
        num_simulations=num_simulations,
        gamma=gamma,
        temperature=temperature,
        debug=debug,
    )
    return best_actions[0], policies[0], values[0]


def search_batch(
    observations: list[np.ndarray],
    action_space: int,
    network,
    num_simulations: int,
    gamma: float,
    temperature: float = 1.0,
    training: bool = False,
    debug: bool = False,
) -> tuple[list[int], list[float], list[float]]:

    batch_size: int = len(observations)
    observations = tf.concat(observations, axis=0)
    root_states = network.encode(observations, training=training)
    root_policy_logits, _, _, root_values = network.predict_pv(
        root_states, training=training
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
        root_state, root_value = root_states[i : i + 1], float(
            root_values[i][0].numpy()
        )
        root_policy_logit = root_policy_logits[i].numpy().flatten()
        tree.setup(
            root_state=root_state,
            root_value=root_value,
            root_policy_logits=root_policy_logit,
        )
        trees.append(tree)

    for n in range(num_simulations):
        simulations: list[Generator] = [
            tree.run_simulation(debug=debug) for tree in trees
        ]

        prev_states, prev_actions = [], []
        for simulation in simulations:
            prev_state, prev_action = next(simulation)
            prev_states.append(prev_state)
            prev_actions.append([prev_action])

        prev_states = tf.concat(prev_states, axis=0)
        prev_actions = tf.convert_to_tensor(np.array(prev_actions), dtype=tf.float32)
        next_states, _, rewards = network.unroll(
            prev_states, prev_actions, training=False
        )
        policy_logits, _, _, values = network.predict_pv(next_states, training=training)

        for i, simulation in enumerate(simulations):
            simulation.send(
                (
                    next_states[i : i + 1],
                    policy_logits[i].numpy().flatten(),
                    values[i][0].numpy(),
                    rewards[i][0].numpy(),
                )
            )
        if debug:
            print(f"\t r_pred:{rewards[0][0]:.2f}, v_pred::{values[0][0]:.2f}")

    best_actions, mcts_policies, mcts_values = zip(
        *[tree.get_simulation_result(debug=debug) for tree in trees]
    )
    mcts_policies = tf.cast(tf.stack(mcts_policies), tf.float32)
    mcts_values = tf.cast(tf.stack(mcts_values), tf.float32)

    return (best_actions, mcts_policies, mcts_values)


@dataclass
class Node:

    policy_logit: float | None
    parent: Optional["Node"] | None
    prev_action: int | None
    depth: int
    gamma: float
    c_visit: float = 50.0
    c_scale: float = 0.1
    visit_count: int = 0
    noise: float = 0.0
    visible: bool | None = None

    state: np.ndarray | None = None
    value_nn: float | None = None
    reward_nn: float | None = None
    children: Optional[List["Node"]] = None
    search_based_values: list[float] = field(default_factory=list)

    def __repr__(self):
        if self.depth != 0 and self.is_expanded:
            return (
                f"\nNode(logit={self.policy_logit:.2f}, noise={self.noise:.2f},"
                f"prev_action={self.prev_action}, visit_count={self.visit_count},"
                f"value={self.value_nn:.2f}, reward={self.reward_nn:.2f},"
                f"depth={self.depth}, visible={self.visible})"
            )
        else:
            return (
                f"\nNode(logit={self.policy_logit}, noise={self.noise},"
                f"prev_action={self.prev_action}, visit_count={self.visit_count},"
                f"value={self.value_nn}, reward={self.reward_nn},"
                f"depth={self.depth}, visible={self.visible})"
            )

    @property
    def is_expanded(self):
        return self.children is not None

    def get_completed_qvalue(self):
        # Using completed Q-values
        if self.search_based_values:
            qvalue = np.mean(self.search_based_values)
        else:
            # No use of "D MIXED VALUE APPROXIMATION"
            qvalue = self.parent.value_nn
        return qvalue

    def get_improved_policy_logit(self, scale=False, debug: bool = False):
        # 4 LEARNING AN IMPROVED POLICY
        policy_logits = [child.policy_logit for child in self.children]

        qvalues = np.array([child.get_completed_qvalue() for child in self.children])
        if scale:
            q_min, q_max = qvalues.min(), qvalues.max()
            scaled_qvalues = (qvalues - q_min) / (q_max - q_min + 1e-8)
        else:
            scaled_qvalues = qvalues
        visit_counts = [child.visit_count for child in self.children]
        transformed_qvalues = [
            (self.c_visit + max(visit_counts)) * self.c_scale * q
            for q in scaled_qvalues
        ]
        if debug:
            print()
            print("\tqvalues:", np.round(qvalues, 3))
            print("\tscaled_qvalues:", np.round(scaled_qvalues, 3))
            print("\ttransformed_qvalues:", np.round(transformed_qvalues, 3))
            print("\tvisit_counts:", visit_counts)
            print("\tpolicy_logits:", np.round(policy_logits, 3))

        imporved_policy_logits = [
            logit + transformed_q
            for logit, transformed_q in zip(
                policy_logits, transformed_qvalues, strict=True
            )
        ]
        return imporved_policy_logits

    def search(self, action: int | None = None):
        self.visit_count += 1
        action = action if action is not None else self.select_action()
        selected_node: Node = self.children[action]
        if selected_node.is_expanded:
            return selected_node.search(action=None)
        else:
            prev_state = self.state
            prev_action = action
            return selected_node, prev_state, prev_action

    def select_action(self) -> int:
        # 5 PLANNING AT NON-ROOT NODES
        policy = tf.nn.softmax(self.get_improved_policy_logit()).numpy()
        visit_counts = [child.visit_count for child in self.children]
        scores = [
            policy[i] - visit_counts[i] / (1 + sum(visit_counts))
            for i in range(len(self.children))
        ]
        action = np.argmax(scores)
        return action

    def expand(
        self,
        state: np.ndarray,
        policy_logits: np.ndarray,
        value_nn: float = None,
        reward_nn: float = None,
        noises: np.ndarray = None,
    ):
        self.state = state
        self.value_nn = value_nn
        self.reward_nn = reward_nn

        if noises is None:
            noises = np.zeros_like(policy_logits, dtype=np.float32)

        self.children = [
            Node(
                policy_logit=logit,
                prev_action=i,
                parent=self,
                depth=self.depth + 1,
                gamma=self.gamma,
                noise=noise,
            )
            for i, (logit, noise) in enumerate(zip(policy_logits, noises, strict=True))
        ]
        self.visit_count += 1

    def backprop(self) -> float:
        value = self.reward_nn + self.gamma * self.value_nn
        self.search_based_values.append(value)

        node = self.parent
        while node.depth > 0:
            value = node.reward_nn + self.gamma * value
            node.search_based_values.append(value)
            node = node.parent
        return value


class GumbelMCTS:
    def __init__(
        self,
        action_space: int,
        num_simulations: int,
        gamma: float,
        temperature: float,
        c_visit: float = 50.0,
        c_scale: float = 0.1,
    ):
        self.root_node = None
        self.search_based_values = []

        self.action_space = action_space
        self.num_simulations = num_simulations
        self.max_considered_actions = num_simulations // 2
        assert (num_simulations & (num_simulations - 1)) == 0
        self.temperature = temperature
        self.gamma = gamma
        self.c_visit = c_visit
        self.c_scale = c_scale

        self.simulation_count = 0
        self.simulation_count_to_next_phase = self.num_simulations // 2

    def setup(
        self, root_state: np.ndarray, root_value: float, root_policy_logits: np.ndarray
    ):
        assert len(root_state.shape) == 4 and root_state.shape[0] == 1
        assert type(root_value) == float
        assert (
            len(root_policy_logits.shape) == 1
            and root_policy_logits.shape[0] == self.action_space
        )
        gumbel_noises = (
            np.random.gumbel(0, 1, size=self.action_space) * self.temperature
        )

        # ルートノードを作成
        self.root_node = Node(
            policy_logit=None,
            parent=None,
            prev_action=None,
            depth=0,
            gamma=self.gamma,
        )
        self.root_node.expand(
            state=root_state,
            policy_logits=root_policy_logits,
            value_nn=root_value,
            reward_nn=0.0,
            noises=gumbel_noises,
        )

        # Find m actions with the highest g(a) + logits(a)
        sorted_child_nodes = sorted(
            self.root_node.children,
            key=lambda x: x.noise + x.policy_logit,
            reverse=True,
        )
        for i, child_node in enumerate(sorted_child_nodes):
            if i < self.max_considered_actions:
                child_node.visible = True
            else:
                child_node.visible = False

    def run_simulation(self, debug: bool = False):

        # Select min visited action from visible children
        # (In each phase, all considered actions are visited equally often)
        action: int = sorted(
            [
                child_node
                for child_node in self.root_node.children
                if child_node.visible
            ],
            key=lambda x: x.visit_count,
            reverse=False,
        )[0].prev_action

        # Retrieve leaf node
        leaf_node, prev_state, prev_action = self.root_node.search(action=action)

        # Expand leaf node
        received_values: tuple = yield (prev_state, prev_action)
        next_state, policy_logits, value_nn, reward_nn = received_values

        leaf_node.expand(
            state=next_state,
            policy_logits=policy_logits,
            value_nn=value_nn,
            reward_nn=reward_nn,
        )

        # backprop
        estimated_value = leaf_node.backprop()
        self.search_based_values.append(estimated_value)
        self.simulation_count += 1

        if debug:
            print(
                f"{self.simulation_count}th sim, a: {action} search_value: {estimated_value:.2f}"
            )

        # Sequential Halving
        if self.simulation_count == self.simulation_count_to_next_phase:
            num_remaining_simulations = self.num_simulations - self.simulation_count
            self.simulation_count_to_next_phase += num_remaining_simulations // 2
            self.sequential_halving(debug=debug)

        yield

    def sequential_halving(self, debug: bool = False):
        # Use Sequential Halving with n simulations to identify the best action from the Atopm actions,
        # by comparing g(a) + logits(a) + σ(ˆq(a)).

        if len([n for n in self.root_node.children if n.visible]) == 1:
            return
        elif debug:
            print("Sequential Halving")

        # g(a) + logits(a) + σ(ˆq(a))
        improved_policy_logits = self.root_node.get_improved_policy_logit(debug=debug)
        noises = [node.noise for node in self.root_node.children]
        scores = [
            noise + improve_policy_logit
            for noise, improve_policy_logit in zip(
                noises, improved_policy_logits, strict=True
            )
        ]
        if debug:
            log = [f"\n\t a: {i}, score: {score:.2f}" for i, score in enumerate(scores)]
            print("".join(log) + "\n")

        scored_remaining_children = [
            (score, node)
            for score, node in zip(scores, self.root_node.children, strict=True)
            if node.visible
        ]
        sorted_scored_remaining_children = sorted(
            scored_remaining_children, key=lambda item: item[0], reverse=True
        )

        top_m = len(sorted_scored_remaining_children) // 2
        for _, child_node in sorted_scored_remaining_children[top_m:]:
            child_node.visible = False

    def get_simulation_result(self, debug: bool = False):
        assert self.simulation_count == self.num_simulations
        if debug:
            print("Simulation finished")

        policy_logits = self.root_node.get_improved_policy_logit(debug=debug)
        mcts_policy = tf.math.softmax(policy_logits).numpy()
        mcts_value = np.mean(self.search_based_values)

        scores = [child.noise + logit for child, logit in zip(self.root_node.children, policy_logits, strict=True)]
        best_action = np.argmax(scores)

        if debug:
            print("MCTS policy:", np.round(mcts_policy, 3))
            print("MCTS values:", np.round(self.search_based_values, 3))
            print("Scores:", np.round(scores, 3))
            print("Best action:", best_action)

        return best_action, mcts_policy, mcts_value
