"""Tests for Phase B MCTS policy diagnostics and micro-tree validation (B2)."""

from __future__ import annotations

import numpy as np
import pytest

from src.mcts.mcts_agent import MCTSHNMAgent, get_mcts_policy_diagnostics
from src.trees.base_tree import FastHierarchyTree


def make_micro_tree() -> FastHierarchyTree:
    """Create a minimal 3-leaf hierarchy tree for deterministic testing.

    Structure:
    - Node 0: root (internal), children [1, 2]
    - Node 1: internal, parent 0, children [3, 4]
    - Node 2: leaf (class 0), parent 0
    - Node 3: leaf (class 1), parent 1
    - Node 4: leaf (class 2), parent 1
    Total nodes: 5. Leaves: [2, 3, 4].
    """
    total_nodes = 5
    parents = np.array([-1, 0, 0, 1, 1], dtype=np.int64)
    children = np.array([
        [1, 2],
        [3, 4],
        [-1, -1],
        [-1, -1],
        [-1, -1],
    ], dtype=np.int64)
    child_counts = np.array([2, 2, 0, 0, 0], dtype=np.int64)
    is_leaf = np.array([False, False, True, True, True], dtype=bool)
    depths = np.array([0, 1, 1, 2, 2], dtype=np.int64)
    leaf_to_class_id = np.array([-1, -1, 0, 1, 2], dtype=np.int64)
    instance_counts = np.array([30, 20, 10, 10, 10], dtype=np.int64)
    visits = np.zeros(total_nodes, dtype=np.int64)
    q_values = np.zeros(total_nodes, dtype=np.float64)
    node_names = ["root", "cluster_1", "class_0", "class_1", "class_2"]
    class_to_indices = {
        0: np.arange(0, 10, dtype=np.int64),
        1: np.arange(10, 20, dtype=np.int64),
        2: np.arange(20, 30, dtype=np.int64),
    }

    return FastHierarchyTree(
        tree_type="synthetic_micro",
        root_id=0,
        parents=parents,
        children=children,
        child_counts=child_counts,
        is_leaf=is_leaf,
        depths=depths,
        leaf_to_class_id=leaf_to_class_id,
        instance_counts=instance_counts,
        visits=visits,
        q_values=q_values,
        node_names=node_names,
        class_to_indices=class_to_indices,
    )


def test_mcts_policy_diagnostics_structure():
    """Verify that get_mcts_policy_diagnostics returns all required metrics (MCTS-01, AC 1)."""
    tree = make_micro_tree()
    agent = MCTSHNMAgent(tree=tree, exploration_c=2.0)

    # Perform an update on leaf 3
    agent.update(leaf_node_id=3, reward=1.5)

    diag = agent.get_policy_diagnostics()
    assert "q_values" in diag and len(diag["q_values"]) == 5
    assert "visits" in diag and len(diag["visits"]) == 5
    assert "exploration_terms" in diag and len(diag["exploration_terms"]) == 5
    assert "ucb_scores" in diag and len(diag["ucb_scores"]) == 5
    assert "raw_rewards" in diag
    assert "smoothed_rewards" in diag
    assert diag["raw_rewards"][3] == 1.5


def test_mcts_exploration_positivity():
    """Verify that U > 0 for siblings when parent visits > 1 and c > 0 (MCTS-01, AC 2)."""
    tree = make_micro_tree()
    agent = MCTSHNMAgent(tree=tree, exploration_c=2.0)

    # Update leaf 2 twice -> root visits = 2 > 1 -> ln(N_root) > 0
    agent.update(leaf_node_id=2, reward=0.8)
    agent.update(leaf_node_id=2, reward=0.8)
    assert tree.visits[0] == 2

    diag = agent.get_policy_diagnostics()
    u_terms = diag["exploration_terms"]

    # Child 1 and Child 2 of root must have U > 0 because parent visits == 2 > 1
    assert u_terms[1] > 0.0, f"Child 1 must have positive exploration term, got {u_terms[1]}"
    assert u_terms[2] > 0.0, f"Child 2 must have positive exploration term, got {u_terms[2]}"


def test_micro_tree_no_starvation():
    """Verify that MCTS visits all leaves without premature starvation (MCTS-02, AC 3)."""
    tree = make_micro_tree()
    agent = MCTSHNMAgent(tree=tree, exploration_c=2.0)

    visited_leaves: set[int] = set()

    for _ in range(10):
        leaf_id, path = agent.select_leaf()
        visited_leaves.add(leaf_id)
        # Uniform deterministic reward
        agent.update(leaf_id, reward=1.0)

    # All 3 leaves [2, 3, 4] must be visited within 10 steps
    assert {2, 3, 4}.issubset(visited_leaves), f"Not all leaves visited: {visited_leaves}"
    for leaf in [2, 3, 4]:
        assert tree.visits[leaf] >= 1


def test_micro_tree_backpropagation_to_root():
    """Verify that backpropagation updates visits and Q along full ancestor path to root (MCTS-02, AC 3)."""
    tree = make_micro_tree()
    agent = MCTSHNMAgent(tree=tree, momentum_alpha=0.5)

    # Leaf 4 has parent 1, and 1 has parent 0 (root)
    # Update leaf 4 with reward 2.0
    agent.update(leaf_node_id=4, reward=2.0)

    # 1. Visits must increment along path: 4, 1, 0
    assert tree.visits[4] == 1
    assert tree.visits[1] == 1
    assert tree.visits[0] == 1
    assert tree.visits[2] == 0
    assert tree.visits[3] == 0

    # 2. Q-value at leaf 4: alpha * 2.0 + (1 - alpha) * 0 = 1.0
    assert np.isclose(tree.q_values[4], 1.0)

    # 3. Node 1 has children 3 (count 10, Q=0) and 4 (count 10, Q=1.0) -> parent count 20
    # Q(1) = (10 * 0 + 10 * 1.0) / 20 = 0.5
    assert np.isclose(tree.q_values[1], 0.5)

    # 4. Root 0 has children 1 (count 20, Q=0.5) and 2 (count 10, Q=0) -> root count 30
    # Q(0) = (20 * 0.5 + 10 * 0) / 30 = 10 / 30 = 1/3
    assert np.isclose(tree.q_values[0], 1.0 / 3.0)
