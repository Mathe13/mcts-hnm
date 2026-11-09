"""Unit tests for Feature F3: Core MCTS Hard Negative Mining Agent (`src/mcts/`)."""

from __future__ import annotations

import numpy as np
import pytest

from src.mcts.agent import (
    MCTSHNMAgent,
    compute_ucb1_scores,
    normalize_sibling_q_values,
)
from src.trees.semantic_tree import build_semantic_tree


def _make_small_semantic_tree() -> tuple[dict[int, np.ndarray], object]:
    """Create a 6-leaf Semantic Tree spanning 3 superclasses (2 fine classes each)."""
    # Fine classes:
    # Superclass 0 (aquatic_mammals): beaver (4), dolphin (30)
    # Superclass 1 (fish): aquarium_fish (1), shark (73)
    # Superclass 2 (flowers): orchid (54), poppy (62)
    class_to_indices = {
        1: np.arange(0, 10, dtype=np.int64),
        4: np.arange(10, 30, dtype=np.int64),
        30: np.arange(30, 60, dtype=np.int64),
        54: np.arange(60, 75, dtype=np.int64),
        62: np.arange(75, 100, dtype=np.int64),
        73: np.arange(100, 140, dtype=np.int64),
    }
    tree = build_semantic_tree(class_to_indices)
    return class_to_indices, tree


def test_ucb1_selection_strict_q_norm_false() -> None:
    """Verify Strict GiantHunter (`q_norm=False`) computes exact raw-Q UCB1 scores and selection (MCTS-01, AC 1)."""
    _, tree = _make_small_semantic_tree()
    agent = MCTSHNMAgent(tree=tree, exploration_c=2.0, momentum_alpha=0.3, q_norm=False, eps=1e-5)

    root_id = tree.root_id
    children = tree.get_children(root_id)
    assert children.size == 3

    # Set all children as visited (n_i >= 1) with large raw Q differences
    tree.visits[root_id] = 10
    tree.visits[children] = np.array([8, 1, 1], dtype=np.int64)
    tree.q_values[children] = np.array([10.0, 6.0, 0.0], dtype=np.float64)

    expected_scores = np.array([10.0, 6.0, 0.0]) + 2.0 * np.sqrt(
        np.log(10.0) / (np.array([8.0, 1.0, 1.0]) + 1e-5)
    )
    selected_child, ucb_scores = agent.select_child(root_id)

    np.testing.assert_allclose(ucb_scores, expected_scores, rtol=1e-12, atol=1e-12)
    # In Strict mode (q_norm=False), raw Q_0 = 10.0 overwhelms the exploration bonus on child 1
    assert selected_child == int(children[0])


def test_ucb1_selection_enhanced_q_norm_true_and_equal_q_zero() -> None:
    """Verify Enhanced MCTS-HNM (`q_norm=True`) normalizes sibling Q-values into [0, 1] and handles Q_max == Q_min (MCTS-01, AC 2)."""
    # 1. Exact Min-Max normalization check
    q_raw = np.array([2.0, 6.0, 10.0], dtype=np.float64)
    q_normed = normalize_sibling_q_values(q_raw, eps=1e-5)
    expected_normed = (q_raw - 2.0) / (8.0 + 1e-5)
    np.testing.assert_allclose(q_normed, expected_normed, rtol=1e-12, atol=1e-12)
    assert np.all((q_normed >= 0.0) & (q_normed <= 1.0))

    # 2. Equal sibling Q-values (Q_max == Q_min) -> 0.0 without NaN
    q_equal = np.array([3.5, 3.5, 3.5], dtype=np.float64)
    q_zero = normalize_sibling_q_values(q_equal, eps=1e-5)
    assert not np.any(np.isnan(q_zero))
    np.testing.assert_array_equal(q_zero, np.zeros(3, dtype=np.float64))

    # 3. On the same [10.0, 6.0, 0.0], n=[8, 1, 1] fixture, q_norm=True restores exploration
    _, tree = _make_small_semantic_tree()
    agent_enhanced = MCTSHNMAgent(
        tree=tree, exploration_c=2.0, momentum_alpha=0.3, q_norm=True, eps=1e-5
    )
    root_id = tree.root_id
    children = tree.get_children(root_id)
    tree.visits[root_id] = 10
    tree.visits[children] = np.array([8, 1, 1], dtype=np.int64)
    tree.q_values[children] = np.array([10.0, 6.0, 0.0], dtype=np.float64)

    selected_child, ucb_scores = agent_enhanced.select_child(root_id)
    expected_q_norm = (np.array([10.0, 6.0, 0.0]) - 0.0) / (10.0 + 1e-5)
    expected_scores = expected_q_norm + 2.0 * np.sqrt(
        np.log(10.0) / (np.array([8.0, 1.0, 1.0]) + 1e-5)
    )
    np.testing.assert_allclose(ucb_scores, expected_scores, rtol=1e-12, atol=1e-12)
    # Under q_norm=True, child 1 (Q_norm=0.6, n=1) beats child 0 (Q_norm=1.0, n=8)
    assert selected_child == int(children[1])

    # 4. When Q_max == Q_min at an internal node, selection picks the least-visited child
    tree.q_values[children] = np.array([4.2, 4.2, 4.2], dtype=np.float64)
    tree.visits[children] = np.array([5, 4, 1], dtype=np.int64)
    selected_equal_q, scores_equal_q = agent_enhanced.select_child(root_id)
    assert not np.any(np.isnan(scores_equal_q))
    assert selected_equal_q == int(children[2])


def test_select_leaf_prioritizes_unvisited_children() -> None:
    """Verify `select_leaf()` deterministically prioritizes unvisited children (n_i = 0) before visited siblings (MCTS-01, AC 3)."""
    _, tree = _make_small_semantic_tree()
    agent = MCTSHNMAgent(tree=tree, exploration_c=2.0, momentum_alpha=0.3, q_norm=False)

    # Even if superclass 0 and its first leaf already have N_root = 1 and huge Q = 500.0,
    # unvisited superclasses (n_i = 0) must be selected first.
    root_id = tree.root_id
    superclasses = tree.get_children(root_id)
    sc0 = int(superclasses[0])
    sc0_leaves = tree.get_children(sc0)
    leaf0 = int(sc0_leaves[0])

    tree.visits[root_id] = 1
    tree.visits[sc0] = 1
    tree.visits[leaf0] = 1
    tree.q_values[sc0] = 500.0
    tree.q_values[leaf0] = 500.0

    chosen_leaf, path = agent.select_leaf()
    assert path[0] == root_id
    assert path[1] == int(superclasses[1])
    assert chosen_leaf == int(tree.get_children(int(superclasses[1]))[0])

    # Reset and verify that 6 consecutive leaf visits cover all 6 leaves before repeating
    agent.reset()
    assert int(tree.visits.sum()) == 0
    visited_leaves: list[int] = []
    for _ in range(tree.num_leaves):
        leaf_id, leaf_path = agent.select_leaf()
        assert leaf_path[0] == root_id
        assert leaf_path[-1] == leaf_id
        assert bool(tree.is_leaf[leaf_id])
        # Simulate visit increment along the selected path
        for nid in leaf_path:
            tree.visits[nid] += 1
            tree.q_values[nid] = 10.0
        visited_leaves.append(leaf_id)

    assert len(set(visited_leaves)) == tree.num_leaves == 6


def test_agent_invalid_hyperparameters_raise() -> None:
    """Verify `MCTSHNMAgent` and helpers raise `ValueError` on invalid hyperparameters (MCTS-01, AC 4)."""
    _, tree = _make_small_semantic_tree()

    with pytest.raises(ValueError, match="exploration_c"):
        MCTSHNMAgent(tree=tree, exploration_c=-0.5)

    with pytest.raises(ValueError, match="momentum_alpha"):
        MCTSHNMAgent(tree=tree, momentum_alpha=0.0)

    with pytest.raises(ValueError, match="momentum_alpha"):
        MCTSHNMAgent(tree=tree, momentum_alpha=1.2)

    with pytest.raises(ValueError, match="eps"):
        MCTSHNMAgent(tree=tree, eps=0.0)

    with pytest.raises(ValueError, match="eps"):
        MCTSHNMAgent(tree=tree, eps=-1e-5)

    with pytest.raises(ValueError, match="q_norm"):
        MCTSHNMAgent(tree=tree, q_norm=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="parent_visits"):
        compute_ucb1_scores(np.array([1.0, 2.0]), np.array([1, 1]), parent_visits=-1)
