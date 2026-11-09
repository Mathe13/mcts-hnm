"""Unit tests for Feature F3: Core MCTS Hard Negative Mining Agent (`src/mcts/`)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.mcts.agent import (
    MCTSHNMAgent,
    compute_mcts_reward,
    compute_ucb1_scores,
    normalize_sibling_q_values,
)
from src.trees.confusion_tree import build_confusion_tree
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


def test_backpropagation_leaf_momentum_update() -> None:
    """Verify `update(leaf_node_id, reward)` increments leaf visits and applies momentum alpha=0.3 (MCTS-02, AC 1)."""
    _, tree = _make_small_semantic_tree()
    alpha = 0.3
    agent = MCTSHNMAgent(tree=tree, exploration_c=2.0, momentum_alpha=alpha, q_norm=True)

    leaf_id = int(tree.leaf_node_ids()[0])
    assert int(tree.visits[leaf_id]) == 0
    assert float(tree.q_values[leaf_id]) == 0.0

    # Step 1: Q^{(1)} = 0.3 * 2.0 + 0.7 * 0.0 = 0.6
    q1 = agent.update(leaf_id, reward=2.0)
    assert int(tree.visits[leaf_id]) == 1
    assert pytest.approx(q1, rel=1e-12, abs=1e-12) == 0.6
    assert pytest.approx(float(tree.q_values[leaf_id]), rel=1e-12, abs=1e-12) == 0.6

    # Step 2: Q^{(2)} = 0.3 * 5.0 + 0.7 * 0.6 = 1.92
    q2 = agent.update(leaf_id, reward=5.0)
    assert int(tree.visits[leaf_id]) == 2
    assert pytest.approx(q2, rel=1e-12, abs=1e-12) == 1.92
    assert pytest.approx(float(tree.q_values[leaf_id]), rel=1e-12, abs=1e-12) == 1.92

    # Step 3: Q^{(3)} = 0.3 * 1.0 + 0.7 * 1.92 = 1.644
    q3 = agent.update(leaf_id, reward=1.0)
    assert int(tree.visits[leaf_id]) == 3
    assert pytest.approx(q3, rel=1e-12, abs=1e-12) == 1.644


def test_backpropagation_weighted_ancestor_aggregation() -> None:
    """Verify ancestor visits and instance-weighted aggregation Q_p = sum(M_c Q_c)/M_p on Semantic and Ward trees (MCTS-02, AC 2)."""
    class_to_indices, sem_tree = _make_small_semantic_tree()
    agent = MCTSHNMAgent(tree=sem_tree, exploration_c=2.0, momentum_alpha=0.5, q_norm=True)

    # Superclass 0 (aquatic_mammals) has beaver (4, M=20) and dolphin (30, M=30) -> M_p = 50
    cls_to_leaf = sem_tree.class_id_to_leaf_node_id()
    leaf_beaver = cls_to_leaf[4]
    leaf_dolphin = cls_to_leaf[30]
    sc_aquatic = int(sem_tree.parents[leaf_beaver])
    assert int(sem_tree.parents[leaf_dolphin]) == sc_aquatic
    assert int(sem_tree.instance_counts[leaf_beaver]) == 20
    assert int(sem_tree.instance_counts[leaf_dolphin]) == 30
    assert int(sem_tree.instance_counts[sc_aquatic]) == 50

    # Update beaver with reward 4.0 (alpha=0.5 -> Q_beaver = 2.0)
    agent.update(leaf_beaver, reward=4.0)
    # Expected Q_{sc_aquatic} = (20 * 2.0 + 30 * 0.0) / 50 = 0.8
    assert pytest.approx(float(sem_tree.q_values[sc_aquatic]), rel=1e-12, abs=1e-12) == 0.8
    # Root has total M_root = 140 -> Expected Q_root = (50 * 0.8) / 140 = 40 / 140
    assert pytest.approx(float(sem_tree.q_values[sem_tree.root_id]), rel=1e-12, abs=1e-12) == (
        40.0 / 140.0
    )

    # Update dolphin with reward 6.0 (alpha=0.5 -> Q_dolphin = 3.0)
    agent.update(leaf_dolphin, reward=6.0)
    # Expected Q_{sc_aquatic} = (20 * 2.0 + 30 * 3.0) / 50 = 130 / 50 = 2.6
    assert pytest.approx(float(sem_tree.q_values[sc_aquatic]), rel=1e-12, abs=1e-12) == 2.6
    assert int(sem_tree.visits[sc_aquatic]) == 2
    assert int(sem_tree.visits[sem_tree.root_id]) == 2
    assert pytest.approx(float(sem_tree.q_values[sem_tree.root_id]), rel=1e-12, abs=1e-12) == (
        130.0 / 140.0
    )

    # Also verify on a deeper binary Ward Confusion Tree
    rng = np.random.default_rng(7)
    conf_mat = rng.uniform(1.0, 10.0, size=(6, 6))
    ward_tree = build_confusion_tree(conf_mat, class_to_indices)
    ward_agent = MCTSHNMAgent(tree=ward_tree, exploration_c=2.0, momentum_alpha=0.3, q_norm=True)

    for step_idx in range(10):
        leaf_id, _ = ward_agent.select_leaf()
        ward_agent.update(leaf_id, reward=float(step_idx + 1) * 0.5)

    # Verify invariant at every internal node of the Ward tree:
    # visits[p] == sum(visits[c]) and q_values[p] == sum(M_c * Q_c) / M_p
    for nid in range(ward_tree.num_nodes):
        if not bool(ward_tree.is_leaf[nid]):
            c_ids = ward_tree.get_children(nid)
            expected_visits = int(ward_tree.visits[c_ids].sum())
            assert int(ward_tree.visits[nid]) == expected_visits
            m_c = ward_tree.instance_counts[c_ids].astype(np.float64)
            m_p = float(ward_tree.instance_counts[nid])
            expected_q = float(np.dot(m_c, ward_tree.q_values[c_ids]) / m_p)
            assert pytest.approx(float(ward_tree.q_values[nid]), rel=1e-12, abs=1e-12) == expected_q


def test_compute_mcts_reward_binary_and_multiclass() -> None:
    """Verify `compute_mcts_reward()` computes exact binary and multi-class cross-entropy rewards (MCTS-02, AC 3)."""
    # 1. Binary Scenario 1: r = -mean(log(1 - sigmoid(z))) = mean(softplus(z))
    logits_bin = np.array([-1.5, 0.0, 2.0], dtype=np.float64)
    probs = 1.0 / (1.0 + np.exp(-logits_bin))
    expected_bin_reward = float(np.mean(-np.log(1.0 - probs)))

    r_np = compute_mcts_reward(logits_bin, is_binary=True)
    r_torch_2d = compute_mcts_reward(
        torch.tensor(logits_bin, dtype=torch.float32).unsqueeze(1), is_binary=True
    )
    assert pytest.approx(r_np, rel=1e-12, abs=1e-12) == expected_bin_reward
    assert pytest.approx(r_torch_2d, rel=1e-6, abs=1e-6) == expected_bin_reward

    # Numerical stability for large positive logit z = 50.0 (where 1 - sigmoid(z) underflows in float64)
    r_extreme = compute_mcts_reward(np.array([50.0]), is_binary=True)
    assert np.isfinite(r_extreme)
    assert pytest.approx(r_extreme, rel=1e-9, abs=1e-9) == 50.0

    # 2. Multi-class Scenario 2: mean cross-entropy loss with scalar or vector target_labels
    logits_mc = torch.tensor([[2.0, -1.0, 0.5], [0.1, 1.5, -0.5]], dtype=torch.float64)
    expected_mc_scalar = float(
        torch.nn.functional.cross_entropy(
            logits_mc, torch.tensor([2, 2], dtype=torch.long), reduction="mean"
        ).item()
    )
    r_mc_scalar = compute_mcts_reward(logits_mc, is_binary=False, target_labels=2)
    r_mc_vec = compute_mcts_reward(
        logits_mc.numpy(), is_binary=False, target_labels=np.array([2, 2], dtype=np.int64)
    )
    assert pytest.approx(r_mc_scalar, rel=1e-12, abs=1e-12) == expected_mc_scalar
    assert pytest.approx(r_mc_vec, rel=1e-12, abs=1e-12) == expected_mc_scalar


def test_backpropagation_invalid_inputs_raise() -> None:
    """Verify `update()` and `compute_mcts_reward()` raise `ValueError` on invalid nodes or non-finite values without mutating tree state (MCTS-02, AC 4)."""
    _, tree = _make_small_semantic_tree()
    agent = MCTSHNMAgent(tree=tree, exploration_c=2.0, momentum_alpha=0.3, q_norm=True)
    leaf_id = int(tree.leaf_node_ids()[0])

    # Internal node rejected
    with pytest.raises(ValueError, match="internal node"):
        agent.update(tree.root_id, reward=1.0)

    # Out-of-range node rejected
    with pytest.raises(ValueError, match="out of range"):
        agent.update(tree.num_nodes + 5, reward=1.0)

    # NaN / Inf rewards rejected without mutating visits or q_values
    with pytest.raises(ValueError, match="finite"):
        agent.update(leaf_id, reward=float("nan"))

    with pytest.raises(ValueError, match="finite"):
        agent.update(leaf_id, reward=float("inf"))

    assert int(tree.visits.sum()) == 0
    assert float(np.abs(tree.q_values).sum()) == 0.0

    # Invalid reward logits rejected
    with pytest.raises(ValueError, match="finite"):
        compute_mcts_reward(np.array([1.0, np.nan]), is_binary=True)

    with pytest.raises(ValueError, match="target_labels"):
        compute_mcts_reward(np.ones((4, 3)), is_binary=False, target_labels=None)

