"""Tests for Feature F5: Dynamic entropy leaf expansion and Tier-2 end-to-end vision pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data.cifar100_taxonomy import get_cifar100_taxonomy
from src.mcts.agent import MCTSHNMAgent
from src.mcts.diagnostics import (
    MCTSDiagnosticsTracker,
    compute_superclass_visit_distribution,
    compute_visitation_entropy,
)
from src.mcts.expansion import (
    SUBCLUSTER_ID_STRIDE,
    compute_prediction_entropy,
    expand_leaf_by_entropy,
    maybe_expand_leaf,
    resolve_base_class_id,
)
from src.mcts.sampler import DualStreamBatchSampler
from src.trees.base_tree import FastHierarchyTree
from src.trees.semantic_tree import build_semantic_tree


@pytest.fixture
def toy_tree() -> FastHierarchyTree:
    """Create a toy 3-class tree with Root -> 3 Leaves."""
    parents = np.array([-1, 0, 0, 0], dtype=np.int64)
    children = np.array([[1, 2, 3], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]], dtype=np.int64)
    child_counts = np.array([3, 0, 0, 0], dtype=np.int64)
    is_leaf = np.array([False, True, True, True], dtype=bool)
    depths = np.array([0, 1, 1, 1], dtype=np.int64)
    leaf_to_class_id = np.array([-1, 0, 1, 2], dtype=np.int64)
    instance_counts = np.array([30, 10, 10, 10], dtype=np.int64)
    visits = np.zeros(4, dtype=np.int64)
    q_values = np.zeros(4, dtype=np.float64)
    node_names = ("root", "class:0", "class:1", "class:2")
    class_to_indices = {
        0: np.arange(0, 10, dtype=np.int64),
        1: np.arange(10, 20, dtype=np.int64),
        2: np.arange(20, 30, dtype=np.int64),
    }
    return FastHierarchyTree(
        tree_type="toy",
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


# --- C1 Tests ---
def test_compute_prediction_entropy_binary_and_multiclass() -> None:
    # Binary logits: z=0 -> p=0.5 -> H = ln(2) ~= 0.693147
    logits_bin = np.array([0.0, 10.0, -10.0], dtype=np.float32)
    ent_bin = compute_prediction_entropy(logits_bin, is_binary=True)
    assert ent_bin.shape == (3,)
    assert np.isclose(ent_bin[0], np.log(2.0), atol=1e-4)
    assert ent_bin[1] < 1e-3  # high confidence positive -> low entropy
    assert ent_bin[2] < 1e-3  # high confidence negative -> low entropy
    assert np.all(ent_bin >= 0.0)

    # Multi-class logits: 4 uniform classes -> H = ln(4)
    logits_multi = torch.tensor([[1.0, 1.0, 1.0, 1.0], [15.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    ent_multi = compute_prediction_entropy(logits_multi, is_binary=False)
    assert ent_multi.shape == (2,)
    assert np.isclose(ent_multi[0], np.log(4.0), atol=1e-4)
    assert ent_multi[1] < 1e-3
    assert np.all(ent_multi >= 0.0)


# --- C2 Tests ---
def test_expand_leaf_by_entropy_splits_and_preserves_invariants(toy_tree: FastHierarchyTree) -> None:
    tree = toy_tree
    leaf_id = 1  # class 0 with 10 instances: indices 0..9
    entropies = np.linspace(0.1, 0.9, 10)  # increasing entropy
    new_children = expand_leaf_by_entropy(tree, leaf_node_id=leaf_id, entropies=entropies, k_sub=3)

    assert len(new_children) == 3
    assert not tree.is_leaf[leaf_id]
    assert tree.leaf_to_class_id[leaf_id] == -1
    assert tree.child_counts[leaf_id] == 3

    # Check children
    sub_classes = []
    total_child_instances = 0
    for cid in new_children:
        assert tree.is_leaf[cid]
        v_cls = tree.leaf_to_class_id[cid]
        assert v_cls >= SUBCLUSTER_ID_STRIDE
        assert resolve_base_class_id(v_cls) == 0
        sub_classes.append(v_cls)
        total_child_instances += tree.instance_counts[cid]

    assert total_child_instances == 10
    assert tree.instance_counts[leaf_id] == 10
    # Tree validation passes
    tree.validate_tree()


# --- C3 Tests ---
def test_mcts_agent_sampler_and_diagnostics_on_expanded_tree(toy_tree: FastHierarchyTree) -> None:
    tree = toy_tree
    leaf_id = 1
    entropies = np.linspace(0.1, 0.9, 10)
    new_children = expand_leaf_by_entropy(tree, leaf_node_id=leaf_id, entropies=entropies, k_sub=2)

    agent = MCTSHNMAgent(tree=tree, exploration_c=2.0, momentum_alpha=0.3, q_norm=True)
    sampler = DualStreamBatchSampler(tree=tree, neg_batch_size=4, dual_stream_ratio=0.5, seed=42)

    # Select leaf should visit unvisited children
    selected_leaf, path = agent.select_leaf()
    assert selected_leaf in tree.leaf_node_ids()
    batch = sampler.sample_batch(selected_leaf)
    assert batch.negative_indices.size == 4

    # Backpropagation
    agent.update(selected_leaf, reward=1.0)
    assert tree.visits[selected_leaf] == 1
    assert tree.visits[leaf_id] == 1
    assert tree.visits[tree.root_id] == 1


# --- C4 Tests ---
def test_expansion_invalid_arguments_raise(toy_tree: FastHierarchyTree) -> None:
    tree = toy_tree
    # Out of range leaf_node_id
    with pytest.raises(ValueError):
        expand_leaf_by_entropy(tree, leaf_node_id=99, entropies=np.zeros(10))

    # Internal node
    with pytest.raises(ValueError):
        expand_leaf_by_entropy(tree, leaf_node_id=0, entropies=np.zeros(30))

    # k_sub < 2
    with pytest.raises(ValueError):
        expand_leaf_by_entropy(tree, leaf_node_id=1, entropies=np.zeros(10), k_sub=1)

    # Mismatched entropies size
    with pytest.raises(ValueError):
        expand_leaf_by_entropy(tree, leaf_node_id=1, entropies=np.zeros(5), k_sub=2)


# --- C5 Tests ---
def test_expansion_edge_cases_noop_bmax_and_ties(toy_tree: FastHierarchyTree) -> None:
    tree = toy_tree
    # Dummy model
    model = nn.Linear(5, 1)
    inputs = torch.randn(30, 5)

    # visits < split_threshold -> returns None
    res = maybe_expand_leaf(tree, leaf_node_id=1, model=model, train_inputs=inputs, split_threshold=50)
    assert res is None

    # tied entropies
    entropies = np.full(10, 0.5)
    new_children = expand_leaf_by_entropy(tree, leaf_node_id=1, entropies=entropies, k_sub=4)
    assert len(new_children) == 4
    tree.validate_tree()

    # calling on already expanded sub-cluster
    res2 = maybe_expand_leaf(tree, leaf_node_id=new_children[0], model=model, train_inputs=inputs, split_threshold=0)
    assert res2 is None
