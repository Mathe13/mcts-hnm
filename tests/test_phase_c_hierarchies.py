"""Tests for Phase C hierarchy topology metrics, random tree, and UPGMA validation (C5)."""

from __future__ import annotations

import numpy as np
import pytest

from src.trees.base_tree import FastHierarchyTree
from src.trees.hierarchy_metrics import compute_tree_structural_metrics


def make_test_tree() -> FastHierarchyTree:
    """Create a minimal 5-node hierarchy tree for structural validation.

    Structure:
    - Node 0: root, depth 0, 2 children [1, 2]
    - Node 1: internal, depth 1, 2 children [3, 4]
    - Node 2: leaf, depth 1, 0 children
    - Node 3: leaf, depth 2, 0 children
    - Node 4: leaf, depth 2, 0 children
    Total nodes: 5. Leaves: [2, 3, 4]. Internal: [0, 1].
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
    node_names = ("root", "internal_1", "class_0", "class_1", "class_2")
    class_to_indices = {
        0: np.arange(0, 10, dtype=np.int64),
        1: np.arange(10, 20, dtype=np.int64),
        2: np.arange(20, 30, dtype=np.int64),
    }

    return FastHierarchyTree(
        tree_type="synthetic_test",
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


def test_tree_structural_metrics_calculation() -> None:
    """Verify compute_tree_structural_metrics correctly computes depth, counts, and branching."""
    tree = make_test_tree()
    metrics = compute_tree_structural_metrics(tree)

    assert metrics["total_nodes"] == 5.0
    assert metrics["num_leaves"] == 3.0
    assert metrics["num_internal_nodes"] == 2.0
    assert metrics["max_depth"] == 2.0
    # Leaf depths are [1, 2, 2], mean = 5/3
    assert pytest.approx(metrics["mean_leaf_depth"], rel=1e-5) == 5.0 / 3.0
    # Internal node branching counts are [2, 2], mean = 2.0
    assert pytest.approx(metrics["mean_branching_factor"], rel=1e-5) == 2.0

    # Type validation
    with pytest.raises(ValueError, match="FastHierarchyTree"):
        compute_tree_structural_metrics("not_a_tree")  # type: ignore
