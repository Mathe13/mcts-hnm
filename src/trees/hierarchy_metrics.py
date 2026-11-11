"""Structural and topological metrics for hierarchy trees (Phase C5)."""

from __future__ import annotations

from typing import Any
import numpy as np

from src.trees.base_tree import FastHierarchyTree


def compute_tree_structural_metrics(tree: FastHierarchyTree) -> dict[str, float]:
    r"""Compute structural topology metrics for a `FastHierarchyTree`.

    Metrics computed:
    - `max_depth`: Maximum depth from root (depth 0) to any leaf.
    - `mean_leaf_depth`: Mean depth of all leaf nodes: $\frac{1}{K} \sum_{l=1}^K \text{depth}(l)$.
    - `total_nodes`: Total number of nodes $V$.
    - `num_leaves`: Number of leaf nodes $K$.
    - `num_internal_nodes`: Number of internal nodes $V - K$.
    - `mean_branching_factor`: Average number of children per internal node.

    Args:
        tree: An instance of `FastHierarchyTree`.

    Returns:
        Dictionary mapping metric names to float values.
    """
    if not isinstance(tree, FastHierarchyTree):
        raise ValueError(f"tree must be a FastHierarchyTree instance, got {type(tree).__name__}.")

    total_nodes = int(tree.num_nodes)
    num_leaves = int(tree.num_leaves)
    num_internal = total_nodes - num_leaves

    max_depth = float(np.max(tree.depths)) if total_nodes > 0 else 0.0
    leaf_depths = tree.depths[tree.is_leaf]
    mean_leaf_depth = float(np.mean(leaf_depths)) if len(leaf_depths) > 0 else 0.0

    internal_counts = tree.child_counts[~tree.is_leaf]
    mean_branching = float(np.mean(internal_counts)) if len(internal_counts) > 0 else 0.0

    return {
        "max_depth": max_depth,
        "mean_leaf_depth": mean_leaf_depth,
        "total_nodes": float(total_nodes),
        "num_leaves": float(num_leaves),
        "num_internal_nodes": float(num_internal),
        "mean_branching_factor": mean_branching,
    }
