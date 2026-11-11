"""Tests for Phase A hierarchy mathematical validation (A6)."""

from __future__ import annotations

import numpy as np
import pytest

from src.trees.confusion_tree import (
    build_confusion_tree,
    compute_confusion_distance_matrix,
    is_euclidean_dissimilarity,
)


def test_is_euclidean_dissimilarity_schoenberg():
    """Verify Schoenberg Euclidean embedding criterion on known metric and non-metric matrices."""
    # 1. Known Euclidean distance matrix from 2D points
    pts = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    from scipy.spatial.distance import cdist
    d_eucl = cdist(pts, pts, metric="euclidean")
    assert is_euclidean_dissimilarity(d_eucl) is True

    # 2. Non-Euclidean distance violating triangle / Schoenberg PSD condition
    d_non_eucl = np.array([
        [0.0, 1.0, 10.0],
        [1.0, 0.0, 1.0],
        [10.0, 1.0, 0.0],
    ], dtype=np.float64)
    assert is_euclidean_dissimilarity(d_non_eucl) is False


def test_ward_and_average_linkage_properties():
    """Verify that Ward over row-normalized confusion vectors and Average linkage produce monotonic valid trees."""
    k = 5
    # Synthetic confusion matrix with strong diagonal and realistic off-diagonal misclassifications
    rng = np.random.default_rng(42)
    conf = rng.uniform(0.1, 1.0, size=(k, k))
    np.fill_diagonal(conf, 10.0)

    class_to_indices = {i: np.arange(i * 10, (i + 1) * 10) for i in range(k)}

    # 1. Ward linkage over row-normalized confusion vectors
    tree_ward, l_ward, d_ward = build_confusion_tree(
        conf,
        class_to_indices,
        return_linkage=True,
        linkage_method="ward",
    )
    assert tree_ward.num_nodes == 2 * k - 1
    assert tree_ward.num_leaves == k
    # Monotonicity check: merge heights in linkage_matrix must be non-decreasing
    merge_heights_ward = l_ward[:, 2]
    assert np.all(np.diff(merge_heights_ward) >= -1e-12), "Ward dendrogram heights must be monotonically non-decreasing."

    # 2. Average linkage over confusion distance matrix
    tree_avg, l_avg, d_avg = build_confusion_tree(
        conf,
        class_to_indices,
        return_linkage=True,
        linkage_method="average",
    )
    assert tree_avg.num_nodes == 2 * k - 1
    assert tree_avg.num_leaves == k
    merge_heights_avg = l_avg[:, 2]
    assert np.all(np.diff(merge_heights_avg) >= -1e-12), "Average linkage heights must be monotonically non-decreasing."


def test_build_confusion_tree_rejects_unknown_method():
    """Verify that unsupported linkage methods raise ValueError."""
    k = 3
    conf = np.eye(k) * 5.0 + 0.1
    class_to_indices = {i: np.array([i]) for i in range(k)}

    with pytest.raises(ValueError, match="Unsupported linkage_method"):
        build_confusion_tree(
            conf,
            class_to_indices,
            linkage_method="unsupported_method",
        )
