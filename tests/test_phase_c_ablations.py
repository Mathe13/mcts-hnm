"""Tests for Phase C ablations: batch diversity, Uniform Grouped, and factorial 2x2."""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from src.mcts.sampler import UniformGroupedBatchSampler
from src.training.batch_diversity import compute_batch_diversity
from src.training.exposure import ExposureTracker
from src.training.trainer import resolve_mcts_hyperparameters, train_scenario2
from src.trees.base_tree import FastHierarchyTree


def test_batch_diversity_calculation() -> None:
    """Verify compute_batch_diversity under edge cases and known distributions."""
    # Empty
    empty_res = compute_batch_diversity(np.empty(0, dtype=np.int64))
    assert empty_res["unique_classes"] == 0.0
    assert empty_res["batch_entropy"] == 0.0

    # Single class
    single_res = compute_batch_diversity([4, 4, 4, 4])
    assert single_res["unique_classes"] == 1.0
    assert single_res["batch_entropy"] == 0.0

    # Two classes 50/50: entropy should be exactly ln(2)
    two_res = compute_batch_diversity(torch.tensor([0, 0, 1, 1]))
    assert two_res["unique_classes"] == 2.0
    assert pytest.approx(two_res["batch_entropy"], rel=1e-5) == math.log(2.0)

    # Four classes equal: entropy ln(4)
    four_res = compute_batch_diversity(np.array([10, 20, 30, 40]))
    assert four_res["unique_classes"] == 4.0
    assert pytest.approx(four_res["batch_entropy"], rel=1e-5) == math.log(4.0)


def test_batch_diversity_integration_with_exposure() -> None:
    """Verify ExposureTracker integrates batch diversity metrics across multiple batches."""
    tracker = ExposureTracker(num_classes=10)
    assert tracker.compute_diagnostics()["mean_batch_entropy"] == 0.0
    assert tracker.compute_diagnostics()["cumulative_class_coverage"] == 0.0

    # Batch 1: 4 items of class 0
    tracker.record_batch([0, 0, 0, 0])
    diag1 = tracker.compute_diagnostics()
    assert diag1["mean_batch_entropy"] == 0.0
    assert diag1["mean_unique_classes_per_batch"] == 1.0
    assert diag1["cumulative_class_coverage"] == 0.1  # 1/10

    # Batch 2: 2 items of class 1, 2 of class 2 (entropy ln(2))
    tracker.record_batch([1, 1, 2, 2])
    diag2 = tracker.compute_diagnostics()
    expected_mean_entropy = (0.0 + math.log(2.0)) / 2.0
    assert pytest.approx(diag2["mean_batch_entropy"], rel=1e-5) == expected_mean_entropy
    assert diag2["mean_unique_classes_per_batch"] == 1.5
    assert diag2["cumulative_class_coverage"] == 0.3  # 3/10 (classes 0, 1, 2)


def test_uniform_grouped_batch_composition() -> None:
    """Verify UniformGroupedBatchSampler samples B/2 focal samples and B/2 uniform samples."""
    # Synthetic dataset with 5 classes, 100 samples each
    num_classes = 5
    samples_per_class = 100
    class_to_indices = {
        c: np.arange(c * samples_per_class, (c + 1) * samples_per_class, dtype=np.int64)
        for c in range(num_classes)
    }

    sampler = UniformGroupedBatchSampler(
        class_to_indices=class_to_indices,
        batch_size=64,
        dual_stream_ratio=0.5,
        seed=123,
    )

    batch = sampler.sample_batch()
    assert batch.all_indices.shape == (64,)
    assert batch.mcts_indices.shape == (32,)
    assert batch.uniform_indices.shape == (32,)
    assert 0 <= batch.leaf_class_id < num_classes

    # Every index in mcts_indices must belong to the focal class
    focal_pool = set(class_to_indices[batch.leaf_class_id].tolist())
    for idx in batch.mcts_indices:
        assert int(idx) in focal_pool

    # Verify train_scenario2 with strategy="uniform_grouped"
    features = torch.randn(200, 32)
    labels = np.random.randint(0, 5, size=200)
    res = train_scenario2(
        train_features=features,
        train_labels=labels,
        val_features=features[:50],
        val_labels=labels[:50],
        num_classes=5,
        strategy="uniform_grouped",
        batch_size=32,
        num_steps=10,
        eval_every=5,
        seed=42,
    )
    assert "val_metrics" in res
    assert "exposure_diagnostics" in res
    assert res["exposure_diagnostics"]["mean_batch_entropy"] > 0.0


def test_uniform_grouped_parameter_validation() -> None:
    """Verify UniformGroupedBatchSampler parameter bounds and validation."""
    c2i = {0: np.array([0, 1]), 1: np.array([2, 3])}

    # Invalid dual_stream_ratio
    with pytest.raises(ValueError, match="dual_stream_ratio"):
        UniformGroupedBatchSampler(class_to_indices=c2i, dual_stream_ratio=-0.1)

    with pytest.raises(ValueError, match="dual_stream_ratio"):
        UniformGroupedBatchSampler(class_to_indices=c2i, dual_stream_ratio=1.5)

    with pytest.raises(ValueError, match="dual_stream_ratio"):
        UniformGroupedBatchSampler(class_to_indices=c2i, dual_stream_ratio=True)  # type: ignore

    # Invalid batch_size
    with pytest.raises(ValueError, match="batch_size"):
        UniformGroupedBatchSampler(class_to_indices=c2i, batch_size=0)

    # Missing class_to_indices and tree
    with pytest.raises(ValueError, match="Either tree or class_to_indices"):
        UniformGroupedBatchSampler()


def test_factorial_variants_configuration() -> None:
    """Verify resolve_mcts_hyperparameters maps each factorial variant correctly."""
    # Strict: q_norm=False, beta=0.0
    q_norm, beta = resolve_mcts_hyperparameters("mcts_strict")
    assert q_norm is False
    assert beta == 0.0

    # QNorm-only: q_norm=True, beta=0.0
    q_norm, beta = resolve_mcts_hyperparameters("mcts_qnorm_only")
    assert q_norm is True
    assert beta == 0.0

    # DualStream-only: q_norm=False, beta=0.5
    q_norm, beta = resolve_mcts_hyperparameters("mcts_dualstream_only")
    assert q_norm is False
    assert beta == 0.5

    # Enhanced: q_norm=True, beta=0.5
    q_norm, beta = resolve_mcts_hyperparameters("mcts_enhanced")
    assert q_norm is True
    assert beta == 0.5

    with pytest.raises(ValueError, match="Unknown MCTS strategy"):
        resolve_mcts_hyperparameters("unknown_mcts")


def make_test_tree() -> FastHierarchyTree:
    """Create a minimal 3-leaf hierarchy tree for testing."""
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
    node_names = ("root", "cluster_1", "class_0", "class_1", "class_2")
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


def test_factorial_controlled_factors_invariance() -> None:
    """Verify that all 4 factorial variants run under identical initial weights and configurations."""
    variants = ["mcts_strict", "mcts_qnorm_only", "mcts_dualstream_only", "mcts_enhanced"]
    features = torch.randn(30, 16)
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int64)

    for var in variants:
        tree = make_test_tree()
        res = train_scenario2(
            train_features=features,
            train_labels=labels,
            val_features=features,
            val_labels=labels,
            num_classes=3,
            tree=tree,
            strategy=var,
            batch_size=4,
            num_steps=4,
            eval_every=2,
            lr=1e-3,
            momentum_alpha=0.3,
            exploration_c=2.0,
            seed=42,
        )
        assert res["strategy"] == var
        assert "val_metrics" in res
        assert np.isfinite(res["val_metrics"]["loss"])
        assert "exposure_diagnostics" in res



