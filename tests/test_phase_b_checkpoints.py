"""Tests for Phase B checkpoints and threshold calibration (B3)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.training.checkpoints import CheckpointManager
from src.trees.base_tree import FastHierarchyTree


def test_checkpoint_save_and_restore_roundtrip(tmp_path: Path):
    """Verify that CheckpointManager saves and restores last and best_validation deterministically (CKPT-01, AC 1)."""
    manager = CheckpointManager(checkpoint_dir=tmp_path, run_id="run_test_001")

    # 1. Dummy model, optimizer, tree
    model = nn.Linear(8, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Simple tree
    tree = FastHierarchyTree(
        tree_type="dummy",
        root_id=0,
        parents=np.array([-1, 0, 0]),
        children=np.array([[1, 2], [-1, -1], [-1, -1]]),
        child_counts=np.array([2, 0, 0]),
        is_leaf=np.array([False, True, True]),
        depths=np.array([0, 1, 1]),
        leaf_to_class_id=np.array([-1, 0, 1]),
        instance_counts=np.array([20, 10, 10]),
        visits=np.array([5, 3, 2]),
        q_values=np.array([0.5, 0.6, 0.4]),
        node_names=["root", "leaf_0", "leaf_1"],
        class_to_indices={0: np.arange(10), 1: np.arange(10, 20)},
    )

    metrics = {"val_loss": 0.32, "val_F1": 0.85}

    # 2. Save best_validation and last
    best_path = manager.save_checkpoint("best_validation", model, optimizer, step=50, metrics=metrics, tree=tree)
    last_path = manager.save_checkpoint("last", model, optimizer, step=100, metrics={"val_loss": 0.40}, tree=tree)

    assert best_path.exists()
    assert last_path.exists()

    # 3. Create fresh model and tree, then restore from best_validation
    fresh_model = nn.Linear(8, 2)
    fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=0.01)
    fresh_tree = FastHierarchyTree(
        tree_type="dummy",
        root_id=0,
        parents=np.array([-1, 0, 0]),
        children=np.array([[1, 2], [-1, -1], [-1, -1]]),
        child_counts=np.array([2, 0, 0]),
        is_leaf=np.array([False, True, True]),
        depths=np.array([0, 1, 1]),
        leaf_to_class_id=np.array([-1, 0, 1]),
        instance_counts=np.array([20, 10, 10]),
        visits=np.zeros(3, dtype=np.int64),
        q_values=np.zeros(3, dtype=np.float64),
        node_names=["root", "leaf_0", "leaf_1"],
        class_to_indices={0: np.arange(10), 1: np.arange(10, 20)},
    )

    ckpt = CheckpointManager.load_checkpoint(best_path, model=fresh_model, optimizer=fresh_optimizer, tree=fresh_tree)

    assert ckpt["step"] == 50
    assert ckpt["metrics"]["val_F1"] == 0.85

    # Check weights match exactly
    for p_orig, p_fresh in zip(model.parameters(), fresh_model.parameters()):
        assert torch.equal(p_orig, p_fresh)

    # Check tree state restored
    assert np.array_equal(fresh_tree.visits, np.array([5, 3, 2]))
    assert np.allclose(fresh_tree.q_values, np.array([0.5, 0.6, 0.4]))


def test_binary_threshold_calibration_optimizes_f1():
    """Verify that calibrate_binary_threshold finds tau_val maximizing validation F1 (CKPT-02, AC 2)."""
    from src.utils.metrics import calibrate_binary_threshold

    # Synthetic imbalanced validation probabilities where positive signals peak around 0.35
    # targets: 10 positives, 90 negatives
    np.random.seed(42)
    val_probs = np.concatenate([
        np.random.uniform(0.3, 0.6, size=10),   # positives centered at ~0.45
        np.random.uniform(0.01, 0.25, size=90), # negatives centered at ~0.13
    ])
    val_targets = np.concatenate([np.ones(10, dtype=np.int64), np.zeros(90, dtype=np.int64)])

    tau_val = calibrate_binary_threshold(val_probs, val_targets, num_thresholds=100)

    # Optimal threshold should be around 0.20 to 0.40 to perfectly separate positives
    assert 0.20 <= tau_val <= 0.40

    # Under tau_val, F1 must be higher than or equal to default tau = 0.5
    from sklearn.metrics import f1_score
    f1_calibrated = f1_score(val_targets, (val_probs >= tau_val).astype(int))
    f1_default = f1_score(val_targets, (val_probs >= 0.5).astype(int))
    assert f1_calibrated >= f1_default


def test_scenario1_reports_calibrated_f1_and_auprc():
    """Verify that evaluate_scenario1_with_calibration reports default F1, calibrated F1, and AUPRC (CKPT-02, AC 3)."""
    from src.utils.metrics import evaluate_scenario1_with_calibration

    val_probs = np.array([0.1, 0.2, 0.4, 0.8, 0.3, 0.15])
    val_targets = np.array([0, 0, 1, 1, 1, 0])

    test_probs = np.array([0.1, 0.35, 0.7, 0.12])
    test_targets = np.array([0, 1, 1, 0])

    eval_res = evaluate_scenario1_with_calibration(val_probs, val_targets, test_probs, test_targets)

    assert "default_f1" in eval_res
    assert "calibrated_f1" in eval_res
    assert "calibrated_threshold" in eval_res
    assert "auprc" in eval_res
    assert 0.0 <= eval_res["default_f1"] <= 1.0
    assert 0.0 <= eval_res["calibrated_f1"] <= 1.0
    assert 0.0 <= eval_res["auprc"] <= 1.0
    assert 0.01 <= eval_res["calibrated_threshold"] <= 0.99
