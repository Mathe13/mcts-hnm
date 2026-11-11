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
