"""Tests for Phase D: Computational Cost Telemetry, Metrics, and Dual Checkpoints."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.mlp_head import Tier1MLPHead
from src.training.trainer import (
    evaluate_scenario1_metrics,
    evaluate_scenario2_metrics,
    train_scenario1,
    train_scenario2,
)
from src.trees.base_tree import FastHierarchyTree


def _make_dummy_tree() -> FastHierarchyTree:
    class_to_indices = {i: np.arange(i * 10, (i + 1) * 10, dtype=np.int64) for i in range(10)}
    node_id_to_classes = {i: [i] for i in range(10)}
    node_id_to_classes[10] = list(range(5))
    node_id_to_classes[11] = list(range(5, 10))
    node_id_to_classes[12] = list(range(10))

    tree = FastHierarchyTree(
        children={12: [10, 11], 10: [0, 1, 2, 3, 4], 11: [5, 6, 7, 8, 9]},
        parents={i: 10 for i in range(5)} | {i: 11 for i in range(5, 10)} | {10: 12, 11: 12, 12: None},
        root_id=12,
        leaf_nodes=list(range(10)),
        class_to_indices=class_to_indices,
        node_id_to_classes=node_id_to_classes,
        visits=np.zeros(13, dtype=np.int64),
        total_q=np.zeros(13, dtype=np.float64),
        tree_id="test_tree",
    )
    return tree


def test_scenario1_computes_auprc_and_calibrated_f1() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    features = torch.randn(100, 16)
    binary_labels = np.zeros(100, dtype=np.int64)
    binary_labels[:10] = 1

    model = Tier1MLPHead(input_dim=16, hidden_dim=32, num_outputs=1)
    metrics = evaluate_scenario1_metrics(model, features, binary_labels, threshold=0.3)

    assert "auprc" in metrics
    assert "calibrated_f1" in metrics
    assert "calibrated_threshold" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert metrics["calibrated_threshold"] == 0.3
    assert 0.0 <= metrics["auprc"] <= 1.0


def test_scenario2_computes_macro_f1_and_worst10() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    features = torch.randn(100, 16)
    labels = np.random.randint(0, 10, size=100)

    model = Tier1MLPHead(input_dim=16, hidden_dim=32, num_outputs=10)
    metrics = evaluate_scenario2_metrics(model, features, labels, num_classes=10)

    assert "macro_f1" in metrics
    assert "balanced_accuracy" in metrics
    assert "worst_10_class_recall" in metrics
    assert "top1_accuracy" in metrics
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_computational_cost_telemetry_recording() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    train_x = torch.randn(80, 16)
    train_y_bin = np.zeros(80, dtype=np.int64)
    train_y_bin[:10] = 1

    val_x = torch.randn(40, 16)
    val_y_bin = np.zeros(40, dtype=np.int64)
    val_y_bin[:5] = 1

    pos_idx = np.where(train_y_bin == 1)[0]
    neg_idx = np.where(train_y_bin == 0)[0]

    # Scenario 1 - OHEM
    res_ohem = train_scenario1(
        train_features=train_x,
        train_binary_labels=train_y_bin,
        positive_indices=pos_idx,
        negative_indices=neg_idx,
        val_features=val_x,
        val_binary_labels=val_y_bin,
        strategy="ohem",
        num_steps=5,
        eval_every=5,
        pos_batch_size=4,
        neg_batch_size=12,
        seed=42,
    )
    costs = res_ohem["computational_costs"]
    assert costs["total_updates"] == 5
    assert costs["gradient_samples"] == 5 * (4 + 12)
    assert costs["mining_forwards"] == 5 * 4 * 12  # 4 * neg_batch_size per step
    assert costs["elapsed_seconds"] >= 0.0

    # Scenario 2 - Uniform
    train_y_multi = np.random.randint(0, 10, size=80)
    val_y_multi = np.random.randint(0, 10, size=40)
    res_uni = train_scenario2(
        train_features=train_x,
        train_labels=train_y_multi,
        val_features=val_x,
        val_labels=val_y_multi,
        strategy="uniform",
        num_classes=10,
        num_steps=6,
        eval_every=3,
        batch_size=16,
        seed=42,
    )
    costs2 = res_uni["computational_costs"]
    assert costs2["total_updates"] == 6
    assert costs2["gradient_samples"] == 6 * 16
    assert costs2["mining_forwards"] == 0
    assert costs2["elapsed_seconds"] >= 0.0


def test_dual_checkpoint_metrics_distinction() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    train_x = torch.randn(100, 16)
    train_y_bin = np.zeros(100, dtype=np.int64)
    train_y_bin[:15] = 1

    val_x = torch.randn(50, 16)
    val_y_bin = np.zeros(50, dtype=np.int64)
    val_y_bin[:8] = 1

    test_x = torch.randn(50, 16)
    test_y_bin = np.zeros(50, dtype=np.int64)
    test_y_bin[:8] = 1

    pos_idx = np.where(train_y_bin == 1)[0]
    neg_idx = np.where(train_y_bin == 0)[0]

    res = train_scenario1(
        train_features=train_x,
        train_binary_labels=train_y_bin,
        positive_indices=pos_idx,
        negative_indices=neg_idx,
        val_features=val_x,
        val_binary_labels=val_y_bin,
        test_features=test_x,
        test_binary_labels=test_y_bin,
        strategy="uniform",
        num_steps=10,
        eval_every=2,
        pos_batch_size=4,
        neg_batch_size=12,
        seed=42,
    )

    assert "last_metrics" in res
    assert "best_validation_metrics" in res
    assert "auprc" in res["last_metrics"]
    assert "calibrated_f1" in res["last_metrics"]
    assert "auprc" in res["best_validation_metrics"]
    assert "calibrated_f1" in res["best_validation_metrics"]
    assert res["best_model_state"] is not None
