"""Tests for Phase B exposure tracking and observability (B1)."""

from __future__ import annotations

import numpy as np
import pytest

from src.training.exposure import ExposureTracker


def test_exposure_entropy_zero_selections():
    """Verify that normalized entropy is 0.0 when no instances are recorded (OBS-02, AC 4)."""
    tracker = ExposureTracker(num_classes=10)
    diag = tracker.compute_diagnostics()

    assert diag["total_instances_selected"] == 0
    assert diag["unique_instances_selected"] == 0
    assert diag["normalized_exposure_entropy"] == 0.0
    assert diag["unexposed_classes_count"] == 10
    assert diag["worst10_class_recall"] is None


def test_exposure_entropy_and_worst10_calculation():
    """Verify normalized entropy in [0, 1] and Worst-10 recall computation (OBS-02, AC 3, 5)."""
    k = 20
    tracker = ExposureTracker(num_classes=k)

    # 1. Uniform distribution across all k classes -> entropy must be ~1.0
    batch_classes = np.repeat(np.arange(k), 10)
    batch_instances = np.arange(k * 10)
    tracker.record_batch(batch_classes, batch_instances)
    tracker.record_leaf_selection(5)
    tracker.record_node_visit(0)

    val_recalls = {i: 0.1 * (i + 1) for i in range(k)}  # recalls from 0.1 to 2.0
    diag = tracker.compute_diagnostics(val_class_recalls=val_recalls)

    assert diag["total_instances_selected"] == k * 10
    assert diag["unique_instances_selected"] == k * 10
    assert diag["total_leaf_selections"] == 1
    assert diag["total_node_visits"] == 1
    assert diag["unexposed_classes_count"] == 0
    assert np.isclose(diag["normalized_exposure_entropy"], 1.0, atol=1e-5)

    # Worst-10 recall: average of first 10 classes (0.1, 0.2, ..., 1.0) -> mean = 0.55
    expected_worst10 = np.mean([0.1 * (i + 1) for i in range(10)])
    assert np.isclose(diag["worst10_class_recall"], expected_worst10, atol=1e-5)

    # 2. Skewed distribution: all samples in a single class -> entropy must be 0.0
    skewed_tracker = ExposureTracker(num_classes=k)
    skewed_tracker.record_batch(np.zeros(50, dtype=np.int64))
    skewed_diag = skewed_tracker.compute_diagnostics()
    assert skewed_diag["normalized_exposure_entropy"] == 0.0
    assert skewed_diag["unexposed_classes_count"] == k - 1


def test_scenario1_step_history_logging():
    """Verify that train_scenario1 logs complete step_history with loss, metrics, and lr (OBS-01, AC 1, 3)."""
    import torch
    from src.training.trainer import train_scenario1

    num_samples = 40
    feat_dim = 16
    train_f = torch.randn(num_samples, feat_dim)
    train_y = np.array([0] * 20 + [1] * 20, dtype=np.int64)
    pos_idx = np.arange(20, dtype=np.int64)
    neg_idx = np.arange(20, 40, dtype=np.int64)

    val_f = torch.randn(10, feat_dim)
    val_y = np.array([0] * 5 + [1] * 5, dtype=np.int64)

    res = train_scenario1(
        train_features=train_f,
        train_binary_labels=train_y,
        positive_indices=pos_idx,
        negative_indices=neg_idx,
        val_features=val_f,
        val_binary_labels=val_y,
        strategy="uniform",
        num_steps=10,
        eval_every=5,
        pos_batch_size=4,
        neg_batch_size=8,
    )

    assert "step_history" in res
    assert "exposure_diagnostics" in res
    step_history = res["step_history"]
    assert len(step_history) == 2  # steps 5 and 10

    for entry in step_history:
        assert "train_loss" in entry and np.isfinite(entry["train_loss"])
        assert "val_loss" in entry and np.isfinite(entry["val_loss"])
        assert "val_F1" in entry and 0.0 <= entry["val_F1"] <= 1.0
        assert "val_AUPRC" in entry and 0.0 <= entry["val_AUPRC"] <= 1.0
        assert "learning_rate" in entry and entry["learning_rate"] > 0
        assert "step" in entry
        assert "elapsed_time" in entry and entry["elapsed_time"] >= 0.0

    diag = res["exposure_diagnostics"]
    assert diag["total_instances_selected"] > 0
    assert 0.0 <= diag["normalized_exposure_entropy"] <= 1.0


def test_scenario2_step_history_logging():
    """Verify that train_scenario2 logs macro_f1 and accuracy in step_history (OBS-01, AC 2)."""
    import torch
    from src.training.trainer import train_scenario2

    num_classes = 4
    samples_per_class = 15
    feat_dim = 16
    train_f = torch.randn(num_classes * samples_per_class, feat_dim)
    train_y = np.repeat(np.arange(num_classes), samples_per_class)

    val_f = torch.randn(num_classes * 5, feat_dim)
    val_y = np.repeat(np.arange(num_classes), 5)

    res = train_scenario2(
        train_features=train_f,
        train_labels=train_y,
        val_features=val_f,
        val_labels=val_y,
        strategy="uniform",
        num_classes=num_classes,
        num_steps=6,
        eval_every=3,
        batch_size=8,
    )

    assert "step_history" in res
    assert "exposure_diagnostics" in res
    step_history = res["step_history"]
    assert len(step_history) == 2  # steps 3 and 6

    for entry in step_history:
        assert "macro_f1" in entry and 0.0 <= entry["macro_f1"] <= 1.0
        assert "accuracy" in entry and 0.0 <= entry["accuracy"] <= 1.0
        assert "train_loss" in entry and np.isfinite(entry["train_loss"])
        assert "val_loss" in entry and np.isfinite(entry["val_loss"])

    diag = res["exposure_diagnostics"]
    assert diag["total_instances_selected"] == 6 * 8
