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
