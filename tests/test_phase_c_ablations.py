"""Tests for Phase C ablations: batch diversity, Uniform Grouped, and factorial 2x2."""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from src.training.batch_diversity import compute_batch_diversity
from src.training.exposure import ExposureTracker


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
