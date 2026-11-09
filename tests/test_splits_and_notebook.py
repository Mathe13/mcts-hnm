"""Unit tests for DATA-03: Scenario 1 and Scenario 2 splitters."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.splits import (
    DEFAULT_SCENARIO1_TARGETS,
    IMBALANCE_RATIOS,
    build_scenario1_split,
    build_scenario2_split,
)


def test_build_scenario1_split_ratios_and_clamping() -> None:
    """Verify DATA-03 AC 1 & Edge Case 3: exact positive counts for 1:10, 1:50, 1:100 and clamping to 1."""
    # Simulate 45,000 stratified training labels (450 per fine class)
    labels = torch.tensor(np.repeat(np.arange(100, dtype=np.int64), 450))

    expected_pos_counts = {
        "1:10": 45,   # floor(0.10 * 450)
        "1:50": 9,    # floor(0.02 * 450)
        "1:100": 4,   # floor(0.01 * 450)
    }

    for ratio_name, frac in IMBALANCE_RATIOS.items():
        split = build_scenario1_split(labels, target_class=73, pos_fraction=frac, seed=42)
        assert split.target_class_id == 73
        assert split.target_class_name == "shark"
        assert split.positive_indices.shape == (expected_pos_counts[ratio_name],)
        assert len(split.negative_class_to_indices) == 99
        assert 73 not in split.negative_class_to_indices
        assert split.negative_indices.shape == (99 * 450,)
        assert np.intersect1d(split.positive_indices, split.negative_indices).size == 0
        assert np.all(split.binary_labels[split.positive_indices] == 1)
        assert np.all(split.binary_labels[split.negative_indices] == 0)

    # Edge Case 3: pos_fraction * N_target < 1.0 clamps to 1 positive instance
    tiny_split = build_scenario1_split(labels, target_class="beaver", pos_fraction=0.0001, seed=42)
    assert tiny_split.positive_indices.shape == (1,)


def test_build_scenario1_split_by_name_and_seed_determinism() -> None:
    """Verify Scenario 1 default target resolution across 5 superclasses and seed reproducibility."""
    labels = np.repeat(np.arange(100, dtype=np.int64), 450)
    assert len(DEFAULT_SCENARIO1_TARGETS) == 5

    for target_name in DEFAULT_SCENARIO1_TARGETS:
        s1 = build_scenario1_split(labels, target_class=target_name, pos_fraction=0.10, seed=99)
        s2 = build_scenario1_split(labels, target_class=target_name, pos_fraction=0.10, seed=99)
        assert s1.target_class_name == target_name
        assert np.array_equal(s1.positive_indices, s2.positive_indices)


def test_build_scenario1_split_invalid_args_raise() -> None:
    """Verify DATA-03 AC 2: invalid target_class or pos_fraction raises ValueError."""
    labels = np.repeat(np.arange(100, dtype=np.int64), 10)

    for bad_target in (-1, 100, 999, "not_a_cifar_class", True):
        with pytest.raises(ValueError):
            build_scenario1_split(labels, target_class=bad_target, pos_fraction=0.1, seed=42)  # type: ignore[arg-type]

    for bad_frac in (0.0, -0.5, 1.01, 2.0, True):
        with pytest.raises(ValueError, match="pos_fraction"):
            build_scenario1_split(labels, target_class=0, pos_fraction=bad_frac, seed=42)  # type: ignore[arg-type]


def test_build_scenario2_split_complete() -> None:
    """Verify DATA-03 AC 3: Scenario 2 maps all 100 fine classes without dropping any."""
    labels = np.repeat(np.arange(100, dtype=np.int64), 450)
    split = build_scenario2_split(labels)

    assert len(split.class_to_indices) == 100
    assert set(split.class_to_indices.keys()) == set(range(100))
    assert split.active_indices.shape == (45000,)
    for cls_id, idx_arr in split.class_to_indices.items():
        assert idx_arr.shape == (450,)
        assert np.all(labels[idx_arr] == cls_id)


def test_build_scenario2_split_missing_class_raises() -> None:
    """Verify build_scenario2_split raises ValueError if a fine class is missing."""
    # Only classes 0..98 present
    incomplete_labels = np.repeat(np.arange(99, dtype=np.int64), 10)
    with pytest.raises(ValueError, match="Fine class 99 is missing"):
        build_scenario2_split(incomplete_labels)
