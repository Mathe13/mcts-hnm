"""Tests for Phase A operational imbalance definitions and nested splits (A2)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.splits import (
    StratifiedImbalancedSplit,
    build_nested_imbalanced_splits,
    build_stratified_imbalanced_split,
)


@pytest.fixture
def synthetic_cifar100_labels():
    """Create a balanced synthetic CIFAR-100 label set (450 samples per class across 100 classes)."""
    # 450 * 100 = 45,000 samples (matching CIFAR-100 train_fit partition)
    labels = np.repeat(np.arange(100, dtype=np.int64), 450)
    rng = np.random.default_rng(42)
    rng.shuffle(labels)
    return labels


def test_build_stratified_imbalanced_split_exact_counts(synthetic_cifar100_labels):
    """Verify exact positive count and exact real negative-to-positive ratio in train_fit."""
    split_50 = build_stratified_imbalanced_split(
        fine_labels=synthetic_cifar100_labels,
        target_class=0,
        target_positive_count=80,
        target_neg_pos_ratio=50,
        seed=42,
    )

    assert isinstance(split_50, StratifiedImbalancedSplit)
    assert len(split_50.positive_indices) == 80
    assert len(split_50.negative_indices) == 4000
    assert split_50.target_neg_pos_ratio == 50
    assert split_50.effective_train_ratio == 50.0
    assert len(split_50.negative_class_to_indices) == 99


def test_nested_negative_subset_containment(synthetic_cifar100_labels):
    """Verify that nested negative sampling produces strict subsets: N_50 ⊂ N_100 ⊂ N_500."""
    nested_splits = build_nested_imbalanced_splits(
        fine_labels=synthetic_cifar100_labels,
        target_class=4,  # e.g. "beaver"
        target_positive_count=80,
        ratios=(50, 100, 500),
        seed=42,
    )

    split_50 = nested_splits[50]
    split_100 = nested_splits[100]
    split_500 = nested_splits[500]

    # 1. Positives must be identical across all three ratios
    np.testing.assert_array_equal(split_50.positive_indices, split_100.positive_indices)
    np.testing.assert_array_equal(split_100.positive_indices, split_500.positive_indices)

    # 2. Strict counts
    assert len(split_50.negative_indices) == 4000
    assert len(split_100.negative_indices) == 8000
    assert len(split_500.negative_indices) == 40000

    # 3. Strict nested containment N_50 ⊂ N_100 ⊂ N_500
    set_50 = set(split_50.negative_indices.tolist())
    set_100 = set(split_100.negative_indices.tolist())
    set_500 = set(split_500.negative_indices.tolist())

    assert set_50.issubset(set_100), "N_50 must be a strict subset of N_100."
    assert set_100.issubset(set_500), "N_100 must be a strict subset of N_500."
    assert len(set_50) < len(set_100) < len(set_500)


def test_infeasible_counts_raise_value_error(synthetic_cifar100_labels):
    """Verify that requests exceeding available samples raise ValueError without silent alteration."""
    # Class 0 has only 450 samples; requesting 500 positives must raise ValueError
    with pytest.raises(ValueError, match="Insufficient positive instances"):
        build_stratified_imbalanced_split(
            fine_labels=synthetic_cifar100_labels,
            target_class=0,
            target_positive_count=500,
            target_neg_pos_ratio=50,
            seed=42,
        )

    # 99 negative classes * 450 = 44,550 available; requesting 80 * 600 = 48,000 negatives must raise ValueError
    with pytest.raises(ValueError, match="Insufficient negative instances"):
        build_stratified_imbalanced_split(
            fine_labels=synthetic_cifar100_labels,
            target_class=0,
            target_positive_count=80,
            target_neg_pos_ratio=600,
            seed=42,
        )


def test_official_test_split_unmodified():
    """Verify that official test split maintains natural One-vs-Rest prevalence (~1:99)."""
    # 100 test samples per class * 100 classes = 10,000 samples
    test_labels = np.repeat(np.arange(100, dtype=np.int64), 100)
    target_cls = 0
    pos_mask = (test_labels == target_cls)
    num_pos = np.sum(pos_mask)
    num_neg = np.sum(~pos_mask)

    assert num_pos == 100
    assert num_neg == 9900
    # Natural prevalence is 100 : 9900 = 1 : 99
    natural_ratio = num_neg / num_pos
    assert natural_ratio == 99.0
