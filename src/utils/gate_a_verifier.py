"""Gate A Verifier: Audits and verifies all Phase A integrity invariants."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import numpy as np
import torch

from src.data.splits import build_nested_imbalanced_splits
from src.training.runner_config import ResolvedTrainingConfig
from src.training.trainer import train_scenario1
from src.trees.confusion_tree import (
    build_confusion_tree,
    estimate_validation_confusion_matrix,
    is_euclidean_dissimilarity,
)
from src.utils.experiment_config import assert_path_mutable
from src.utils.statistics import compute_paired_statistical_tests

logger = logging.getLogger(__name__)


def verify_gate_a(verbose: bool = True) -> int:
    """Run all Phase A verification checks and return 0 on success.

    Checks:
    - A1: Experimental audit documentation exists and is non-empty.
    - A1: Legacy results immutability check.
    - A2: Stratified nested imbalanced splits (1:50, 1:100, 1:500) with strict subset nesting.
    - A3: Absolute data segregation preventing test data usage in validation/calibration.
    - A4: Resolved training configuration governance and immutability.
    - A5: Paired statistical testing rejects key mismatches.
    - A6: Tree hierarchy validation (Schoenberg dissimilarity, Euclidean Ward, and Average linkage).

    Returns:
        0 if all checks pass, 1 if any check fails.
    """
    passed = 0
    total = 7

    # Check 1: Audit documentation
    doc_path = Path("docs/experimental_audit.md")
    if not doc_path.exists() or doc_path.stat().st_size == 0:
        logger.error("Gate A Check 1 FAILED: docs/experimental_audit.md missing or empty")
        return 1
    passed += 1
    if verbose:
        print("[Gate A] Check 1/7 (Audit documentation) PASSED")

    # Check 2: Legacy immutability
    try:
        assert_path_mutable(Path("results/legacy/run_manifest.json"))
        logger.error("Gate A Check 2 FAILED: results/legacy was not protected")
        return 1
    except PermissionError:
        passed += 1
        if verbose:
            print("[Gate A] Check 2/7 (Legacy immutability) PASSED")

    # Check 3: Nested imbalanced splits
    num_classes = 100
    samples_per_class = 450
    labels = np.repeat(np.arange(num_classes), samples_per_class)
    splits = build_nested_imbalanced_splits(
        fine_labels=labels,
        target_class=0,
        ratios=(50, 100, 500),
        target_positive_count=80,
        seed=42,
    )
    neg_50 = set(splits[50].negative_indices)
    neg_100 = set(splits[100].negative_indices)
    neg_500 = set(splits[500].negative_indices)

    if not (len(neg_50) == 4000 and len(neg_100) == 8000 and len(neg_500) == 40000):
        logger.error("Gate A Check 3 FAILED: Incorrect negative sample counts in nested splits")
        return 1
    if not (neg_50.issubset(neg_100) and neg_100.issubset(neg_500)):
        logger.error("Gate A Check 3 FAILED: Subset nesting property violated")
        return 1
    passed += 1
    if verbose:
        print("[Gate A] Check 3/7 (Nested imbalanced splits) PASSED")

    # Check 4: Data segregation
    try:
        mock_feat = torch.randn(10, 8)
        mock_y = np.array([1, 0] * 5, dtype=np.int64)
        pos_idx = np.array([0, 2], dtype=np.int64)
        neg_idx = np.array([1, 3], dtype=np.int64)
        train_scenario1(
            train_features=mock_feat,
            train_binary_labels=mock_y,
            positive_indices=pos_idx,
            negative_indices=neg_idx,
            val_features=None,
            val_binary_labels=None,
            test_features=mock_feat,
            test_binary_labels=mock_y,
            strategy="uniform",
            num_steps=1,
        )
        logger.error("Gate A Check 4 FAILED: train_scenario1 did not require val_features")
        return 1
    except ValueError:
        pass

    try:
        estimate_validation_confusion_matrix(
            np.random.randn(10, 8),
            np.zeros(10, dtype=np.int64),
            is_test_data=True,
        )
        logger.error("Gate A Check 4 FAILED: estimate_validation_confusion_matrix accepted test data")
        return 1
    except ValueError:
        pass
    passed += 1
    if verbose:
        print("[Gate A] Check 4/7 (Data segregation) PASSED")

    # Check 5: ResolvedTrainingConfig
    cfg = ResolvedTrainingConfig(scenario="scenario1", target_class=5, imbalance_ratio=100)
    try:
        cfg.target_class = 6  # type: ignore[misc]
        logger.error("Gate A Check 5 FAILED: ResolvedTrainingConfig is mutable")
        return 1
    except (TypeError, AttributeError):
        pass
    passed += 1
    if verbose:
        print("[Gate A] Check 5/7 (Runner configuration equivalence) PASSED")

    # Check 6: Paired statistics rejection
    scores_a = [0.5, 0.6]
    scores_b = [0.55, 0.65]
    keys_a = [(42, "hash1", 0, 100), (43, "hash1", 0, 100)]
    keys_b = [(42, "hash1", 0, 100), (44, "hash1", 0, 100)]  # seed mismatch in second element
    try:
        compute_paired_statistical_tests(scores_a, scores_b, keys_a=keys_a, keys_b=keys_b)
        logger.error("Gate A Check 6 FAILED: compute_paired_statistical_tests accepted mismatched keys")
        return 1
    except ValueError:
        pass
    passed += 1
    if verbose:
        print("[Gate A] Check 6/7 (Paired statistics validation) PASSED")

    # Check 7: Mathematically valid hierarchies
    d_eucl = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    if not is_euclidean_dissimilarity(d_eucl):
        logger.error("Gate A Check 7 FAILED: is_euclidean_dissimilarity failed on metric matrix")
        return 1
    d_non_eucl = np.array([[0.0, 1.0, 10.0], [1.0, 0.0, 1.0], [10.0, 1.0, 0.0]], dtype=np.float64)
    if is_euclidean_dissimilarity(d_non_eucl):
        logger.error("Gate A Check 7 FAILED: is_euclidean_dissimilarity passed on non-metric matrix")
        return 1

    conf = np.eye(4) * 10.0 + 1.0
    pools = {i: np.arange(i * 5, (i + 1) * 5) for i in range(4)}
    tree_w, l_w, _ = build_confusion_tree(conf, pools, return_linkage=True, linkage_method="ward")
    tree_a, l_a, _ = build_confusion_tree(conf, pools, return_linkage=True, linkage_method="average")
    if not (np.all(np.diff(l_w[:, 2]) >= -1e-12) and np.all(np.diff(l_a[:, 2]) >= -1e-12)):
        logger.error("Gate A Check 7 FAILED: Monotonicity violated in linkage merge heights")
        return 1
    passed += 1
    if verbose:
        print("[Gate A] Check 7/7 (Tree hierarchy geometry) PASSED")

    if verbose:
        print(f"\n[Gate A] ALL {passed}/{total} INTEGRITY CHECKS PASSED SUCCESSFULLY.")
    return 0


if __name__ == "__main__":
    sys.exit(verify_gate_a(verbose=True))
