"""Tests for Phase A data split segregation and test leakage prevention (A3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.training.trainer import train_scenario1, train_scenario2
from src.trees.confusion_tree import estimate_validation_confusion_matrix


@pytest.fixture
def dummy_data():
    """Create dummy embeddings and labels for fast testing."""
    features = torch.randn(20, 16)
    labels = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5, dtype=np.int64)
    bin_labels = (labels == 0).astype(np.int64)
    pos_idx = np.array([0, 1], dtype=np.int64)
    neg_idx = np.array([5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int64)
    return {
        "features": features,
        "labels": labels,
        "bin_labels": bin_labels,
        "pos_idx": pos_idx,
        "neg_idx": neg_idx,
    }


def test_periodic_eval_uses_validation_strictly(dummy_data, monkeypatch):
    """Verify that periodic evaluation at eval_every steps calls validation data strictly, never test."""
    f = dummy_data["features"]
    bl = dummy_data["bin_labels"]
    pos_idx = dummy_data["pos_idx"]
    neg_idx = dummy_data["neg_idx"]

    val_f = torch.randn(10, 16)
    val_bl = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)

    test_f = torch.randn(10, 16)
    test_bl = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)

    eval_calls = []

    import src.training.trainer as tr_module
    original_eval = tr_module.evaluate_scenario1_metrics

    def spy_eval(model, features, binary_labels, **kwargs):
        # Record identity of features tensor being evaluated
        eval_calls.append(id(features))
        return original_eval(model, features, binary_labels, **kwargs)

    monkeypatch.setattr(tr_module, "evaluate_scenario1_metrics", spy_eval)

    res = train_scenario1(
        train_features=f,
        train_binary_labels=bl,
        positive_indices=pos_idx,
        negative_indices=neg_idx,
        val_features=val_f,
        val_binary_labels=val_bl,
        test_features=test_f,
        test_binary_labels=test_bl,
        strategy="uniform",
        num_steps=10,
        eval_every=5,
        pos_batch_size=2,
        neg_batch_size=4,
    )

    val_id = id(val_f)
    test_id = id(test_f)

    # In-training evaluations (step 5 and step 10) must be against val_f
    in_training_evals = eval_calls[:-1]
    assert len(in_training_evals) >= 2
    for call_id in in_training_evals:
        assert call_id == val_id, "In-training evaluation must evaluate strictly on val_features."
        assert call_id != test_id, "In-training evaluation must NEVER evaluate on test_features."

    # Final evaluation is permitted to evaluate test_features once at the end
    assert eval_calls[-1] == test_id


def test_missing_validation_raises_value_error(dummy_data):
    """Verify that omitting val_features or val_labels raises ValueError immediately."""
    f = dummy_data["features"]
    bl = dummy_data["bin_labels"]
    pos_idx = dummy_data["pos_idx"]
    neg_idx = dummy_data["neg_idx"]

    # Omitting val_features in train_scenario1
    with pytest.raises(ValueError, match="val_features"):
        train_scenario1(
            train_features=f,
            train_binary_labels=bl,
            positive_indices=pos_idx,
            negative_indices=neg_idx,
            val_features=None,
            val_binary_labels=None,
            test_features=f,
            test_binary_labels=bl,
            strategy="uniform",
            num_steps=10,
        )

    # Omitting val_features in train_scenario2
    with pytest.raises(ValueError, match="val_features"):
        train_scenario2(
            train_features=f,
            train_labels=dummy_data["labels"],
            val_features=None,
            val_labels=None,
            test_features=f,
            test_labels=dummy_data["labels"],
            strategy="uniform",
            num_steps=10,
        )


def test_tree_calibration_rejects_test_data(dummy_data):
    """Verify that tree confusion estimation raises ValueError if test data is passed."""
    f = dummy_data["features"]
    lbl = dummy_data["labels"]

    # Calling with is_test=True or passing explicit test marker
    with pytest.raises(ValueError, match="test data"):
        estimate_validation_confusion_matrix(
            val_features=f,
            val_labels=lbl,
            is_test_data=True,
        )
