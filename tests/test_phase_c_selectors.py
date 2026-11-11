"""Tests for Phase C simple baseline selectors: GreedyLossEMA and FlatUCB (C3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.training.selectors import FlatUCBSelector, GreedyLossEMASelector
from src.training.trainer import train_scenario2


def test_greedy_loss_ema_selector_behavior() -> None:
    """Verify GreedyLossEMASelector selects class with highest EMA loss and updates smoothly."""
    c2i = {
        0: np.arange(0, 10, dtype=np.int64),
        1: np.arange(10, 20, dtype=np.int64),
        2: np.arange(20, 30, dtype=np.int64),
    }

    # Parameter validation
    with pytest.raises(ValueError, match="alpha"):
        GreedyLossEMASelector(class_to_indices=c2i, num_classes=3, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        GreedyLossEMASelector(class_to_indices=c2i, num_classes=3, alpha=1.5)
    with pytest.raises(ValueError, match="batch_size"):
        GreedyLossEMASelector(class_to_indices=c2i, num_classes=3, batch_size=0)

    selector = GreedyLossEMASelector(
        class_to_indices=c2i,
        num_classes=3,
        batch_size=8,
        dual_stream_ratio=0.5,
        alpha=0.5,
        seed=42,
    )

    # Initial update: set class 1 to high loss (2.0) and class 0 to low loss (0.5)
    selector.update(0, 0.5)
    selector.update(1, 2.0)
    selector.update(2, 0.1)

    # Greedy choice must pick class 1
    selected = selector.select_class()
    assert selected == 1

    batch = selector.sample_batch()
    assert batch.leaf_class_id == 1
    assert batch.all_indices.shape == (8,)
    assert batch.mcts_indices.shape == (4,)
    assert batch.uniform_indices.shape == (4,)

    # Train scenario 2 with greedy_loss_ema
    features = torch.randn(30, 8)
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int64)
    res = train_scenario2(
        train_features=features,
        train_labels=labels,
        val_features=features,
        val_labels=labels,
        num_classes=3,
        strategy="greedy_loss_ema",
        batch_size=8,
        num_steps=6,
        eval_every=3,
        seed=42,
    )
    assert res["strategy"] == "greedy_loss_ema"
    assert "val_metrics" in res
    assert np.isfinite(res["val_metrics"]["loss"])


def test_flat_ucb_unvisited_priority_and_selection() -> None:
    """Verify FlatUCBSelector samples all unvisited classes first, then follows UCB1."""
    c2i = {
        0: np.arange(0, 10, dtype=np.int64),
        1: np.arange(10, 20, dtype=np.int64),
        2: np.arange(20, 30, dtype=np.int64),
    }

    # Parameter validation
    with pytest.raises(ValueError, match="exploration_c"):
        FlatUCBSelector(class_to_indices=c2i, num_classes=3, exploration_c=-1.0)

    selector = FlatUCBSelector(
        class_to_indices=c2i,
        num_classes=3,
        batch_size=8,
        dual_stream_ratio=0.5,
        exploration_c=2.0,
        seed=42,
    )

    # Step 1: all unvisited -> picks class 0
    c0 = selector.select_class()
    assert c0 == 0
    selector.update(0, 0.5)

    # Step 2: class 1 and 2 unvisited -> picks class 1
    c1 = selector.select_class()
    assert c1 == 1
    selector.update(1, 1.5)

    # Step 3: class 2 unvisited -> picks class 2
    c2 = selector.select_class()
    assert c2 == 2
    selector.update(2, 0.2)

    # Now all visited: N = 3.
    # Q0 = 0.5, U0 = 2*sqrt(ln(3)/1) ~ 2.096 -> 2.596
    # Q1 = 1.5, U1 = 2*sqrt(ln(3)/1) ~ 2.096 -> 3.596
    # Q2 = 0.2, U2 = 2*sqrt(ln(3)/1) ~ 2.096 -> 2.296
    # Highest score is class 1
    c_next = selector.select_class()
    assert c_next == 1

    # Train scenario 2 with flat_ucb
    features = torch.randn(30, 8)
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int64)
    res = train_scenario2(
        train_features=features,
        train_labels=labels,
        val_features=features,
        val_labels=labels,
        num_classes=3,
        strategy="flat_ucb",
        batch_size=8,
        num_steps=6,
        eval_every=3,
        seed=42,
    )
    assert res["strategy"] == "flat_ucb"
    assert "val_metrics" in res
    assert np.isfinite(res["val_metrics"]["loss"])
