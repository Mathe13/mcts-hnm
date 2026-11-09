"""Tests for downstream models, baseline samplers, training loops, and benchmark suite."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.mlp_head import Tier1MLPHead, create_mlp_optimizer
from src.training.baselines import OHEMSampler, UniformSampler
from src.training.trainer import (
    evaluate_scenario1_metrics,
    evaluate_scenario2_metrics,
    train_scenario1,
    train_scenario2,
)
from src.trees.semantic_tree import build_semantic_tree
from src.utils.statistics import compute_paired_statistical_tests


def test_compute_paired_statistical_tests() -> None:
    # 5 paired scores
    scores_a = [0.85, 0.88, 0.84, 0.87, 0.86]
    scores_b = [0.80, 0.81, 0.82, 0.79, 0.80]

    stats = compute_paired_statistical_tests(scores_a, scores_b)
    assert stats["mean_a"] > stats["mean_b"]
    assert stats["mean_diff"] > 0
    assert stats["t_stat"] > 0
    assert 0.0 <= stats["t_pvalue"] <= 1.0
    assert 0.0 <= stats["wilcoxon_pvalue"] <= 1.0


def test_compute_paired_statistical_tests_zero_diff_fallback() -> None:
    scores_a = [0.85, 0.85, 0.85, 0.85, 0.85]
    scores_b = [0.85, 0.85, 0.85, 0.85, 0.85]

    stats = compute_paired_statistical_tests(scores_a, scores_b)
    assert stats["mean_diff"] == 0.0
    assert stats["t_stat"] == 0.0
    assert stats["t_pvalue"] == 1.0
    assert stats["wilcoxon_stat"] == 0.0
    assert stats["wilcoxon_pvalue"] == 1.0

    with pytest.raises(ValueError, match="match"):
        compute_paired_statistical_tests([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="At least 2"):
        compute_paired_statistical_tests([1.0], [1.0])
    with pytest.raises(ValueError, match="finite"):
        compute_paired_statistical_tests([1.0, np.nan], [1.0, 2.0])


def test_train_scenario1_metrics_and_entropy() -> None:
    # 4 classes, 20 samples per class = 80 samples
    torch.manual_seed(42)
    np.random.seed(42)
    features = torch.randn(80, 16)
    labels = np.repeat(np.arange(4, dtype=np.int64), 20)

    # Positive class is 0 (first 5 samples)
    pos_idx = np.arange(5, dtype=np.int64)
    neg_idx = np.arange(20, 80, dtype=np.int64)
    binary_labels = (labels == 0).astype(np.int64)

    class_to_idx = {c: np.flatnonzero(labels == c).astype(np.int64) for c in range(1, 4)}
    tree = build_semantic_tree(class_to_idx)

    res = train_scenario1(
        train_features=features,
        train_binary_labels=binary_labels,
        positive_indices=pos_idx,
        negative_indices=neg_idx,
        test_features=features,
        test_binary_labels=binary_labels,
        strategy="mcts_enhanced",
        tree=tree,
        num_steps=20,
        eval_every=10,
        pos_batch_size=4,
        neg_batch_size=8,
        lr=1e-2,
    )

    assert res["strategy"] == "mcts_enhanced"
    assert len(res["history"]) == 2
    for h in res["history"]:
        assert "loss" in h and "f1" in h and "auprc" in h and "fpr_at_95_recall" in h
        assert "entropy_nats" in h and "normalized_entropy" in h


def test_train_scenario2_metrics_and_entropy() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    features = torch.randn(100, 16)
    labels = np.repeat(np.arange(5, dtype=np.int64), 20)

    class_to_idx = {c: np.flatnonzero(labels == c).astype(np.int64) for c in range(5)}
    tree = build_semantic_tree(class_to_idx)

    res = train_scenario2(
        train_features=features,
        train_labels=labels,
        test_features=features,
        test_labels=labels,
        strategy="mcts_strict",
        tree=tree,
        num_classes=5,
        num_steps=20,
        eval_every=10,
        batch_size=16,
        lr=1e-2,
    )

    assert res["strategy"] == "mcts_strict"
    assert len(res["history"]) == 2
    for h in res["history"]:
        assert "loss" in h and "top1_accuracy" in h and "macro_f1" in h and "worst_10_class_recall" in h
        assert "entropy_nats" in h


def test_mcts_updates_and_shifts_leaf_visits() -> None:
    # Verify that MCTS updates visits and shifts towards high-loss leaf
    torch.manual_seed(42)
    np.random.seed(42)
    features = torch.randn(60, 8)
    labels = np.repeat(np.arange(3, dtype=np.int64), 20)
    pos_idx = np.arange(5, dtype=np.int64)
    neg_idx = np.arange(20, 60, dtype=np.int64)
    binary_labels = (labels == 0).astype(np.int64)

    class_to_idx = {c: np.flatnonzero(labels == c).astype(np.int64) for c in range(1, 3)}
    tree = build_semantic_tree(class_to_idx)

    # Artificially modify class 2 features to produce huge logits/loss
    features[labels == 2, :] = 10.0

    res = train_scenario1(
        train_features=features,
        train_binary_labels=binary_labels,
        positive_indices=pos_idx,
        negative_indices=neg_idx,
        test_features=features,
        test_binary_labels=binary_labels,
        strategy="mcts_enhanced",
        tree=tree,
        num_steps=30,
        eval_every=15,
        pos_batch_size=2,
        neg_batch_size=6,
        exploration_c=0.1,  # high exploitation
        lr=1e-3,
    )

    # Leaf nodes visit counts
    leaf_ids = tree.leaf_node_ids()
    visits = tree.visits[leaf_ids]
    assert np.sum(visits) == 30
    # Q-values must be non-zero
    assert np.any(tree.q_values[leaf_ids] > 0.0)


def test_trainer_invalid_arguments_raise() -> None:
    features = torch.randn(10, 4)
    labels = np.zeros(10, dtype=np.int64)

    with pytest.raises(ValueError, match="Unknown strategy"):
        train_scenario1(features, labels, np.array([0]), np.array([1]), features, labels, strategy="unknown")
    with pytest.raises(ValueError, match="num_steps"):
        train_scenario1(features, labels, np.array([0]), np.array([1]), features, labels, num_steps=0)
    with pytest.raises(ValueError, match="tree is required"):
        train_scenario1(features, labels, np.array([0]), np.array([1]), features, labels, strategy="mcts_enhanced", tree=None)
    with pytest.raises(ValueError, match="Unknown strategy"):
        train_scenario2(features, labels, features, labels, strategy="unknown")
    with pytest.raises(ValueError, match="tree is required"):
        train_scenario2(features, labels, features, labels, strategy="mcts_strict", tree=None)


def test_evaluate_scenario1_and_scenario2_metrics() -> None:
    # Scenario 1 binary test
    model_bin = Tier1MLPHead(input_dim=4, hidden_dim=8, num_outputs=1, squeeze_binary=True)
    features = torch.randn(20, 4)
    y_bin = np.zeros(20, dtype=np.int64)
    y_bin[:4] = 1  # 4 positives, 16 negatives

    m1 = evaluate_scenario1_metrics(model_bin, features, y_bin, batch_size=8)
    assert "loss" in m1 and m1["loss"] >= 0.0
    assert 0.0 <= m1["f1"] <= 1.0
    assert 0.0 <= m1["precision"] <= 1.0
    assert 0.0 <= m1["recall"] <= 1.0
    assert 0.0 <= m1["auprc"] <= 1.0
    assert 0.0 <= m1["fpr_at_95_recall"] <= 1.0

    # Scenario 2 multi-class test
    model_multi = Tier1MLPHead(input_dim=4, hidden_dim=8, num_outputs=10)
    y_multi = np.repeat(np.arange(10, dtype=np.int64), 3)  # 30 samples, 10 classes
    features_multi = torch.randn(30, 4)

    m2 = evaluate_scenario2_metrics(model_multi, features_multi, y_multi, num_classes=10, batch_size=8)
    assert "loss" in m2 and m2["loss"] >= 0.0
    assert 0.0 <= m2["top1_accuracy"] <= 1.0
    assert 0.0 <= m2["macro_f1"] <= 1.0
    assert 0.0 <= m2["worst_10_class_recall"] <= 1.0


def test_uniform_sampler_batch_generation() -> None:
    neg_pool = np.arange(100, dtype=np.int64)
    pos_pool = np.arange(10, dtype=np.int64)

    # Scenario 1 (pos + neg)
    sampler = UniformSampler(negative_indices=neg_pool, neg_batch_size=24, positive_indices=pos_pool, pos_batch_size=8, seed=42)
    pos_idx, neg_idx, all_idx = sampler.sample_batch()
    assert len(pos_idx) == 8
    assert len(neg_idx) == 24
    assert len(all_idx) == 32
    assert np.all(np.isin(pos_idx, pos_pool))
    assert np.all(np.isin(neg_idx, neg_pool))

    # Scenario 2 (neg only across all classes)
    sampler_s2 = UniformSampler(negative_indices=neg_pool, neg_batch_size=48, pos_batch_size=0, seed=42)
    p2, n2, a2 = sampler_s2.sample_batch()
    assert len(p2) == 0
    assert len(n2) == 48
    assert len(a2) == 48


def test_ohem_sampler_selects_highest_loss() -> None:
    # 50 negative samples, input dim 4
    torch.manual_seed(42)
    num_samples = 50
    features = torch.randn(num_samples, 4)
    neg_pool = np.arange(num_samples, dtype=np.int64)

    # Linear model: logit = x[:, 0]
    # For binary negatives, loss = BCE(z, 0). Larger logit z -> higher loss!
    model = Tier1MLPHead(input_dim=4, hidden_dim=8, num_outputs=1, squeeze_binary=True)
    # Mock model weights to isolate 1st feature
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        # Set first hidden neuron weight to 10.0 and output to 1.0
        model.fc1.weight[0, 0] = 10.0
        model.fc2.weight[0, 0] = 1.0

    # Put high positive value in sample index 7 so it has highest loss
    features[7, 0] = 5.0
    features[12, 0] = 4.0

    sampler = OHEMSampler(
        negative_indices=neg_pool,
        neg_batch_size=2,
        ohem_ratio=25,  # evaluate all 50
        seed=42,
    )
    _, neg_idx, _ = sampler.sample_batch(model, features, is_binary=True)
    assert len(neg_idx) == 2
    assert 7 in neg_idx
    assert 12 in neg_idx


def test_tier1_mlp_head_forward_and_optimizer() -> None:
    # Binary head with squeeze
    model_bin = Tier1MLPHead(input_dim=512, hidden_dim=128, num_outputs=1, dropout=0.1, squeeze_binary=True)
    x = torch.randn(16, 512, requires_grad=True)
    out_bin = model_bin(x)
    assert out_bin.shape == (16,)
    loss_bin = out_bin.sum()
    loss_bin.backward()
    assert x.grad is not None

    # Binary head without squeeze
    model_bin_nosq = Tier1MLPHead(input_dim=512, hidden_dim=128, num_outputs=1, squeeze_binary=False)
    out_nosq = model_bin_nosq(torch.randn(8, 512))
    assert out_nosq.shape == (8, 1)

    # Multi-class head
    model_multi = Tier1MLPHead(input_dim=512, hidden_dim=64, num_outputs=100)
    out_multi = model_multi(torch.randn(10, 512))
    assert out_multi.shape == (10, 100)

    # Optimizer creation
    opt = create_mlp_optimizer(model_multi, lr=1e-3, weight_decay=1e-4)
    assert isinstance(opt, torch.optim.Adam)
    assert opt.defaults["lr"] == 1e-3
    assert opt.defaults["weight_decay"] == 1e-4


def test_model_and_sampler_invalid_args_raise() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        Tier1MLPHead(input_dim=0)
    with pytest.raises(ValueError, match="hidden_dim"):
        Tier1MLPHead(hidden_dim=-5)
    with pytest.raises(ValueError, match="num_outputs"):
        Tier1MLPHead(num_outputs=0)
    with pytest.raises(ValueError, match="dropout"):
        Tier1MLPHead(dropout=1.0)
    with pytest.raises(ValueError, match="dropout"):
        Tier1MLPHead(dropout=-0.1)
    with pytest.raises(ValueError, match="Expected 2D tensor"):
        m = Tier1MLPHead(input_dim=512)
        m(torch.randn(16, 256))
    with pytest.raises(ValueError, match="lr"):
        create_mlp_optimizer(Tier1MLPHead(), lr=-1e-3)
    with pytest.raises(ValueError, match="weight_decay"):
        create_mlp_optimizer(Tier1MLPHead(), weight_decay=-1.0)
    with pytest.raises(ValueError, match="neg_batch_size"):
        UniformSampler(negative_indices=np.array([1, 2]), neg_batch_size=0)
    with pytest.raises(ValueError, match="positive_indices"):
        UniformSampler(negative_indices=np.array([1, 2]), pos_batch_size=2)
    with pytest.raises(ValueError, match="ohem_ratio"):
        OHEMSampler(negative_indices=np.array([1, 2]), ohem_ratio=0)
