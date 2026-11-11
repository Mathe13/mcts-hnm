"""Tests for Phase A runner governance and configuration equivalence (A4)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import numpy as np
import pytest
import torch

from src.training.benchmark_runner import run_benchmark, run_sensitivity_ablation
from src.training.runner_config import ResolvedTrainingConfig
from src.training.trainer import train_scenario1
from src.trees.semantic_tree import build_semantic_tree
from src.utils.experiment_config import ExecutionMode, ExperimentConfig


def test_resolved_training_config_immutability():
    """Verify that ResolvedTrainingConfig is immutable, serializable, and hashable."""
    cfg = ResolvedTrainingConfig(
        scenario="scenario1",
        target_class=0,
        imbalance_ratio="1:50",
        target_positive_count=80,
        pos_batch_size=16,
        neg_batch_size=48,
        num_steps=100,
        eval_every=20,
        lr=1e-3,
        exploration_c=2.0,
        momentum_alpha=0.3,
        dual_stream_ratio=0.5,
        seed=42,
        strategy="mcts_enhanced",
        tree_type="semantic",
        device="cpu",
    )

    # Immutability
    with pytest.raises(FrozenInstanceError):
        cfg.lr = 2e-3  # type: ignore

    d = cfg.to_dict()
    assert d["scenario"] == "scenario1"
    assert d["exploration_c"] == 2.0
    assert d["momentum_alpha"] == 0.3

    cfg_hash = cfg.compute_config_hash()
    assert isinstance(cfg_hash, str) and len(cfg_hash) == 64


def test_runners_deterministic_equivalence(tmp_path):
    """Verify that train_scenario1 and run_benchmark produce identical results on identical configs."""
    train_feats = torch.randn(100, 32)
    train_labels = np.array([i % 10 for i in range(100)], dtype=np.int64)
    val_feats = torch.randn(20, 32)
    val_labels = np.array([i % 10 for i in range(20)], dtype=np.int64)
    test_feats = torch.randn(20, 32)
    test_labels = np.array([i % 10 for i in range(20)], dtype=np.int64)

    exp_cfg = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)

    # Run via benchmark runner with 1 seed and single strategy
    bench_res = run_benchmark(
        scenario="scenario1",
        train_features=train_feats,
        train_labels=train_labels,
        val_features=val_feats,
        val_labels=val_labels,
        test_features=test_feats,
        test_labels=test_labels,
        strategies=("uniform",),
        tree_types=("semantic",),
        seeds=(42,),
        output_dir=tmp_path / "results" / "smoke",
        benchmark_name="equiv_bench",
        num_steps=10,
        eval_every=5,
        config=exp_cfg,
    )

    # Run direct train_scenario1 with identical split and seed
    pos_mask = train_labels == 0
    pos_idx = np.flatnonzero(pos_mask).astype(np.int64)
    neg_idx = np.flatnonzero(~pos_mask).astype(np.int64)
    bin_train = pos_mask.astype(np.int64)
    bin_val = (val_labels == 0).astype(np.int64)
    bin_test = (test_labels == 0).astype(np.int64)

    direct_res = train_scenario1(
        train_features=train_feats,
        train_binary_labels=bin_train,
        positive_indices=pos_idx[:1],  # pos_fraction=0.02 keeps 1 sample out of 10
        negative_indices=neg_idx,
        val_features=val_feats,
        val_binary_labels=bin_val,
        test_features=test_feats,
        test_binary_labels=bin_test,
        strategy="uniform",
        num_steps=10,
        eval_every=5,
        seed=42,
    )

    # Metric history lengths must match
    assert len(bench_res["per_seed_results"]) == 1
    bench_final = bench_res["per_seed_results"][0]["final_metrics"]
    direct_final = direct_res["final_metrics"]

    assert pytest.approx(bench_final["loss"], abs=1e-4) == direct_final["loss"]
    assert pytest.approx(bench_final["auprc"], abs=1e-4) == direct_final["auprc"]
