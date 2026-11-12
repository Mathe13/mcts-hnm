"""Automated regression and validation tests for Scenario 2 coverage policies and diagnostics.

Tests requirements S2-01, S2-02, S2-03 and Acceptance Criteria C1 through C6.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.mcts.sampler import DualStreamBatchSampler
from src.training.benchmark_runner import run_benchmark
from src.training.trainer import (
    MCTS_STRATEGIES,
    SUPPORTED_STRATEGIES,
    resolve_mcts_hyperparameters,
)
from src.trees.semantic_tree import build_semantic_tree
from src.utils.experiment_config import ExecutionMode, ExperimentConfig


def _create_mock_scenario2_dataset(
    num_classes: int = 100,
    samples_per_class: int = 6,
    feature_dim: int = 16,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor, np.ndarray, torch.Tensor, np.ndarray]:
    """Create reproducible synthetic dataset for fast Scenario 2 unit testing."""
    torch.manual_seed(42)
    np.random.seed(42)

    total_samples = num_classes * samples_per_class
    train_features = torch.randn(total_samples, feature_dim)
    train_labels = np.array([c for c in range(num_classes) for _ in range(samples_per_class)], dtype=np.int64)

    val_samples_per_class = 2
    val_total = num_classes * val_samples_per_class
    val_features = torch.randn(val_total, feature_dim)
    val_labels = np.array([c for c in range(num_classes) for _ in range(val_samples_per_class)], dtype=np.int64)

    test_samples_per_class = 2
    test_total = num_classes * test_samples_per_class
    test_features = torch.randn(test_total, feature_dim)
    test_labels = np.array([c for c in range(num_classes) for _ in range(test_samples_per_class)], dtype=np.int64)

    return train_features, train_labels, val_features, val_labels, test_features, test_labels


def test_mcts_coverage_balanced_strategy_registration() -> None:
    """C1: mcts_coverage_balanced is registered in SUPPORTED_STRATEGIES and partitions batches with 25% focal ratio (S2-02, AC 1, AC 2)."""
    assert "mcts_coverage_balanced" in SUPPORTED_STRATEGIES
    assert "mcts_coverage_balanced" in MCTS_STRATEGIES

    # Hyperparameter resolution: beta=0.75 uniform anchor ratio yields 25% focal stream
    q_norm_bal, beta_bal = resolve_mcts_hyperparameters("mcts_coverage_balanced")
    assert q_norm_bal is True
    assert beta_bal == 0.75

    # Historical enhanced baseline preserved unchanged (beta=0.5 yields 50% focal stream)
    q_norm_enh, beta_enh = resolve_mcts_hyperparameters("mcts_enhanced")
    assert q_norm_enh is True
    assert beta_enh == 0.5


def test_mcts_coverage_balanced_batch_composition() -> None:
    """C2: mcts_coverage_balanced composes minibatches with 16 focal and 48 uniform samples in batch 64 (S2-02, AC 1)."""
    num_classes = 20
    class_to_indices = {
        c: np.arange(c * 50, (c + 1) * 50, dtype=np.int64)
        for c in range(num_classes)
    }
    tree = build_semantic_tree(class_to_indices)

    # beta=0.75 yields K_mcts = floor((1 - 0.75) * 64) = 16 focal, K_uni = ceil(0.75 * 64) = 48
    sampler = DualStreamBatchSampler(
        tree=tree,
        neg_batch_size=64,
        dual_stream_ratio=0.75,
        pos_batch_size=0,
        seed=42,
    )

    leaf_ids = tree.leaf_node_ids()
    target_leaf = leaf_ids[0]
    target_class = int(tree.leaf_to_class_id[target_leaf])

    batch = sampler.sample_batch(target_leaf)

    assert len(batch.all_indices) == 64
    assert len(batch.mcts_indices) == 16
    assert len(batch.uniform_indices) == 48
    assert batch.leaf_class_id == target_class

    # Verify that the 16 focal indices all belong to target_class
    focal_pool = set(class_to_indices[target_class].tolist())
    for idx in batch.mcts_indices:
        assert int(idx) in focal_pool


def test_scenario2_benchmark_runs_all_variants(tmp_path: Path) -> None:
    """C3: run_benchmark() executes uniform_grouped and mcts_coverage_balanced in Scenario 2 (S2-01, AC 1, AC 2)."""
    tr_x, tr_y, val_x, val_y, te_x, te_y = _create_mock_scenario2_dataset(num_classes=100, samples_per_class=5)

    smoke_dir = tmp_path / "results" / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)

    strategies_to_test = ("uniform", "uniform_grouped", "mcts_enhanced", "mcts_coverage_balanced")

    res = run_benchmark(
        scenario="scenario2",
        train_features=tr_x,
        train_labels=tr_y,
        val_features=val_x,
        val_labels=val_y,
        test_features=te_x,
        test_labels=te_y,
        strategies=strategies_to_test,
        tree_types=("semantic",),
        seeds=(42,),
        num_steps=5,
        eval_every=5,
        output_dir=smoke_dir,
        benchmark_name="tier1_scenario2_test",
        config=cfg,
    )

    assert "runs" in res
    evaluated_strategies = {r["strategy"] for r in res["runs"]}
    for strat in strategies_to_test:
        assert strat in evaluated_strategies

    json_file = smoke_dir / "tier1_scenario2_test.json"
    csv_file = smoke_dir / "tier1_scenario2_test.csv"
    assert json_file.exists()
    assert csv_file.exists()

    for r in res["runs"]:
        fm = r["final_metrics"]
        assert "top1_accuracy" in fm
        assert "macro_f1" in fm
        assert "worst_10_class_recall" in fm


def test_scenario2_records_quantitative_diagnostics(tmp_path: Path) -> None:
    """C4: Scenario 2 records H_norm, C_cov, N_unvisited, and exposure-recall correlation (S2-03, AC 1, AC 3)."""
    tr_x, tr_y, val_x, val_y, te_x, te_y = _create_mock_scenario2_dataset(num_classes=100, samples_per_class=5)

    smoke_dir = tmp_path / "results" / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)

    res = run_benchmark(
        scenario="scenario2",
        train_features=tr_x,
        train_labels=tr_y,
        val_features=val_x,
        val_labels=val_y,
        test_features=te_x,
        test_labels=te_y,
        strategies=("uniform", "mcts_coverage_balanced"),
        tree_types=("semantic",),
        seeds=(42,),
        num_steps=5,
        eval_every=5,
        output_dir=smoke_dir,
        benchmark_name="tier1_scenario2_diag_test",
        config=cfg,
    )

    # Check that individual runs record quantitative telemetry in final_metrics
    for r in res["runs"]:
        fm = r["final_metrics"]
        assert "exposure_entropy" in fm
        assert 0.0 <= fm["exposure_entropy"] <= 1.0
        assert "class_coverage" in fm
        assert 0.0 <= fm["class_coverage"] <= 1.0
        assert "unvisited_classes" in fm
        assert fm["unvisited_classes"] >= 0
        assert "exposure_recall_correlation" in fm
        assert -1.0 <= fm["exposure_recall_correlation"] <= 1.0

    # Check dedicated tier1_scenario2_diagnostics.json
    diag_file = smoke_dir / "tier1_scenario2_diagnostics.json"
    assert diag_file.exists()
    with open(diag_file, "r", encoding="utf-8") as f:
        diag_data = json.load(f)

    assert diag_data["benchmark_name"] == "tier1_scenario2_diagnostics"
    assert "diagnostics" in diag_data
    for s in ("uniform", "mcts_coverage_balanced"):
        assert s in diag_data["diagnostics"]
        s_diag = diag_data["diagnostics"][s]
        assert "class_exposure_counts" in s_diag
        assert "exposure_entropy" in s_diag
        assert "class_coverage_pct" in s_diag
        assert "unvisited_classes_count" in s_diag
        assert "per_class_recall" in s_diag
        assert "exposure_recall_correlation" in s_diag

    # Check CSV export columns
    csv_file = smoke_dir / "tier1_scenario2_diag_test.csv"
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        for col in ("exposure_entropy", "class_coverage", "unvisited_classes", "exposure_recall_correlation"):
            assert col in header


def test_scenario2_paired_differences_against_uniform(tmp_path: Path) -> None:
    """C5: Pairwise statistical differences against uniform baseline are computed and persisted (S2-03, AC 2, AC 4)."""
    tr_x, tr_y, val_x, val_y, te_x, te_y = _create_mock_scenario2_dataset(num_classes=100, samples_per_class=6)

    smoke_dir = tmp_path / "results" / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)

    res = run_benchmark(
        scenario="scenario2",
        train_features=tr_x,
        train_labels=tr_y,
        val_features=val_x,
        val_labels=val_y,
        test_features=te_x,
        test_labels=te_y,
        strategies=("uniform", "mcts_enhanced", "mcts_coverage_balanced"),
        tree_types=("semantic",),
        seeds=(42, 43),
        num_steps=5,
        eval_every=5,
        output_dir=smoke_dir,
        benchmark_name="tier1_scenario2_paired_test",
        config=cfg,
    )

    assert "paired_differences_against_uniform" in res
    paired_diff = res["paired_differences_against_uniform"]
    assert "mcts_coverage_balanced" in paired_diff
    assert "mcts_enhanced" in paired_diff

    bal_diff = paired_diff["mcts_coverage_balanced"]
    assert bal_diff["common_seeds"] == [42, 43]
    assert "delta_primary_mean" in bal_diff
    assert "delta_f1_mean" in bal_diff
    assert "delta_worst_10_mean" in bal_diff
    assert "primary_stats" in bal_diff

    stats = bal_diff["primary_stats"]
    assert "mean_diff" in stats
    assert "t_pvalue" in stats
    assert "cohens_d" in stats


def test_scenario2_rejects_unsupported_strategy(tmp_path: Path) -> None:
    """C6: Unsupported strategy in Scenario 2 raises ValueError with clear message (S2-01, AC 4)."""
    tr_x, tr_y, val_x, val_y, te_x, te_y = _create_mock_scenario2_dataset(num_classes=100, samples_per_class=5)

    smoke_dir = tmp_path / "results" / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)

    with pytest.raises(ValueError, match="Unknown strategy 'invalid_strategy_xyz' for scenario2"):
        run_benchmark(
            scenario="scenario2",
            train_features=tr_x,
            train_labels=tr_y,
            val_features=val_x,
            val_labels=val_y,
            test_features=te_x,
            test_labels=te_y,
            strategies=("invalid_strategy_xyz",),
            tree_types=("semantic",),
            seeds=(42,),
            num_steps=5,
            eval_every=5,
            output_dir=smoke_dir,
            config=cfg,
        )
