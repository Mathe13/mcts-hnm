"""Automated regression and validation tests for Feature 24 (P2 - Study Expansion: Multi-Ratio & Multi-Target Classes).

Tests requirements EXP-01, EXP-02, EXP-03 and Acceptance Criteria C1 through C6.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.training.benchmark_runner import run_benchmark
from src.utils.experiment_config import ExecutionMode, ExperimentConfig
from src.utils.paper_artifacts import load_real_benchmark_results
from src.utils.statistics import compute_method_by_ratio_interaction


def _create_mock_scenario1_dataset(
    num_classes: int = 20,
    samples_per_class: int = 15,
    embedding_dim: int = 64,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor, np.ndarray, torch.Tensor, np.ndarray]:
    """Deterministically create mock features and labels for multi-class and multi-ratio testing."""
    rng = np.random.RandomState(42)

    tr_x_list: list[np.ndarray] = []
    tr_y_list: list[int] = []
    val_x_list: list[np.ndarray] = []
    val_y_list: list[int] = []
    te_x_list: list[np.ndarray] = []
    te_y_list: list[int] = []

    for c in range(num_classes):
        mean = rng.randn(embedding_dim) * 0.5
        c_tr = mean + rng.randn(samples_per_class, embedding_dim) * 0.1
        c_val = mean + rng.randn(2, embedding_dim) * 0.1
        c_te = mean + rng.randn(4, embedding_dim) * 0.1

        tr_x_list.append(c_tr)
        tr_y_list.extend([c] * samples_per_class)
        val_x_list.append(c_val)
        val_y_list.extend([c] * 2)
        te_x_list.append(c_te)
        te_y_list.extend([c] * 4)

    tr_x = torch.from_numpy(np.concatenate(tr_x_list, axis=0)).float()
    tr_y = np.array(tr_y_list, dtype=np.int64)
    val_x = torch.from_numpy(np.concatenate(val_x_list, axis=0)).float()
    val_y = np.array(val_y_list, dtype=np.int64)
    te_x = torch.from_numpy(np.concatenate(te_x_list, axis=0)).float()
    te_y = np.array(te_y_list, dtype=np.int64)

    return tr_x, tr_y, val_x, val_y, te_x, te_y


def test_multi_ratio_execution_and_positive_fractions(tmp_path: Path) -> None:
    """C1: run_benchmark() evaluates multiple imbalance ratios under paired seeds (EXP-01, AC 1, AC 2)."""
    tr_x, tr_y, val_x, val_y, te_x, te_y = _create_mock_scenario1_dataset()
    smoke_dir = tmp_path / "results" / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)

    res = run_benchmark(
        scenario="scenario1",
        train_features=tr_x,
        train_labels=tr_y,
        val_features=val_x,
        val_labels=val_y,
        test_features=te_x,
        test_labels=te_y,
        strategies=("uniform",),
        tree_types=("semantic",),
        seeds=(42,),
        imbalance_ratios=("1:10", "1:50"),
        num_steps=5,
        eval_every=5,
        output_dir=smoke_dir,
        benchmark_name="tier1_multi_ratio_test",
        config=cfg,
    )

    assert len(res["runs"]) == 2
    ratios_seen = {r["imbalance_ratio"] for r in res["runs"]}
    assert ratios_seen == {"1:10", "1:50"}


def test_method_by_ratio_interaction_calculation() -> None:
    """C2: compute_method_by_ratio_interaction() calculates interaction effects across ratios (EXP-01, AC 3)."""
    # At ratio 1: A and B are equal
    s_r1_a = [0.80, 0.82, 0.84, 0.81]
    s_r1_b = [0.80, 0.82, 0.84, 0.81]
    # At ratio 2: A is significantly higher than B by 0.10
    s_r2_a = [0.75, 0.76, 0.74, 0.75]
    s_r2_b = [0.65, 0.66, 0.64, 0.65]

    res = compute_method_by_ratio_interaction(s_r1_a, s_r1_b, s_r2_a, s_r2_b)
    assert res["n"] == 4
    assert np.isclose(res["mean_diff_r1"], 0.0, atol=1e-5)
    assert np.isclose(res["mean_diff_r2"], 0.10, atol=1e-5)
    assert np.isclose(res["mean_interaction"], 0.10, atol=1e-5)
    assert res["ci_95_lower"] > 0.05
    assert res["t_pvalue"] < 0.01


def test_multi_ratio_persistence_and_interactions_payload(tmp_path: Path) -> None:
    """C3: Individual CSV rows and JSON payload persist imbalance_ratio and ratio_interactions (EXP-01, AC 4)."""
    tr_x, tr_y, val_x, val_y, te_x, te_y = _create_mock_scenario1_dataset()
    smoke_dir = tmp_path / "results" / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)

    res = run_benchmark(
        scenario="scenario1",
        train_features=tr_x,
        train_labels=tr_y,
        val_features=val_x,
        val_labels=val_y,
        test_features=te_x,
        test_labels=te_y,
        strategies=("uniform", "mcts_enhanced"),
        tree_types=("semantic",),
        seeds=(42, 43),
        imbalance_ratios=("1:10", "1:50"),
        num_steps=5,
        eval_every=5,
        output_dir=smoke_dir,
        benchmark_name="tier1_ratio_interactions_test",
        config=cfg,
    )

    assert "ratio_interactions" in res
    assert "1:10_vs_1:50" in res["ratio_interactions"]

    # Verify CSV rows
    csv_file = smoke_dir / "tier1_ratio_interactions_test.csv"
    assert csv_file.is_file()
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 8  # 2 ratios x 2 strategies x 2 seeds
    assert "imbalance_ratio" in reader.fieldnames
    assert "target_class" in reader.fieldnames


def test_multi_target_class_execution(tmp_path: Path) -> None:
    """C4: run_benchmark() evaluates multiple target classes under identical seed schedules (EXP-02, AC 1, AC 2)."""
    tr_x, tr_y, val_x, val_y, te_x, te_y = _create_mock_scenario1_dataset()
    smoke_dir = tmp_path / "results" / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)

    res = run_benchmark(
        scenario="scenario1",
        train_features=tr_x,
        train_labels=tr_y,
        val_features=val_x,
        val_labels=val_y,
        test_features=te_x,
        test_labels=te_y,
        strategies=("uniform",),
        tree_types=("semantic",),
        seeds=(42,),
        target_classes=(0, 1),
        num_steps=5,
        eval_every=5,
        output_dir=smoke_dir,
        benchmark_name="tier1_multi_class_test",
        config=cfg,
    )

    assert len(res["runs"]) == 2
    classes_seen = {r["target_class"] for r in res["runs"]}
    assert classes_seen == {0, 1}


def test_target_class_persistence_and_aggregation(tmp_path: Path) -> None:
    """C5: Output JSON and CSV record target_class and aggregate metrics across classes (EXP-02, AC 3, AC 4)."""
    tr_x, tr_y, val_x, val_y, te_x, te_y = _create_mock_scenario1_dataset()
    smoke_dir = tmp_path / "results" / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)

    res = run_benchmark(
        scenario="scenario1",
        train_features=tr_x,
        train_labels=tr_y,
        val_features=val_x,
        val_labels=val_y,
        test_features=te_x,
        test_labels=te_y,
        strategies=("uniform", "mcts_enhanced"),
        tree_types=("semantic",),
        seeds=(42,),
        target_classes=(0, 1),
        num_steps=5,
        eval_every=5,
        output_dir=smoke_dir,
        benchmark_name="tier1_class_agg_test",
        config=cfg,
    )

    assert res["target_classes"] == [0, 1]
    assert "aggregated_by_class" in res
    assert 0 in res["aggregated_by_class"]
    assert 1 in res["aggregated_by_class"]
    assert "uniform" in res["aggregated_by_class"][0]

    # Check JSON on disk
    json_path = smoke_dir / "tier1_class_agg_test.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["target_classes"] == [0, 1]
    assert "0" in data["aggregated_by_class"] or 0 in data["aggregated_by_class"]


def test_notebook_03_multi_ratio_and_multi_class_integration() -> None:
    """C6: Notebook 03 Section 2 displays multi-ratio and multi-class configuration (EXP-03, AC 1, AC 3)."""
    root = Path(__file__).resolve().parent.parent
    nb_path = root / "notebooks" / "03_tier1_mcts_benchmarks.ipynb"
    assert nb_path.is_file()

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    full_text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])

    assert "IMBALANCE_RATIOS_AXIS1" in full_text
    assert "TARGET_CLASSES_AXIS1" in full_text
    assert "ratio_interactions" in full_text
