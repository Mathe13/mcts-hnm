"""Automated regression and validation tests for Feature 23 (P1 - Mechanism Ablations and Paired Hierarchy Tests).

Tests requirements ABL-01, HIER-01 and Acceptance Criteria C1 through C6.
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
from src.trees.base_tree import FastHierarchyTree, compute_tree_topology_diagnostics
from src.trees.semantic_tree import build_semantic_tree
from src.utils.experiment_config import ExecutionMode, ExperimentConfig
from src.utils.statistics import compute_2x2_factorial_effects


def _create_mock_scenario1_dataset(
    num_classes: int = 20,
    samples_per_class: int = 15,
    feature_dim: int = 16,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor, np.ndarray, torch.Tensor, np.ndarray]:
    """Create reproducible synthetic dataset for fast Scenario 1 unit testing."""
    torch.manual_seed(42)
    np.random.seed(42)

    total_samples = num_classes * samples_per_class
    train_features = torch.randn(total_samples, feature_dim)
    train_labels = np.array([c for c in range(num_classes) for _ in range(samples_per_class)], dtype=np.int64)

    val_samples = num_classes * 4
    val_features = torch.randn(val_samples, feature_dim)
    val_labels = np.array([c for c in range(num_classes) for _ in range(4)], dtype=np.int64)

    test_samples = num_classes * 4
    test_features = torch.randn(test_samples, feature_dim)
    test_labels = np.array([c for c in range(num_classes) for _ in range(4)], dtype=np.int64)

    return train_features, train_labels, val_features, val_labels, test_features, test_labels


def test_tree_topology_diagnostics() -> None:
    """C1: compute_tree_topology_diagnostics() computes max depth, mean leaf depth, node counts, and branching factors (HIER-01, AC 3)."""
    num_classes = 20
    class_to_indices = {
        c: np.arange(c * 20, (c + 1) * 20, dtype=np.int64)
        for c in range(num_classes)
    }
    tree = build_semantic_tree(class_to_indices)

    diag = compute_tree_topology_diagnostics(tree)

    assert diag["tree_type"] == "semantic"
    assert diag["total_nodes"] == tree.num_nodes
    assert diag["num_leaves"] == num_classes
    assert diag["num_internal_nodes"] > 0
    assert diag["max_depth"] >= 2
    assert diag["mean_leaf_depth"] >= 1.0
    assert diag["min_leaf_depth"] >= 1
    assert diag["branching_factor_mean"] > 0.0
    assert diag["branching_factor_max"] >= diag["branching_factor_min"]


def test_factorial_2x2_effects_calculation() -> None:
    """C2: compute_2x2_factorial_effects() calculates main effects of Q-norm and Dual-Stream and interaction term (ABL-01, AC 3)."""
    scores = {
        "mcts_strict": [0.50, 0.52, 0.51, 0.49, 0.53],
        "mcts_qnorm_only": [0.55, 0.58, 0.56, 0.54, 0.57],
        "mcts_dualstream_only": [0.60, 0.61, 0.62, 0.59, 0.63],
        "mcts_enhanced": [0.70, 0.72, 0.71, 0.69, 0.73],
    }

    res = compute_2x2_factorial_effects(scores)

    assert res["n"] == 5
    assert "main_effect_qnorm" in res
    assert "main_effect_dualstream" in res
    assert "interaction_effect" in res
    assert "variant_means" in res

    me_q = res["main_effect_qnorm"]
    me_ds = res["main_effect_dualstream"]
    ie = res["interaction_effect"]

    # Q-norm main effect should be ~ +0.075
    assert me_q["mean"] > 0.05
    assert me_q["cohens_d"] > 0.0
    assert me_q["t_pvalue"] < 0.05

    # Dual-stream main effect should be ~ +0.125
    assert me_ds["mean"] > 0.10
    assert me_ds["cohens_d"] > 0.0
    assert me_ds["t_pvalue"] < 0.05

    # Interaction effect: (0.71 - 0.56) - (0.61 - 0.51) = 0.15 - 0.10 = +0.05
    assert ie["mean"] > 0.0
    assert len(ie["per_seed"]) == 5


def test_benchmark_records_computational_costs(tmp_path: Path) -> None:
    """C3: run_benchmark() records computational cost metrics in final_metrics and CSV output (ABL-01, AC 2)."""
    tr_x, tr_y, val_x, val_y, te_x, te_y = _create_mock_scenario1_dataset(num_classes=20, samples_per_class=10)

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
        num_steps=5,
        eval_every=5,
        output_dir=smoke_dir,
        benchmark_name="tier1_cost_test",
        config=cfg,
    )

    cost_keys = ("total_updates", "gradient_samples", "mining_forwards", "elapsed_seconds")
    for r in res["runs"]:
        fm = r["final_metrics"]
        for ck in cost_keys:
            assert ck in fm
            assert fm[ck] >= 0

    csv_file = smoke_dir / "tier1_cost_test.csv"
    assert csv_file.exists()
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        for ck in cost_keys:
            assert ck in header


def test_axis2_paired_hierarchy_inferential_tests(tmp_path: Path) -> None:
    """C4: Axis 2 computes seed-by-seed paired statistical tests between Semantic and Ward hierarchies (HIER-01, AC 1, AC 2)."""
    tr_x, tr_y, val_x, val_y, te_x, te_y = _create_mock_scenario1_dataset(num_classes=20, samples_per_class=10)

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
        strategies=("mcts_enhanced",),
        tree_types=("semantic", "confusion_ward"),
        seeds=(42, 43),
        num_steps=5,
        eval_every=5,
        output_dir=smoke_dir,
        benchmark_name="tier1_axis2_tree_ablation",
        config=cfg,
    )

    assert "paired_hierarchy_tests" in res
    pht = res["paired_hierarchy_tests"]
    assert "mcts_enhanced" in pht

    enh_hier = pht["mcts_enhanced"]
    assert enh_hier["common_seeds"] == [42, 43]
    assert "delta_primary_mean" in enh_hier
    assert "delta_f1_mean" in enh_hier
    assert "primary_stats" in enh_hier
    assert "f1_stats" in enh_hier

    stats = enh_hier["primary_stats"]
    assert "mean_diff" in stats
    assert "t_stat" in stats
    assert "t_pvalue" in stats
    assert "cohens_d" in stats
    assert "ci_95" in stats


def test_axis2_persists_tree_topology_diagnostics(tmp_path: Path) -> None:
    """C5: Axis 2 records tree topological diagnostics in tier1_axis2_tree_ablation.json (HIER-01, AC 3, AC 4)."""
    tr_x, tr_y, val_x, val_y, te_x, te_y = _create_mock_scenario1_dataset(num_classes=20, samples_per_class=10)

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
        strategies=("mcts_enhanced",),
        tree_types=("semantic", "confusion_ward"),
        seeds=(42,),
        num_steps=5,
        eval_every=5,
        output_dir=smoke_dir,
        benchmark_name="tier1_axis2_tree_ablation",
        config=cfg,
    )

    assert "tree_diagnostics" in res
    assert "semantic" in res["tree_diagnostics"]
    assert "confusion_ward" in res["tree_diagnostics"]

    json_file = smoke_dir / "tier1_axis2_tree_ablation.json"
    assert json_file.exists()
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "tree_diagnostics" in data
    assert "semantic" in data["tree_diagnostics"]
    assert "confusion_ward" in data["tree_diagnostics"]
    sem_diag = data["tree_diagnostics"]["semantic"]
    assert "max_depth" in sem_diag
    assert "mean_leaf_depth" in sem_diag
    assert "total_nodes" in sem_diag
    assert "num_leaves" in sem_diag


def test_factorial_2x2_execution_and_persistence(tmp_path: Path) -> None:
    """C6: Factorial 2x2 execution computes and persists factorial decomposition payload (ABL-01, AC 1, AC 4)."""
    tr_x, tr_y, val_x, val_y, te_x, te_y = _create_mock_scenario1_dataset(num_classes=20, samples_per_class=10)

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
        strategies=("mcts_strict", "mcts_qnorm_only", "mcts_dualstream_only", "mcts_enhanced"),
        tree_types=("semantic",),
        seeds=(42, 43),
        num_steps=5,
        eval_every=5,
        output_dir=smoke_dir,
        benchmark_name="tier1_factorial_test",
        config=cfg,
    )

    assert "factorial_2x2_effects" in res
    assert "semantic" in res["factorial_2x2_effects"]

    sem_fact = res["factorial_2x2_effects"]["semantic"]
    assert sem_fact["common_seeds"] == [42, 43]
    assert "primary_effects" in sem_fact
    assert "f1_effects" in sem_fact

    prim_eff = sem_fact["primary_effects"]
    assert "main_effect_qnorm" in prim_eff
    assert "main_effect_dualstream" in prim_eff
    assert "interaction_effect" in prim_eff
    assert "variant_means" in prim_eff


def test_notebook_03_axis2_and_factorial_integration():
    """NOTE-01: Verify Notebook 03 contains paired hierarchy inference and 2x2 factorial tables."""
    root = Path(__file__).resolve().parent.parent
    nb_path = root / "notebooks" / "03_tier1_mcts_benchmarks.ipynb"
    assert nb_path.is_file()

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    full_text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])

    # Verify 2x2 Factorial Mechanism references
    assert "mcts_qnorm_only" in full_text
    assert "mcts_dualstream_only" in full_text
    assert "factorial_2x2_effects" in full_text
    assert "Main Effect Q-Norm" in full_text
    assert "Main Effect Dual-Stream" in full_text

    # Verify Hierarchy paired inferential references
    assert "paired_hierarchy_tests" in full_text
    assert "tree_diagnostics" in full_text
    assert "Formal Statistical Verdict" in full_text
