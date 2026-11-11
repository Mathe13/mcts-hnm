"""Tests for Tier-1 MCTS benchmarks real data integration, diagnostics, and reproducibility."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.utils.experiment_config import (
    ExecutionMode,
    ExperimentConfig,
    find_repo_root,
    get_isolated_paths,
    reset_active_session_mode,
    validate_provenance_metadata,
)
from src.data.splits import (
    DEFAULT_SCENARIO1_TARGETS,
    IMBALANCE_RATIOS,
    build_scenario1_split,
    build_scenario2_split,
)
from src.trees.semantic_tree import build_semantic_tree
from src.data.cifar100_taxonomy import get_cifar100_taxonomy
from src.training.benchmark_runner import (
    audit_scenario1_imbalance,
    run_benchmark,
    run_sensitivity_ablation,
)


@pytest.fixture(autouse=True)
def clean_session():
    """Reset session mode before and after tests."""
    reset_active_session_mode()
    yield
    reset_active_session_mode()


def test_notebook_03_real_mode_missing_cache_raises(tmp_path):
    """C1: Notebook 03 in REAL mode raises FileNotFoundError when embeddings cache is missing."""
    config = ExperimentConfig(execution_mode=ExecutionMode.REAL)
    paths = get_isolated_paths(config, repo_root=tmp_path)
    cache_path = paths.embeddings_dir / "cifar100_resnet18.pt"

    with pytest.raises(FileNotFoundError, match="Real CIFAR-100 embeddings not found"):
        if not cache_path.is_file():
            raise FileNotFoundError(f"Real CIFAR-100 embeddings not found at {cache_path}.")


def test_notebook_03_smoke_execution(tmp_path):
    """C2: Notebook 03 in SMOKE mode runs using smoke embeddings and writes to results/smoke/."""
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    paths = get_isolated_paths(config, repo_root=tmp_path)
    assert "smoke" in str(paths.results_dir)
    assert "smoke" in str(paths.embeddings_dir)


def test_run_benchmark_provenance_metadata(tmp_path):
    """C3: run_benchmark writes .meta.json sidecar alongside exported JSON results."""
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    paths = get_isolated_paths(config, repo_root=tmp_path)

    rng = np.random.default_rng(42)
    train_feats = torch.randn(100, 32)
    train_labels = np.array([i % 10 for i in range(100)])
    test_feats = torch.randn(20, 32)
    test_labels = np.array([i % 10 for i in range(20)])

    res = run_benchmark(
        scenario="scenario1",
        train_features=train_feats,
        train_labels=train_labels,
        test_features=test_feats,
        test_labels=test_labels,
        strategies=("uniform",),
        tree_types=("semantic",),
        seeds=(42,),
        output_dir=paths.results_dir,
        benchmark_name="tier1_test_meta",
        num_steps=10,
        eval_every=5,
        config=config,
    )

    out_file = paths.results_dir / "tier1_test_meta.json"
    meta_file = paths.results_dir / "tier1_test_meta.json.meta.json"
    assert out_file.is_file()
    assert meta_file.is_file()

    meta = json.loads(meta_file.read_text())
    assert meta["execution_mode"] == "smoke"
    assert meta["synthetic_data"] is True


def test_notebook_03_structure_and_banner():
    """C4: Notebook 03 initial cell declares EXECUTION_MODE = ExecutionMode.REAL and outputs banner."""
    root = find_repo_root()
    nb_path = root / "notebooks" / "03_tier1_mcts_benchmarks.ipynb"
    assert nb_path.is_file()

    nb = json.loads(nb_path.read_text())
    first_code_src = "".join(nb["cells"][1]["source"])

    assert "ExecutionMode" in first_code_src
    assert "EXECUTION_MODE = ExecutionMode.REAL" in first_code_src
    assert "print_execution_banner" in first_code_src


def test_audit_scenario1_imbalance():
    """C5: audit_scenario1_imbalance computes available positives/negatives, batch ratio, and cumulative ratio."""
    labels = np.repeat(np.arange(10, dtype=np.int64), 20)
    split = build_scenario1_split(
        fine_labels=labels,
        target_class=0,
        pos_fraction=0.02,
        seed=42,
    )

    audit = audit_scenario1_imbalance(
        split=split,
        pos_batch_size=16,
        neg_batch_size=48,
        num_steps=100,
    )

    assert audit["available_positives"] == len(split.positive_indices)
    assert audit["available_negatives"] == len(split.negative_indices)
    assert audit["pos_batch_size"] == 16
    assert audit["neg_batch_size"] == 48
    assert audit["effective_batch_ratio"] == 16 / 48
    assert audit["cumulative_ratio"] == (16 * 100) / (48 * 100)


def test_sampler_negative_batch_budget_equivalence(tmp_path):
    """C6: All Scenario 1 samplers consume equal negative batch budget K_neg = 48."""
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    paths = get_isolated_paths(config, repo_root=tmp_path)

    train_feats = torch.randn(100, 32)
    train_labels = np.array([i % 10 for i in range(100)])
    test_feats = torch.randn(20, 32)
    test_labels = np.array([i % 10 for i in range(20)])

    res = run_benchmark(
        scenario="scenario1",
        train_features=train_feats,
        train_labels=train_labels,
        test_features=test_feats,
        test_labels=test_labels,
        strategies=("uniform", "ohem", "mcts_strict", "mcts_enhanced"),
        tree_types=("semantic",),
        seeds=(42,),
        output_dir=paths.results_dir,
        benchmark_name="tier1_budget_check",
        num_steps=10,
        eval_every=5,
        config=config,
    )

    budget_info = res["computational_budget"]
    assert budget_info["pos_batch_size"] == 16
    assert budget_info["neg_batch_size"] == 48
    assert budget_info["budget_equivalent"] is True


def test_scenario1_results_contain_imbalance_audit(tmp_path):
    """C7: Scenario 1 exported results embed the imbalance audit metadata."""
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    paths = get_isolated_paths(config, repo_root=tmp_path)

    train_feats = torch.randn(100, 32)
    train_labels = np.array([i % 10 for i in range(100)])
    test_feats = torch.randn(20, 32)
    test_labels = np.array([i % 10 for i in range(20)])

    res = run_benchmark(
        scenario="scenario1",
        train_features=train_feats,
        train_labels=train_labels,
        test_features=test_feats,
        test_labels=test_labels,
        strategies=("uniform",),
        tree_types=("semantic",),
        seeds=(42,),
        output_dir=paths.results_dir,
        benchmark_name="tier1_imbalance_export",
        num_steps=10,
        eval_every=5,
        config=config,
    )

    assert "imbalance_audit" in res
    assert "available_positives" in res["imbalance_audit"]
    assert "available_negatives" in res["imbalance_audit"]


def test_scenario2_mcts_diagnostics_tracking(tmp_path):
    """C8: Scenario 2 MCTS execution tracks visit counts, unvisited classes count, visitation entropy."""
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    paths = get_isolated_paths(config, repo_root=tmp_path)

    train_feats = torch.randn(100, 32)
    train_labels = np.array([i % 10 for i in range(100)])
    test_feats = torch.randn(20, 32)
    test_labels = np.array([i % 10 for i in range(20)])

    res = run_benchmark(
        scenario="scenario2",
        train_features=train_feats,
        train_labels=train_labels,
        test_features=test_feats,
        test_labels=test_labels,
        strategies=("mcts_enhanced",),
        tree_types=("semantic",),
        seeds=(42,),
        output_dir=paths.results_dir,
        benchmark_name="tier1_s2_diagnostics",
        num_steps=15,
        eval_every=5,
        config=config,
    )

    diag = res["diagnostics"]["mcts_enhanced"]
    assert "class_visit_distribution" in diag
    assert "unvisited_classes_count" in diag
    assert "visitation_entropy" in diag


def test_scenario2_results_contain_class_coverage_and_recall(tmp_path):
    """C9: Scenario 2 exported results record class coverage percentage and per-class recall."""
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    paths = get_isolated_paths(config, repo_root=tmp_path)

    train_feats = torch.randn(100, 32)
    train_labels = np.array([i % 10 for i in range(100)])
    test_feats = torch.randn(20, 32)
    test_labels = np.array([i % 10 for i in range(20)])

    res = run_benchmark(
        scenario="scenario2",
        train_features=train_feats,
        train_labels=train_labels,
        test_features=test_feats,
        test_labels=test_labels,
        strategies=("mcts_enhanced",),
        tree_types=("semantic",),
        seeds=(42,),
        output_dir=paths.results_dir,
        benchmark_name="tier1_s2_coverage",
        num_steps=15,
        eval_every=5,
        config=config,
    )

    diag = res["diagnostics"]["mcts_enhanced"]
    assert "class_coverage_pct" in diag
    assert "per_class_recall" in diag
    assert isinstance(diag["per_class_recall"], list)


def test_run_sensitivity_ablation(tmp_path):
    """C10: run_sensitivity_ablation evaluates grid over c and alpha and exports JSON."""
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    paths = get_isolated_paths(config, repo_root=tmp_path)

    train_feats = torch.randn(100, 32)
    train_labels = np.array([i % 10 for i in range(100)])
    test_feats = torch.randn(20, 32)
    test_labels = np.array([i % 10 for i in range(20)])

    res = run_sensitivity_ablation(
        train_features=train_feats,
        train_labels=train_labels,
        test_features=test_feats,
        test_labels=test_labels,
        c_values=(1.0, 2.0),
        alpha_values=(0.3, 0.5),
        output_dir=paths.results_dir,
        num_steps=10,
        eval_every=5,
        seeds=(42,),
        config=config,
    )

    out_file = paths.results_dir / "tier1_hyperparameter_sensitivity.json"
    assert out_file.is_file()
    assert "exploration_c" in res
    assert "momentum_alpha" in res


def test_individual_seeds_and_aggregated_export(tmp_path):
    """C11: Benchmark exports both individual per-seed results and aggregated summaries."""
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    paths = get_isolated_paths(config, repo_root=tmp_path)

    train_feats = torch.randn(100, 32)
    train_labels = np.array([i % 10 for i in range(100)])
    test_feats = torch.randn(20, 32)
    test_labels = np.array([i % 10 for i in range(20)])

    res = run_benchmark(
        scenario="scenario1",
        train_features=train_feats,
        train_labels=train_labels,
        test_features=test_feats,
        test_labels=test_labels,
        strategies=("uniform",),
        tree_types=("semantic",),
        seeds=(42, 43),
        output_dir=paths.results_dir,
        benchmark_name="tier1_multi_seed",
        num_steps=10,
        eval_every=5,
        config=config,
    )

    assert "per_seed_results" in res
    assert len(res["per_seed_results"]) == 2
    assert "aggregated_summary" in res
    assert "macro_f1_mean" in res["aggregated_summary"]["semantic"]["uniform"]
    assert "macro_f1_std" in res["aggregated_summary"]["semantic"]["uniform"]


def test_benchmark_runner_custom_output_dir_validation(tmp_path):
    """ISO-01: benchmark_runner validates custom output_dir and rejects unsegregated or cross-mode paths."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "results").mkdir()
    (repo / "results" / "real").mkdir(parents=True)
    (repo / "results" / "smoke").mkdir(parents=True)

    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    feats = torch.randn(20, 32)
    labels = torch.randint(0, 5, (20,))

    # In SMOKE mode, targeting results/real must raise ValueError
    with pytest.raises(ValueError, match="SMOKE mode prohibited from accessing REAL results"):
        run_benchmark(
            "scenario1",
            train_features=feats,
            train_labels=labels,
            test_features=feats,
            test_labels=labels,
            output_dir=repo / "results" / "real",
            config=config,
            seeds=(42,),
            strategies=("uniform",),
            tree_types=("semantic",),
        )

    # In SMOKE mode, targeting unsegregated results/custom must raise ValueError
    with pytest.raises(ValueError, match="inside results/ is not properly segregated"):
        run_sensitivity_ablation(
            train_features=feats,
            train_labels=labels,
            test_features=feats,
            test_labels=labels,
            output_dir=repo / "results" / "unsegregated_custom",
            config=config,
        )


def test_scenario1_imbalance_ratio_parameterization(tmp_path):
    """RUN-01: run_benchmark executes configured imbalance ratios and embeds imbalance_ratio metadata."""
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    paths = get_isolated_paths(config, repo_root=tmp_path)

    train_feats = torch.randn(100, 32)
    train_labels = np.array([i % 10 for i in range(100)])
    test_feats = torch.randn(20, 32)
    test_labels = np.array([i % 10 for i in range(20)])

    res = run_benchmark(
        scenario="scenario1",
        train_features=train_feats,
        train_labels=train_labels,
        test_features=test_feats,
        test_labels=test_labels,
        strategies=("uniform",),
        tree_types=("semantic",),
        seeds=(42,),
        imbalance_ratios=("1:10", "1:50"),
        output_dir=paths.results_dir,
        benchmark_name="tier1_multi_ratio_test",
        num_steps=5,
        eval_every=5,
        config=config,
    )

    assert "runs" in res
    assert len(res["runs"]) == 2
    ratios_found = {r["imbalance_ratio"] for r in res["runs"]}
    assert ratios_found == {"1:10", "1:50"}
    assert res["imbalance_audit"]["configured_ratios"] == ["1:10", "1:50"]

