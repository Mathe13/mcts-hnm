"""Tests for Phase D: Scientific Campaign Runner and Manifest Orchestration."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.training.campaign_runner import run_scientific_campaign
from src.utils.experiment_config import ExecutionMode, ExperimentConfig


def _make_dummy_dataset():
    torch.manual_seed(42)
    np.random.seed(42)
    features = torch.randn(100, 16)
    labels = np.repeat(np.arange(10, dtype=np.int64), 10)
    val_features = torch.randn(40, 16)
    val_labels = np.repeat(np.arange(10, dtype=np.int64), 4)
    return features, labels, val_features, val_labels


def test_campaign_runner_scenario1_execution(tmp_path: Path) -> None:
    features, labels, val_features, val_labels = _make_dummy_dataset()
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)

    res = run_scientific_campaign(
        scenarios=("scenario1",),
        scenario1_ratios=("1:50",),
        scenario1_methods=("uniform", "mcts_strict"),
        seeds=(42, 43),
        target_class_scenario1=0,
        target_positive_count=4,
        config=config,
        train_features=features,
        train_labels=labels,
        val_features=val_features,
        val_labels=val_labels,
        output_dir=tmp_path / "campaign",
        num_steps_scenario1=4,
        eval_every=2,
    )

    assert res["completed_runs"] == 4  # 1 ratio * 2 methods * 2 seeds
    assert res["failed_runs"] == 0
    assert (tmp_path / "campaign" / "campaign_manifest.json").is_file()
    assert (tmp_path / "campaign" / "campaign_manifest.json.meta.json").is_file()

    # Check atomic run files
    for seed in (42, 43):
        for method in ("uniform", "mcts_strict"):
            run_id = f"s1_1_50_{method}_seed{seed}"
            run_path = tmp_path / "campaign" / f"{run_id}.json"
            assert run_path.is_file()
            assert (tmp_path / "campaign" / f"{run_id}.json.meta.json").is_file()
            with open(run_path) as f:
                data = json.load(f)
            assert data["status"] == "completed"
            assert "last_metrics" in data
            assert "best_validation_metrics" in data
            assert "computational_costs" in data


def test_campaign_runner_scenario2_execution(tmp_path: Path) -> None:
    features, labels, val_features, val_labels = _make_dummy_dataset()
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)

    res = run_scientific_campaign(
        scenarios=("scenario2",),
        scenario2_methods=("uniform", "uniform_grouped", "mcts_enhanced"),
        seeds=(42,),
        config=config,
        train_features=features,
        train_labels=labels,
        val_features=val_features,
        val_labels=val_labels,
        output_dir=tmp_path / "campaign_s2",
        num_steps_scenario2=4,
        eval_every=2,
    )

    assert res["completed_runs"] == 3
    assert res["failed_runs"] == 0
    assert (tmp_path / "campaign_s2" / "campaign_manifest.json").is_file()

    for method in ("uniform", "uniform_grouped", "mcts_enhanced"):
        run_id = f"s2_{method}_seed42"
        run_path = tmp_path / "campaign_s2" / f"{run_id}.json"
        assert run_path.is_file()
        with open(run_path) as f:
            data = json.load(f)
        assert data["status"] == "completed"
        assert "balanced_accuracy" in data["last_metrics"]
        assert "macro_f1" in data["last_metrics"]


def test_campaign_runner_provenance_and_manifest(tmp_path: Path) -> None:
    features, labels, val_features, val_labels = _make_dummy_dataset()
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)

    res = run_scientific_campaign(
        scenarios=("scenario1", "scenario2"),
        scenario1_ratios=("1:50",),
        scenario1_methods=("uniform",),
        scenario2_methods=("uniform",),
        seeds=(42,),
        target_positive_count=4,
        config=config,
        train_features=features,
        train_labels=labels,
        val_features=val_features,
        val_labels=val_labels,
        output_dir=tmp_path / "campaign_both",
        num_steps_scenario1=2,
        num_steps_scenario2=2,
        eval_every=2,
    )

    manifest_path = tmp_path / "campaign_both" / "campaign_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    assert manifest["mode"] == "smoke"
    assert manifest["total_expected_runs"] == 2
    assert manifest["completed_runs"] == 2
    assert len(manifest["runs"]) == 2
    assert "s1_1_50_uniform_seed42" in manifest["runs"]
    assert "s2_uniform_seed42" in manifest["runs"]


def test_campaign_runner_handles_failure_gracefully(tmp_path: Path) -> None:
    features, labels, val_features, val_labels = _make_dummy_dataset()
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)

    # Unknown strategy will raise inside train_scenario1
    res = run_scientific_campaign(
        scenarios=("scenario1",),
        scenario1_ratios=("1:50",),
        scenario1_methods=("non_existent_method",),
        seeds=(42,),
        target_positive_count=4,
        config=config,
        train_features=features,
        train_labels=labels,
        val_features=val_features,
        val_labels=val_labels,
        output_dir=tmp_path / "campaign_err",
        num_steps_scenario1=2,
        eval_every=2,
    )

    assert res["completed_runs"] == 0
    assert res["failed_runs"] == 1
    run_file = tmp_path / "campaign_err" / "s1_1_50_non_existent_method_seed42.json"
    assert run_file.is_file()
    with open(run_file) as f:
        data = json.load(f)
    assert data["status"] == "failed"
    assert data["error"] is not None
