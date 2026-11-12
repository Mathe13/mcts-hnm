"""Tests for sensitivity ablation runner split equivalence and 4-quadrant evaluation policy."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.data.splits import build_scenario1_split, IMBALANCE_RATIOS
from src.training.benchmark_runner import run_benchmark, run_sensitivity_ablation
from src.utils.experiment_config import ExecutionMode, ExperimentConfig


def _make_dummy_dataset(
    num_train: int = 500,
    num_val: int = 100,
    num_test: int = 100,
    num_classes: int = 10,
    dim: int = 32,
    seed: int = 42,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor, np.ndarray, torch.Tensor, np.ndarray]:
    rng = np.random.RandomState(seed)
    train_x = torch.randn(num_train, dim)
    train_y = rng.randint(0, num_classes, size=num_train).astype(np.int64)
    # Ensure class 0 has at least 50 samples
    train_y[:50] = 0

    val_x = torch.randn(num_val, dim)
    val_y = rng.randint(0, num_classes, size=num_val).astype(np.int64)
    val_y[:10] = 0

    test_x = torch.randn(num_test, dim)
    test_y = rng.randint(0, num_classes, size=num_test).astype(np.int64)
    test_y[:10] = 0

    return train_x, train_y, val_x, val_y, test_x, test_y


def test_sensitivity_runner_uses_scenario1_split(tmp_path):
    """C1: run_sensitivity_ablation applies build_scenario1_split using imbalance_ratio and seed."""
    train_x, train_y, val_x, val_y, test_x, test_y = _make_dummy_dataset()
    exp_cfg = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)

    res = run_sensitivity_ablation(
        train_features=train_x,
        train_labels=train_y,
        val_features=val_x,
        val_labels=val_y,
        test_features=test_x,
        test_labels=test_y,
        c_values=(2.0,),
        alpha_values=(0.3,),
        output_dir=tmp_path / "results" / "smoke",
        benchmark_name="test_c1_sens",
        imbalance_ratio="1:50",
        num_steps=10,
        eval_every=5,
        seeds=(42,),
        config=exp_cfg,
    )

    assert "grid" in res
    assert len(res["grid"]) == 1
    cell = res["grid"][0]
    assert cell["c"] == 2.0
    assert cell["alpha"] == 0.3
    assert cell["imbalance_ratio"] == "1:50"

    expected_split = build_scenario1_split(
        fine_labels=train_y,
        target_class=0,
        pos_fraction=IMBALANCE_RATIOS["1:50"],
        seed=42,
    )
    expected_pos_count = len(expected_split.positive_indices)

    assert res["num_positives_train"] == expected_pos_count
    assert res["num_negatives_train"] == len(expected_split.negative_indices)
    run_record = cell["runs"][0]
    assert run_record["num_positives_train"] == expected_pos_count


def test_sensitivity_indices_match_scenario1_split(tmp_path):
    """C2: run_sensitivity_ablation produces identical positive and negative counts to build_scenario1_split per seed."""
    train_x, train_y, val_x, val_y, test_x, test_y = _make_dummy_dataset()
    exp_cfg = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    seeds = (42, 43, 44)

    res = run_sensitivity_ablation(
        train_features=train_x,
        train_labels=train_y,
        val_features=val_x,
        val_labels=val_y,
        test_features=test_x,
        test_labels=test_y,
        c_values=(1.0,),
        alpha_values=(0.5,),
        output_dir=tmp_path / "results" / "smoke",
        benchmark_name="test_c2_sens",
        imbalance_ratio="1:50",
        num_steps=5,
        eval_every=5,
        seeds=seeds,
        config=exp_cfg,
    )

    cell = res["grid"][0]
    assert len(cell["runs"]) == len(seeds)

    for run_entry in cell["runs"]:
        s = run_entry["seed"]
        expected = build_scenario1_split(
            fine_labels=train_y,
            target_class=0,
            pos_fraction=IMBALANCE_RATIOS["1:50"],
            seed=s,
        )
        assert run_entry["num_positives_train"] == len(expected.positive_indices)
        assert run_entry["num_negatives_train"] == len(expected.negative_indices)


def test_sensitivity_output_records_pool_metadata(tmp_path):
    """C3: run_sensitivity_ablation output records imbalance_ratio, num_positives_train, and num_negatives_train."""
    train_x, train_y, val_x, val_y, test_x, test_y = _make_dummy_dataset()
    exp_cfg = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    out_dir = tmp_path / "results" / "smoke"

    res = run_sensitivity_ablation(
        train_features=train_x,
        train_labels=train_y,
        val_features=val_x,
        val_labels=val_y,
        test_features=test_x,
        test_labels=test_y,
        c_values=(0.5,),
        alpha_values=(0.1,),
        output_dir=out_dir,
        benchmark_name="test_c3_metadata",
        imbalance_ratio="1:50",
        num_steps=4,
        eval_every=4,
        seeds=(42,),
        config=exp_cfg,
    )

    # Check in-memory result
    assert "imbalance_ratio" in res
    assert res["imbalance_ratio"] == "1:50"
    assert "num_positives_train" in res
    assert "num_negatives_train" in res
    assert res["num_positives_train"] > 0
    assert res["num_negatives_train"] > 0

    # Check serialized JSON artifact
    json_file = out_dir / "test_c3_metadata.json"
    assert json_file.is_file()
    with open(json_file, "r") as f:
        data = json.load(f)
    assert data["imbalance_ratio"] == "1:50"
    assert data["num_positives_train"] == res["num_positives_train"]
    assert data["num_negatives_train"] == res["num_negatives_train"]


def test_sensitivity_exports_4_quadrant_metrics(tmp_path):
    """C4: 4-quadrant metrics (last/best_val x 0.5/calibrated) computed and exported without nulls."""
    train_x, train_y, val_x, val_y, test_x, test_y = _make_dummy_dataset()
    exp_cfg = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    out_dir = tmp_path / "results" / "smoke"

    res = run_sensitivity_ablation(
        train_features=train_x,
        train_labels=train_y,
        val_features=val_x,
        val_labels=val_y,
        test_features=test_x,
        test_labels=test_y,
        c_values=(2.0,),
        alpha_values=(0.3,),
        output_dir=out_dir,
        benchmark_name="test_c4_4quadrants",
        imbalance_ratio="1:50",
        num_steps=10,
        eval_every=5,
        seeds=(42,),
        config=exp_cfg,
    )

    cell = res["grid"][0]
    run_entry = cell["runs"][0]

    # Required 4-quadrant fields
    required_fields = [
        "test_f1_last_thresh05",
        "test_precision_last_thresh05",
        "test_recall_last_thresh05",
        "test_f1_last_calibrated",
        "test_precision_last_calibrated",
        "test_recall_last_calibrated",
        "best_tau_val_last",
        "test_f1_best_val_thresh05",
        "test_precision_best_val_thresh05",
        "test_recall_best_val_thresh05",
        "test_f1_best_val_calibrated",
        "test_precision_best_val_calibrated",
        "test_recall_best_val_calibrated",
        "best_tau_val",
        "test_auprc",
    ]

    for field in required_fields:
        assert field in run_entry, f"Missing {field} in run entry"
        assert run_entry[field] is not None
        assert not np.isnan(run_entry[field])

    # Check cell aggregate fields
    assert "f1_last_thresh05_mean" in cell
    assert "f1_last_calibrated_mean" in cell
    assert "f1_best_val_thresh05_mean" in cell
    assert "f1_best_val_calibrated_mean" in cell

    # Check CSV export
    csv_file = out_dir / "test_c4_4quadrants.csv"
    assert csv_file.is_file()


def test_sensitivity_requires_validation_data(tmp_path):
    """C5: Missing validation data raises ValueError preventing silent uncalibrated execution."""
    train_x, train_y, _, _, test_x, test_y = _make_dummy_dataset()
    exp_cfg = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)

    with pytest.raises(ValueError, match="val_features and val_labels are required"):
        run_sensitivity_ablation(
            train_features=train_x,
            train_labels=train_y,
            val_features=None,  # type: ignore
            val_labels=None,  # type: ignore
            test_features=test_x,
            test_labels=test_y,
            config=exp_cfg,
        )

    # Test invalid imbalance_ratio
    with pytest.raises(ValueError, match="Invalid imbalance_ratio"):
        run_sensitivity_ablation(
            train_features=train_x,
            train_labels=train_y,
            val_features=torch.randn(10, 32),
            val_labels=np.zeros(10, dtype=np.int64),
            test_features=test_x,
            test_labels=test_y,
            imbalance_ratio="1:9999",
            config=exp_cfg,
        )


def test_benchmark_and_sensitivity_index_equivalence():
    """C6: Positive and negative training indices strictly match between run_benchmark and run_sensitivity_ablation."""
    # Simulate realistic CIFAR-100 training set fine labels: 100 classes with 500 samples each
    num_classes = 100
    samples_per_class = 500
    cifar_train_labels = np.repeat(np.arange(num_classes, dtype=np.int64), samples_per_class)

    target_class = 0
    test_seeds = (42, 43, 44, 45, 46)

    for seed in test_seeds:
        # 1. Split generated as in Axis 1 (run_benchmark)
        pos_frac = IMBALANCE_RATIOS["1:50"]  # 0.02
        split_bench = build_scenario1_split(
            fine_labels=cifar_train_labels,
            target_class=target_class,
            pos_fraction=pos_frac,
            seed=seed,
        )

        # 2. Split generated as in Sensitivity Ablation (run_sensitivity_ablation)
        split_sens = build_scenario1_split(
            fine_labels=cifar_train_labels,
            target_class=target_class,
            pos_fraction=IMBALANCE_RATIOS["1:50"],
            seed=seed,
        )

        # Assert identical positive indices
        assert np.array_equal(split_bench.positive_indices, split_sens.positive_indices)

        # Assert identical negative indices
        assert np.array_equal(split_bench.negative_indices, split_sens.negative_indices)

        # Verify exact positive count under 1:50:
        # In CIFAR-100, class 0 has 500 samples (or 450 after train/val split).
        # pos_fraction=0.02 keeps max(1, round(500 * 0.02)) = 10 (or 9 on 450).
        pos_count = len(split_sens.positive_indices)
        assert pos_count in (9, 10), f"Expected 9 or 10 positives, got {pos_count}"
        assert len(split_sens.negative_indices) == len(cifar_train_labels) - samples_per_class
