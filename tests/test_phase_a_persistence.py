"""Tests for Phase A fine-grained run persistence and strictly paired statistics (A5)."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.training.benchmark_runner import run_benchmark
from src.utils.experiment_config import ExecutionMode, ExperimentConfig
from src.utils.statistics import compute_paired_statistical_tests


def test_atomic_run_id_persistence(tmp_path):
    """Verify that run_benchmark serializes individual records per run_id with metadata sidecars."""
    train_feats = torch.randn(100, 32)
    train_labels = np.array([i % 10 for i in range(100)], dtype=np.int64)
    val_feats = torch.randn(20, 32)
    val_labels = np.array([i % 10 for i in range(20)], dtype=np.int64)
    test_feats = torch.randn(20, 32)
    test_labels = np.array([i % 10 for i in range(20)], dtype=np.int64)

    exp_cfg = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    out_dir = tmp_path / "results" / "smoke"

    res = run_benchmark(
        scenario="scenario1",
        train_features=train_feats,
        train_labels=train_labels,
        val_features=val_feats,
        val_labels=val_labels,
        test_features=test_feats,
        test_labels=test_labels,
        strategies=("uniform", "mcts_enhanced"),
        tree_types=("semantic",),
        seeds=(42,),
        output_dir=out_dir,
        benchmark_name="granularity_test",
        num_steps=10,
        eval_every=5,
        config=exp_cfg,
    )

    runs_dir = out_dir / "runs"
    assert runs_dir.is_dir(), "runs/ subdirectory must be created for fine-grained run persistence."

    run_files = list(runs_dir.glob("*.json"))
    # Exclude .meta.json files to count raw runs
    raw_run_files = [f for f in run_files if not f.name.endswith(".meta.json")]
    assert len(raw_run_files) == 2, f"Expected 2 atomic run files, found {len(raw_run_files)}."

    for rf in raw_run_files:
        meta_file = rf.with_suffix(".json.meta.json")
        assert meta_file.is_file(), f"Missing .meta.json sidecar for run file: {rf}"
        run_data = json.loads(rf.read_text(encoding="utf-8"))
        assert "run_id" in run_data
        assert "final_metrics" in run_data
        assert "history" in run_data


def test_paired_statistics_key_validation():
    """Verify that compute_paired_statistical_tests enforces identical matching keys."""
    scores_a = [0.85, 0.88, 0.82]
    scores_b = [0.80, 0.84, 0.79]

    # Matching keys -> passes
    keys_matching = [(42, 0, "1:50"), (43, 0, "1:50"), (44, 0, "1:50")]
    stats = compute_paired_statistical_tests(
        scores_a,
        scores_b,
        keys_a=keys_matching,
        keys_b=keys_matching,
    )
    assert "t_stat" in stats
    assert stats["mean_diff"] > 0

    # Mismatched keys -> raises ValueError
    keys_mismatched = [(42, 0, "1:50"), (43, 0, "1:100"), (44, 0, "1:50")]
    with pytest.raises(ValueError, match="matching pairing keys"):
        compute_paired_statistical_tests(
            scores_a,
            scores_b,
            keys_a=keys_matching,
            keys_b=keys_mismatched,
        )
