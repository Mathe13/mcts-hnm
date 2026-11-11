"""Tests for Phase D: Publication Artifacts, Figures, and LaTeX Tables."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.utils.experiment_config import ExecutionMode
from src.utils.paper_artifacts import (
    generate_exposure_heatmap,
    generate_qnorm_dualstream_ablation_plot,
    generate_scenario1_latex_table,
    generate_scenario2_latex_table,
    generate_validation_curves_with_bands,
)


def _make_dummy_campaign_manifest(tmp_path: Path, mode: str = "smoke") -> dict:
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "campaign_id": f"test_campaign_{mode}",
        "mode": mode,
        "total_expected_runs": 2,
        "completed_runs": 2,
        "failed_runs": 0,
        "campaign_dir": str(campaign_dir),
        "runs": {
            "s1_1_50_uniform_seed42": {
                "run_id": "s1_1_50_uniform_seed42",
                "scenario": "scenario1",
                "imbalance_ratio": "1:50",
                "strategy": "uniform",
                "seed": 42,
                "status": "completed",
                "last_metrics": {"auprc": 0.12, "calibrated_f1": 0.25, "precision": 0.22, "recall": 0.30},
                "best_validation_metrics": {"auprc": 0.15, "calibrated_f1": 0.28, "precision": 0.25, "recall": 0.32},
                "computational_costs": {"total_updates": 100, "gradient_samples": 6400, "mining_forwards": 0, "elapsed_seconds": 12.5},
                "history": [
                    {"step": 50, "auprc": 0.10, "f1": 0.20},
                    {"step": 100, "auprc": 0.15, "f1": 0.28},
                ],
            },
            "s2_mcts_enhanced_seed42": {
                "run_id": "s2_mcts_enhanced_seed42",
                "scenario": "scenario2",
                "strategy": "mcts_enhanced",
                "seed": 42,
                "status": "completed",
                "last_metrics": {"top1_accuracy": 0.65, "macro_f1": 0.62, "balanced_accuracy": 0.63, "worst_10_class_recall": 0.40},
                "best_validation_metrics": {"top1_accuracy": 0.68, "macro_f1": 0.66, "balanced_accuracy": 0.67, "worst_10_class_recall": 0.45},
                "computational_costs": {"total_updates": 100, "gradient_samples": 6400, "mining_forwards": 0, "elapsed_seconds": 15.0},
                "history": [
                    {"step": 50, "macro_f1": 0.55},
                    {"step": 100, "macro_f1": 0.66},
                ],
            },
            "s2_mcts_strict_seed42": {
                "run_id": "s2_mcts_strict_seed42",
                "scenario": "scenario2",
                "strategy": "mcts_strict",
                "seed": 42,
                "status": "completed",
                "last_metrics": {"top1_accuracy": 0.58, "macro_f1": 0.54, "balanced_accuracy": 0.55, "worst_10_class_recall": 0.32},
                "best_validation_metrics": {"top1_accuracy": 0.60, "macro_f1": 0.56, "balanced_accuracy": 0.57, "worst_10_class_recall": 0.35},
                "computational_costs": {"total_updates": 100, "gradient_samples": 6400, "mining_forwards": 0, "elapsed_seconds": 14.0},
                "history": [
                    {"step": 50, "macro_f1": 0.50},
                    {"step": 100, "macro_f1": 0.56},
                ],
            },
        },
    }

    manifest_file = campaign_dir / "campaign_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    return manifest


def test_figure_generation_validation_and_ablation(tmp_path: Path) -> None:
    manifest = _make_dummy_campaign_manifest(tmp_path, mode="smoke")
    out_dir = tmp_path / "figures"

    # Validation curves
    res_curves = generate_validation_curves_with_bands(manifest, output_dir=out_dir)
    assert res_curves["pdf"].is_file()
    assert res_curves["png"].is_file()

    # Exposure heatmap
    res_heat = generate_exposure_heatmap(manifest, output_dir=out_dir)
    assert res_heat["pdf"].is_file()
    assert res_heat["png"].is_file()

    # 2x2 Factorial ablation plot
    res_abl = generate_qnorm_dualstream_ablation_plot(manifest, output_dir=out_dir)
    assert res_abl["pdf"].is_file()
    assert res_abl["png"].is_file()


def test_latex_table_generation_with_checkpoints_and_auprc(tmp_path: Path) -> None:
    manifest = _make_dummy_campaign_manifest(tmp_path, mode="smoke")
    tab_dir = tmp_path / "tables"

    # Scenario 1 Table
    s1_tex_file = tab_dir / "tab_s1.tex"
    s1_tex = generate_scenario1_latex_table(manifest, output_path=s1_tex_file)
    assert s1_tex_file.is_file()
    assert "AUPRC" in s1_tex
    assert "Calibrated F1" in s1_tex
    assert "Best Val" in s1_tex
    assert "Last" in s1_tex
    assert "Gradient Samples" in s1_tex

    # Scenario 2 Table
    s2_tex_file = tab_dir / "tab_s2.tex"
    s2_tex = generate_scenario2_latex_table(manifest, output_path=s2_tex_file)
    assert s2_tex_file.is_file()
    assert "Top-1 Accuracy" in s2_tex
    assert "Macro-F1" in s2_tex
    assert "Balanced Acc" in s2_tex
    assert "Worst-10 Recall" in s2_tex
    assert "Best Val" in s2_tex
    assert "Last" in s2_tex


def test_artifacts_reject_synthetic_or_missing_real_data(tmp_path: Path) -> None:
    # 1. Missing manifest
    with pytest.raises(FileNotFoundError, match="Campaign manifest not found"):
        generate_scenario1_latex_table(tmp_path / "nonexistent_dir")

    # 2. Incomplete campaign in REAL mode
    incomplete_manifest = {
        "campaign_id": "test_real",
        "mode": "real",
        "total_expected_runs": 10,
        "completed_runs": 4,
        "failed_runs": 1,
        "runs": {},
    }
    with pytest.raises(FileNotFoundError, match="incomplete"):
        generate_scenario1_latex_table(incomplete_manifest, expected_mode=ExecutionMode.REAL)
