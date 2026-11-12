"""Scientific integrity regressions for paper artifact generation."""
from __future__ import annotations

import pytest

from src.utils.experiment_config import ExecutionMode
from src.utils.paper_artifacts import (
    OptionalEvidenceMissing,
    generate_ablation_table,
    generate_interaction_ratio_plot,
    generate_precision_recall_curves_plot,
    generate_quadrant_evaluation_plot,
    _measured_factorial_plot,
)


def _measure(f1, ap):
    return {"f1": {"mean": f1, "std": 0.01},
            "auprc": {"mean": ap, "std": 0.02},
            "fpr95": {"mean": 0.2, "std": 0.01}}


def _scenario():
    return {
        "mcts_strict": {"50": _measure(0.697, 0.736)},
        "mcts_qnorm_only": {"50": _measure(0.656, 0.684)},
        "mcts_dualstream_only": {"50": _measure(0.711, 0.734)},
        "mcts_enhanced": {"50": _measure(0.680, 0.685)},
        "uniform": {"50": _measure(0.685, 0.708)},
        "ohem": {"50": _measure(0.670, 0.703)},
    }


def test_real_interaction_refuses_single_observed_ratio(tmp_path):
    with pytest.raises(OptionalEvidenceMissing, match="measured imbalance ratios"):
        generate_interaction_ratio_plot(
            {"scenario1": _scenario()}, output_dir=tmp_path,
            expected_mode=ExecutionMode.REAL,
        )
    assert not list(tmp_path.glob("fig_interaction_ratio_methods.*"))


def test_real_pr_refuses_unmeasured_curves(tmp_path):
    with pytest.raises(OptionalEvidenceMissing, match="Per-seed"):
        generate_precision_recall_curves_plot(
            {"runs": {}}, output_dir=tmp_path, expected_mode=ExecutionMode.REAL,
        )
    assert not list(tmp_path.glob("fig_precision_recall_curves.*"))


def test_real_quadrants_refuse_placeholder_values(tmp_path):
    with pytest.raises(OptionalEvidenceMissing, match="Four measured"):
        generate_quadrant_evaluation_plot(
            {}, output_dir=tmp_path, expected_mode=ExecutionMode.REAL,
        )
    assert not list(tmp_path.glob("fig_evaluation_4quadrants_comparison.*"))


def test_measured_factorial_uses_actual_values(tmp_path):
    data = {"scenario1": _scenario()}
    result = _measured_factorial_plot(data, tmp_path)
    assert result["ratio"] == "1:50"
    assert result["pdf"].is_file() and result["png"].is_file()


def test_ablation_table_uses_real_strict_reference(tmp_path):
    data = {"scenario1": _scenario(), "ablation": {
        "semantic_tree": {"f1": 0.680, "auprc": 0.685},
        "confusion_ward_tree": {"f1": 0.647, "auprc": 0.675},
    }}
    text = generate_ablation_table(data, tmp_path).read_text()
    assert "Strict GiantHunter" in text
    assert "Q-Norm only" in text
    assert "Dual-Stream only" in text
    assert "Enhanced / Ward Confusion" in text
    assert "+0.0" in text   # Strict reference
    assert "-1.7" in text   # Semantic Enhanced vs Strict, not 0.0
    assert "dynamic prediction-entropy" not in text


def test_real_ratio_figure_only_observed_ratios(tmp_path):
    s = _scenario()
    for strategy, ratios in s.items():
        ratios["100"] = _measure(0.55, 0.58)
    result = generate_interaction_ratio_plot(
        {"scenario1": s}, output_dir=tmp_path, expected_mode=ExecutionMode.REAL,
    )
    assert result["ratios"] == ["1:50", "1:100"]
    assert result["pdf"].is_file() and result["png"].is_file()
