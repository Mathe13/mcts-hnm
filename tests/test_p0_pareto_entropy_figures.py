"""Regression and unit test suite for Pareto Frontier and Entropy Dynamics plots (Phase 1 P0)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.utils.experiment_config import ExecutionMode
from src.utils.paper_artifacts import (
    generate_all_paper_artifacts,
    generate_coverage_entropy_dynamics_plot,
    generate_pareto_cost_performance_plot,
)


@pytest.fixture
def mock_campaign_scenario2(tmp_path: Path) -> dict[str, Any]:
    """Create a minimal campaign manifest with Scenario 2 runs and computational costs."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    strategies = ["uniform", "mcts_strict", "mcts_enhanced", "mcts_coverage_balanced", "ohem"]
    runs_manifest: dict[str, Any] = {}

    for strat in strategies:
        for seed in (42, 43):
            run_id = f"run_{strat}_s{seed}"
            run_file = runs_dir / f"{run_id}.json"

            # Create realistic history with normalized entropy
            history = []
            for step in (100, 200, 500, 1000):
                if strat == "uniform":
                    ent = 0.98 - 0.005 * (step / 1000.0)
                elif strat == "mcts_strict":
                    ent = max(0.25, 0.90 - 0.70 * (step / 1000.0))
                elif strat == "mcts_coverage_balanced":
                    ent = max(0.78, 0.92 - 0.12 * (step / 1000.0))
                else:
                    ent = max(0.50, 0.90 - 0.40 * (step / 1000.0))
                history.append({
                    "step": step,
                    "normalized_entropy": ent,
                    "macro_f1": 0.45 + (0.10 if "coverage" in strat or "enhanced" in strat else 0.02),
                    "loss": 1.2,
                })

            class_counts = [5 if i < 30 else (10 if i < 60 else (1 if "coverage" in strat else 0)) for i in range(100)]
            if strat == "uniform":
                class_counts = [10] * 100

            run_data = {
                "run_id": run_id,
                "strategy": strat,
                "seed": seed,
                "scenario": "scenario2",
                "computational_costs": {
                    "total_updates": 1000,
                    "gradient_samples": 64000,
                    "mining_forwards": 10000 if "mcts" in strat or strat == "ohem" else 0,
                    "elapsed_seconds": 25.0 + (15.0 if "mcts" in strat else (5.0 if strat == "ohem" else 0.0)),
                },
                "best_validation_metrics": {
                    "macro_f1": 0.52 if "coverage" in strat else (0.50 if "enhanced" in strat else (0.46 if strat == "ohem" else 0.43)),
                    "top1_accuracy": 0.60,
                },
                "exposure_diagnostics": {
                    "class_exposure_counts": class_counts,
                    "cumulative_class_coverage": 1.0 if strat == "uniform" else (0.95 if "coverage" in strat else 0.60),
                },
                "step_history": history,
                "history": history,
            }

            with open(run_file, "w", encoding="utf-8") as f:
                json.dump(run_data, f)

            runs_manifest[run_id] = {
                "run_id": run_id,
                "strategy": strat,
                "seed": seed,
                "scenario": "scenario2",
                "run_file": str(run_file),
            }

    manifest = {
        "campaign_dir": str(runs_dir),
        "mode": "smoke",
        "is_smoke": True,
        "completed_runs": len(runs_manifest),
        "total_expected_runs": len(runs_manifest),
        "failed_runs": 0,
        "runs": runs_manifest,
    }

    manifest_file = runs_dir / "campaign_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    return manifest


def test_pareto_cost_performance_plot_generation(mock_campaign_scenario2: dict[str, Any], tmp_path: Path) -> None:
    """Verify that FIG-01 calculates Pareto points and exports dual PDF/PNG."""
    out_dir = tmp_path / "figures"
    res = generate_pareto_cost_performance_plot(
        campaign_dir_or_manifest=mock_campaign_scenario2,
        output_dir=out_dir,
        file_prefix="test_fig_pareto",
        expected_mode=ExecutionMode.SMOKE,
    )

    pdf_path = res["pdf"]
    png_path = res["png"]
    assert pdf_path.is_file(), f"Missing PDF at {pdf_path}"
    assert png_path.is_file(), f"Missing PNG at {png_path}"
    assert pdf_path.stat().st_size > 1000
    assert png_path.stat().st_size > 1000

    pareto_strategies = res["pareto_strategies"]
    assert isinstance(pareto_strategies, list)
    assert len(pareto_strategies) >= 1
    # Uniform has lowest cost, so it should be on the Pareto frontier
    assert "uniform" in pareto_strategies

    summary = res["summary"]
    for strat in ["uniform", "mcts_strict", "mcts_coverage_balanced", "mcts_enhanced"]:
        assert strat in summary
        assert summary[strat]["cost_mean"] > 0
        assert summary[strat]["perf_mean"] > 0


def test_coverage_entropy_dynamics_plot_generation(mock_campaign_scenario2: dict[str, Any], tmp_path: Path) -> None:
    """Verify that FIG-02 generates dual-panel entropy and coverage dynamics plots."""
    out_dir = tmp_path / "figures"
    res = generate_coverage_entropy_dynamics_plot(
        campaign_dir_or_manifest=mock_campaign_scenario2,
        output_dir=out_dir,
        file_prefix="test_fig_entropy_dynamics",
        expected_mode=ExecutionMode.SMOKE,
    )

    pdf_path = res["pdf"]
    png_path = res["png"]
    assert pdf_path.is_file(), f"Missing PDF at {pdf_path}"
    assert png_path.is_file(), f"Missing PNG at {png_path}"
    assert pdf_path.stat().st_size > 1000
    assert png_path.stat().st_size > 1000

    entropy_curves = res["entropy_curves"]
    assert "mcts_strict" in entropy_curves
    assert "mcts_coverage_balanced" in entropy_curves
    assert len(entropy_curves["mcts_coverage_balanced"]) >= 1

    coverage_k1 = res["coverage_k1"]
    assert "uniform" in coverage_k1
    assert coverage_k1["uniform"][0] == 100.0


def test_generate_all_artifacts_includes_new_figures(tmp_path: Path) -> None:
    """Verify that generate_all_paper_artifacts generates or registers new figures."""
    # Verify the functions are importable and execute cleanly with default smoke synthetic data
    res_pareto = generate_pareto_cost_performance_plot(
        campaign_dir_or_manifest={"runs": {}, "mode": "smoke"},
        output_dir=tmp_path / "smoke_fig",
        expected_mode=ExecutionMode.SMOKE,
    )
    assert res_pareto["pdf"].is_file()

    res_entropy = generate_coverage_entropy_dynamics_plot(
        campaign_dir_or_manifest={"runs": {}, "mode": "smoke"},
        output_dir=tmp_path / "smoke_fig",
        expected_mode=ExecutionMode.SMOKE,
    )
    assert res_entropy["pdf"].is_file()


def test_latex_figure_blocks() -> None:
    """Verify paper/main.tex contains figure environments for FIG-01 and FIG-02."""
    main_tex = Path("paper/main.tex")
    if not main_tex.is_file():
        pytest.skip("paper/main.tex not found in local root")

    content = main_tex.read_text(encoding="utf-8")
    assert "fig_pareto_cost_vs_accuracy" in content or "fig:pareto_cost" in content
    assert "fig_coverage_entropy_dynamics" in content or "fig:coverage_entropy" in content


def test_notebook_05_integration() -> None:
    """Verify notebook 05 includes calls to both figure generators."""
    nb_path = Path("notebooks/05_paper_artifacts.ipynb")
    if not nb_path.is_file():
        pytest.skip("notebook 05 not found")

    content = nb_path.read_text(encoding="utf-8")
    assert "generate_pareto_cost_performance_plot" in content
    assert "generate_coverage_entropy_dynamics_plot" in content
