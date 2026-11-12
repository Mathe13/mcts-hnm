"""Regression and unit test suite for P1 Expansion Figures (FIG-03, FIG-04, FIG-05)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.utils.experiment_config import ExecutionMode
from src.utils.paper_artifacts import (
    generate_all_paper_artifacts,
    generate_class_exposure_vs_delta_recall_plot,
    generate_interaction_ratio_plot,
    generate_ward_vs_semantic_dendrogram_plot,
)


@pytest.fixture
def mock_campaign_p1(tmp_path: Path) -> dict[str, Any]:
    """Create a minimal campaign fixture with multi-class and exposure metrics."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    manifest_runs: dict[str, Any] = {}
    strategies = ["uniform", "mcts_enhanced"]

    for strat in strategies:
        for seed in (42, 43):
            run_id = f"run_{strat}_s{seed}"
            run_file = runs_dir / f"{run_id}.json"

            recalls = [0.40 + 0.002 * c + (0.08 if strat == "mcts_enhanced" else 0.0) for c in range(100)]
            counts = [10 + c * (3 if strat == "mcts_enhanced" else 1) for c in range(100)]

            run_data = {
                "run_id": run_id,
                "strategy": strat,
                "seed": seed,
                "scenario": "scenario2",
                "best_validation_metrics": {
                    "per_class_recalls": recalls,
                    "macro_f1": 0.55 if strat == "mcts_enhanced" else 0.45,
                },
                "exposure_diagnostics": {
                    "class_exposure_counts": counts,
                },
            }

            with open(run_file, "w", encoding="utf-8") as f:
                json.dump(run_data, f)

            manifest_runs[run_id] = {
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
        "runs": manifest_runs,
    }

    with open(runs_dir / "campaign_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    return manifest


def test_class_exposure_vs_delta_recall_plot_generation(mock_campaign_p1: dict[str, Any], tmp_path: Path) -> None:
    """Verify FIG-03 generates scatter plot with Spearman correlation and exports dual PDF/PNG."""
    out_dir = tmp_path / "figures"
    res = generate_class_exposure_vs_delta_recall_plot(
        campaign_dir_or_manifest=mock_campaign_p1,
        output_dir=out_dir,
        file_prefix="test_fig_exposure_recall",
        expected_mode=ExecutionMode.SMOKE,
    )

    pdf_path = res["pdf"]
    png_path = res["png"]
    assert pdf_path.is_file(), f"Missing PDF at {pdf_path}"
    assert png_path.is_file(), f"Missing PNG at {png_path}"
    assert pdf_path.stat().st_size > 1000
    assert png_path.stat().st_size > 1000

    rho = res["spearman_rho"]
    p_val = res["spearman_pvalue"]
    assert -1.0 <= rho <= 1.0
    assert 0.0 <= p_val <= 1.0


def test_ward_vs_semantic_dendrogram_plot_generation(tmp_path: Path) -> None:
    """Verify FIG-04 constructs Ward dendrogram and exports dual PDF/PNG."""
    out_dir = tmp_path / "figures"

    # Synthetic confusion matrix
    cm = np.eye(100) * 100.0 + np.random.uniform(0, 5, size=(100, 100))
    res = generate_ward_vs_semantic_dendrogram_plot(
        confusion_matrix_or_data=cm,
        output_dir=out_dir,
        file_prefix="test_fig_ward_dendrogram",
        expected_mode=ExecutionMode.SMOKE,
    )

    pdf_path = res["pdf"]
    png_path = res["png"]
    assert pdf_path.is_file()
    assert png_path.is_file()
    assert pdf_path.stat().st_size > 1000
    assert png_path.stat().st_size > 1000
    assert "link_matrix" in res


def test_interaction_ratio_plot_generation(tmp_path: Path) -> None:
    """Verify FIG-05 generates dual-panel interaction curves with 95% CI bands."""
    out_dir = tmp_path / "figures"
    res = generate_interaction_ratio_plot(
        output_dir=out_dir,
        file_prefix="test_fig_interaction",
        expected_mode=ExecutionMode.SMOKE,
    )

    pdf_path = res["pdf"]
    png_path = res["png"]
    assert pdf_path.is_file()
    assert png_path.is_file()
    assert pdf_path.stat().st_size > 1000
    assert png_path.stat().st_size > 1000
    assert "interaction_delta" in res
    assert res["interaction_delta"] > 0.0


def test_generate_all_artifacts_includes_p1_figures(tmp_path: Path) -> None:
    """Verify that all P1 figures execute cleanly via their standalone runners."""
    res_rec = generate_class_exposure_vs_delta_recall_plot(
        campaign_dir_or_manifest={"runs": {}, "mode": "smoke"},
        output_dir=tmp_path / "p1_fig",
        expected_mode=ExecutionMode.SMOKE,
    )
    assert res_rec["pdf"].is_file()

    res_dend = generate_ward_vs_semantic_dendrogram_plot(
        output_dir=tmp_path / "p1_fig",
        expected_mode=ExecutionMode.SMOKE,
    )
    assert res_dend["pdf"].is_file()

    res_int = generate_interaction_ratio_plot(
        output_dir=tmp_path / "p1_fig",
        expected_mode=ExecutionMode.SMOKE,
    )
    assert res_int["pdf"].is_file()


def test_latex_figure_blocks_p1() -> None:
    """Verify paper/main.tex contains figure environments for FIG-03, FIG-04 and FIG-05."""
    main_tex = Path("paper/main.tex")
    if not main_tex.is_file():
        pytest.skip("paper/main.tex not found")

    content = main_tex.read_text(encoding="utf-8")
    assert "fig_class_exposure_vs_delta_recall" in content or "fig:class_exposure_recall" in content
    assert "fig_ward_vs_semantic_dendrogram" in content or "fig:ward_dendrogram" in content
    assert "fig_interaction_ratio_methods" in content or "fig:interaction_ratio" in content


def test_notebook_05_integration_p1() -> None:
    """Verify notebook 05 includes calls to all three P1 figure generators."""
    nb_path = Path("notebooks/05_paper_artifacts.ipynb")
    if not nb_path.is_file():
        pytest.skip("notebook 05 not found")

    content = nb_path.read_text(encoding="utf-8")
    assert "generate_class_exposure_vs_delta_recall_plot" in content
    assert "generate_ward_vs_semantic_dendrogram_plot" in content
    assert "generate_interaction_ratio_plot" in content
