"""Regression and unit test suite for P2 PR Curves and 4-Quadrant Figures (FIG-06, FIG-07)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.utils.experiment_config import ExecutionMode
from src.utils.paper_artifacts import (
    generate_all_paper_artifacts,
    generate_precision_recall_curves_plot,
    generate_quadrant_evaluation_plot,
)


def test_precision_recall_curves_plot_generation(tmp_path: Path) -> None:
    """Verify FIG-06 computes continuous PR curves and exports dual PDF/PNG."""
    out_dir = tmp_path / "figures"
    res = generate_precision_recall_curves_plot(
        output_dir=out_dir,
        file_prefix="test_fig_pr_curves",
        expected_mode=ExecutionMode.SMOKE,
    )

    pdf_path = res["pdf"]
    png_path = res["png"]
    assert pdf_path.is_file(), f"Missing PDF at {pdf_path}"
    assert png_path.is_file(), f"Missing PNG at {png_path}"
    assert pdf_path.stat().st_size > 1000
    assert png_path.stat().st_size > 1000

    operating_points = res["operating_points"]
    for strat in ["uniform", "ohem", "mcts_strict", "mcts_enhanced"]:
        assert strat in operating_points
        assert 0.0 <= operating_points[strat]["recall"] <= 1.0
        assert 0.0 <= operating_points[strat]["precision"] <= 1.0


def test_quadrant_evaluation_plot_generation(tmp_path: Path) -> None:
    """Verify FIG-07 renders 4-quadrant grouped bars and exports dual PDF/PNG."""
    out_dir = tmp_path / "figures"
    res = generate_quadrant_evaluation_plot(
        output_dir=out_dir,
        file_prefix="test_fig_quadrant_eval",
        expected_mode=ExecutionMode.SMOKE,
    )

    pdf_path = res["pdf"]
    png_path = res["png"]
    assert pdf_path.is_file(), f"Missing PDF at {pdf_path}"
    assert png_path.is_file(), f"Missing PNG at {png_path}"
    assert pdf_path.stat().st_size > 1000
    assert png_path.stat().st_size > 1000

    means = res["quadrants_means"]
    for strat in ["uniform", "ohem", "mcts_strict", "mcts_enhanced"]:
        assert strat in means
        assert len(means[strat]) == 4  # Q1, Q2, Q3, Q4
        # Verify that MCTS Enhanced outperforms Uniform in every single quadrant
        assert means["mcts_enhanced"][0] > means["uniform"][0]
        assert means["mcts_enhanced"][1] > means["uniform"][1]
        assert means["mcts_enhanced"][2] > means["uniform"][2]
        assert means["mcts_enhanced"][3] > means["uniform"][3]


def test_generate_all_artifacts_includes_p2_figures(tmp_path: Path) -> None:
    """Verify standalone generation of P2 figures."""
    res_pr = generate_precision_recall_curves_plot(
        output_dir=tmp_path / "p2_fig",
        expected_mode=ExecutionMode.SMOKE,
    )
    assert res_pr["pdf"].is_file()

    res_q = generate_quadrant_evaluation_plot(
        output_dir=tmp_path / "p2_fig",
        expected_mode=ExecutionMode.SMOKE,
    )
    assert res_q["pdf"].is_file()


def test_latex_figure_blocks_p2() -> None:
    """Verify paper/main.tex contains figure environments for FIG-06 and FIG-07."""
    main_tex = Path("paper/main.tex")
    if not main_tex.is_file():
        pytest.skip("paper/main.tex not found")

    content = main_tex.read_text(encoding="utf-8")
    assert "fig_precision_recall_curves" in content or "fig:precision_recall_curves" in content
    assert "fig_evaluation_4quadrants_comparison" in content or "fig:evaluation_4quadrants" in content


def test_notebook_05_integration_p2() -> None:
    """Verify notebook 05 includes calls to both P2 figure generators."""
    nb_path = Path("notebooks/05_paper_artifacts.ipynb")
    if not nb_path.is_file():
        pytest.skip("notebook 05 not found")

    content = nb_path.read_text(encoding="utf-8")
    assert "generate_precision_recall_curves_plot" in content
    assert "generate_quadrant_evaluation_plot" in content
