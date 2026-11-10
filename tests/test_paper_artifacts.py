"""Unit and integration tests for paper artifacts, figure/table generation, and manuscript compilation."""

import os
from pathlib import Path
import pytest
import yaml

from src.utils.paper_artifacts import (
    DEFAULT_PAPER_METADATA,
    generate_convergence_plot,
    generate_hyperparameter_sensitivity_plot,
    generate_tree_visitation_heatmap,
    load_or_synthesize_benchmark_results,
    load_paper_metadata,
)


def test_load_paper_metadata(tmp_path: Path):
    # 1. Existing root yaml
    meta = load_paper_metadata("paper_metadata.yaml")
    assert isinstance(meta, dict)
    assert "MCTS-HNM" in meta["title"]
    assert "authors" in meta
    assert len(meta["authors"]) >= 1
    assert "PPGC" in meta["institution"] or "PPGC" in meta["authors"][0]["affiliation"]

    # 2. Custom path
    custom_yaml = tmp_path / "custom_meta.yaml"
    custom_data = {
        "title": "Custom Test Paper",
        "authors": [{"name": "Test Author", "affiliation": "Test Univ"}],
        "institution": "Test Inst",
        "keywords": ["Test", "RL"],
    }
    with open(custom_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(custom_data, f)

    loaded_custom = load_paper_metadata(custom_yaml)
    assert loaded_custom["title"] == "Custom Test Paper"
    assert loaded_custom["authors"][0]["name"] == "Test Author"

    # 3. Non-existent path falls back to default
    non_existent = tmp_path / "non_existent.yaml"
    fallback = load_paper_metadata(non_existent)
    assert fallback["title"] == DEFAULT_PAPER_METADATA["title"]
    assert "PPGC" in fallback["institution"]


def test_benchmark_results_loader_and_synthesizer(tmp_path: Path):
    # Test synthetic fallback
    empty_results = tmp_path / "results_empty"
    data = load_or_synthesize_benchmark_results(empty_results, seed=42)
    assert "scenario1" in data
    assert "scenario2" in data
    assert "ablation" in data
    assert "convergence" in data
    assert "heatmap" in data
    assert "sensitivity" in data

    # Verify scenario1 strategies and metrics
    for strat in ["uniform", "ohem", "mcts_strict", "mcts_enhanced"]:
        assert strat in data["scenario1"]
        for rho in ["10", "50", "100"]:
            entry = data["scenario1"][strat][rho]
            assert "f1" in entry and "mean" in entry["f1"]
            assert len(entry["f1"]["values"]) == 5
            assert "auprc" in entry
            assert "fpr95" in entry

    # Verify scenario2
    for strat in ["uniform", "ohem", "mcts_strict", "mcts_enhanced"]:
        assert strat in data["scenario2"]
        scen2_entry = data["scenario2"][strat]
        assert "top1_accuracy" in scen2_entry
        assert "macro_f1" in scen2_entry
        assert "worst_10_recall" in scen2_entry


def test_publication_figures_generation(tmp_path: Path):
    fig_dir = tmp_path / "paper" / "figures"
    data = load_or_synthesize_benchmark_results(tmp_path, seed=42)

    # 1. Convergence plot
    conv_pdf, conv_png = generate_convergence_plot(data, output_dir=fig_dir)
    assert conv_pdf.exists() and conv_pdf.stat().st_size > 0
    assert conv_png.exists() and conv_png.stat().st_size > 0
    assert conv_pdf.suffix == ".pdf"
    assert conv_png.suffix == ".png"

    # 2. Heatmap
    heat_pdf, heat_png = generate_tree_visitation_heatmap(data, output_dir=fig_dir)
    assert heat_pdf.exists() and heat_pdf.stat().st_size > 0
    assert heat_png.exists() and heat_png.stat().st_size > 0

    # 3. Sensitivity plot
    sens_pdf, sens_png = generate_hyperparameter_sensitivity_plot(data, output_dir=fig_dir)
    assert sens_pdf.exists() and sens_pdf.stat().st_size > 0
    assert sens_png.exists() and sens_png.stat().st_size > 0

