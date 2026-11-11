"""Tests for Feature 11: Paper artifacts, real benchmark ingestion, and provenance manifest."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.utils.experiment_config import (
    ExecutionMode,
    ExperimentConfig,
    find_repo_root,
    get_isolated_paths,
    reset_active_session_mode,
    save_provenance_metadata,
)
from src.utils.paper_artifacts import (
    compile_paper,
    generate_ablation_table,
    generate_all_paper_artifacts,
    generate_artifact_manifest,
    generate_convergence_plot,
    generate_hyperparameter_sensitivity_plot,
    generate_scenario1_table,
    generate_scenario2_table,
    generate_tree_visitation_heatmap,
    load_or_synthesize_benchmark_results,
    load_real_benchmark_results,
    validate_artifact_completeness,
)


@pytest.fixture(autouse=True)
def clean_session():
    """Reset global execution mode before and after each test."""
    reset_active_session_mode()
    yield
    reset_active_session_mode()


def test_load_real_benchmark_missing_raises(tmp_path: Path):
    """C1: load_or_synthesize_benchmark_results in REAL mode raises FileNotFoundError if results are missing."""
    config = ExperimentConfig(execution_mode=ExecutionMode.REAL)
    empty_dir = tmp_path / "empty_results"
    empty_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="Real benchmark results not found"):
        load_or_synthesize_benchmark_results(results_dir=empty_dir, config=config)


def test_load_smoke_benchmark_synthetic(tmp_path: Path):
    """C2: load_or_synthesize_benchmark_results in SMOKE mode generates synthetic benchmark data."""
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    empty_dir = tmp_path / "smoke_results"
    empty_dir.mkdir(parents=True, exist_ok=True)

    data = load_or_synthesize_benchmark_results(results_dir=empty_dir, config=config)
    assert isinstance(data, dict)
    assert "scenario1" in data
    assert "scenario2" in data
    assert "ablation" in data
    assert "convergence" in data
    assert data.get("is_smoke") is True


def test_load_real_benchmark_parsing(tmp_path: Path):
    """C3: load_real_benchmark_results parses real benchmark summary JSON into normalized dictionary."""
    results_dir = tmp_path / "real_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Write a valid tier1_benchmark_summary.json fixture
    summary_data = {
        "scenario1": {
            "uniform": {
                "10": {"f1": {"mean": 0.42, "std": 0.01, "values": [0.42] * 5},
                       "auprc": {"mean": 0.40, "std": 0.01, "values": [0.40] * 5},
                       "fpr95": {"mean": 0.35, "std": 0.01, "values": [0.35] * 5}}
            },
            "mcts_enhanced": {
                "10": {"f1": {"mean": 0.56, "std": 0.01, "values": [0.56] * 5},
                       "auprc": {"mean": 0.55, "std": 0.01, "values": [0.55] * 5},
                       "fpr95": {"mean": 0.21, "std": 0.01, "values": [0.21] * 5}}
            }
        },
        "scenario2": {
            "uniform": {
                "top1_accuracy": {"mean": 0.64, "std": 0.01, "values": [0.64] * 5},
                "macro_f1": {"mean": 0.62, "std": 0.01, "values": [0.62] * 5},
                "worst_10_recall": {"mean": 0.38, "std": 0.01, "values": [0.38] * 5}
            },
            "mcts_enhanced": {
                "top1_accuracy": {"mean": 0.70, "std": 0.01, "values": [0.70] * 5},
                "macro_f1": {"mean": 0.69, "std": 0.01, "values": [0.69] * 5},
                "worst_10_recall": {"mean": 0.49, "std": 0.01, "values": [0.49] * 5}
            }
        },
        "ablation": {
            "q_norm_off": {"f1": 0.51, "auprc": 0.50},
            "q_norm_on": {"f1": 0.56, "auprc": 0.55}
        },
        "convergence": {
            "steps": [0, 500, 1000],
            "uniform_f1": [0.05, 0.30, 0.42],
            "uniform_loss": [0.70, 0.40, 0.28],
            "mcts_enhanced_f1": [0.05, 0.40, 0.56],
            "mcts_enhanced_loss": [0.70, 0.30, 0.12]
        },
        "heatmap": {
            "steps": [0, 500, 1000],
            "superclasses": ["SC-00", "SC-01"],
            "matrix": [[0.5, 0.3, 0.1], [0.5, 0.7, 0.9]]
        },
        "sensitivity": {
            "c_values": [0.5, 1.0, 2.0],
            "c_f1": [0.54, 0.56, 0.54],
            "alpha_values": [0.1, 0.3, 0.7],
            "alpha_f1": [0.54, 0.56, 0.53]
        }
    }
    summary_path = results_dir / "tier1_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f)
    save_provenance_metadata(summary_path, ExperimentConfig(execution_mode=ExecutionMode.REAL))

    parsed = load_real_benchmark_results(results_dir)
    assert parsed["scenario1"]["uniform"]["10"]["f1"]["mean"] == 0.42
    assert parsed["scenario2"]["mcts_enhanced"]["top1_accuracy"]["mean"] == 0.70


def test_load_real_benchmark_results_provenance_validation(tmp_path: Path):
    """C2: load_real_benchmark_results strictly validates provenance sidecars for benchmark JSON files."""
    results_dir = tmp_path / "provenance_test"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "tier1_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"scenario1": {"uniform": {"10": {"f1": [0.5]}}}}, f)

    # 1. Missing sidecar -> FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Missing mandatory provenance metadata in REAL mode"):
        load_real_benchmark_results(results_dir, expected_mode=ExecutionMode.REAL)

    # 2. SMOKE sidecar loaded in REAL mode -> ValueError
    save_provenance_metadata(summary_path, ExperimentConfig(execution_mode=ExecutionMode.SMOKE))
    with pytest.raises(ValueError, match="REAL mode rejected synthetic/smoke artifact"):
        load_real_benchmark_results(results_dir, expected_mode=ExecutionMode.REAL)

    # 3. REAL sidecar loaded in SMOKE mode -> ValueError
    reset_active_session_mode()
    save_provenance_metadata(summary_path, ExperimentConfig(execution_mode=ExecutionMode.REAL))
    with pytest.raises(ValueError, match="SMOKE mode rejected REAL artifact"):
        load_real_benchmark_results(results_dir, expected_mode=ExecutionMode.SMOKE)


def test_notebook_05_structure_and_banner():
    """C4: Notebook 05 initial cell declares EXECUTION_MODE = ExecutionMode.REAL and outputs banner."""
    root = find_repo_root()
    nb_path = root / "notebooks" / "05_paper_artifacts.ipynb"
    assert nb_path.exists(), f"Notebook not found at {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_json = json.load(f)

    cells = nb_json.get("cells", [])
    assert len(cells) > 0

    first_code_cell = next(c for c in cells if c.get("cell_type") == "code")
    src = "".join(first_code_cell.get("source", []))
    assert "ExecutionMode.REAL" in src
    assert "get_isolated_paths" in src
    assert "print_execution_banner" in src or "banner" in src.lower()


def test_figure_generation_pdf_and_png(tmp_path: Path):
    """C5: Figure generation produces valid PDF and PNG files."""
    out_dir = tmp_path / "figures"
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    data = load_or_synthesize_benchmark_results(config=config)

    c_pdf, c_png = generate_convergence_plot(data, output_dir=out_dir)
    h_pdf, h_png = generate_tree_visitation_heatmap(data, output_dir=out_dir)
    s_pdf, s_png = generate_hyperparameter_sensitivity_plot(data, output_dir=out_dir)

    for p in [c_pdf, c_png, h_pdf, h_png, s_pdf, s_png]:
        assert p.exists()
        assert p.stat().st_size > 0


def test_table_generation_latex(tmp_path: Path):
    """C6: Table generation produces valid LaTeX booktabs tables."""
    out_dir = tmp_path / "tables"
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    data = load_or_synthesize_benchmark_results(config=config)

    t1 = generate_scenario1_table(data, output_dir=out_dir)
    t2 = generate_scenario2_table(data, output_dir=out_dir)
    t3 = generate_ablation_table(data, output_dir=out_dir)

    for t in [t1, t2, t3]:
        assert t.exists()
        content = t.read_text(encoding="utf-8")
        assert "\\begin{table" in content
        assert "\\toprule" in content
        assert "\\bottomrule" in content
        assert "\\end{table" in content


def test_smoke_watermark_and_directory_isolation(tmp_path: Path):
    """C7: In SMOKE mode, figures contain smoke indicator and save under isolated smoke directory."""
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    paths = get_isolated_paths(config, repo_root=tmp_path)

    data = load_or_synthesize_benchmark_results(results_dir=paths.results_dir, config=config)
    assert data.get("is_smoke") is True

    c_pdf, c_png = generate_convergence_plot(data, output_dir=paths.paper_figures_dir, watermark="SMOKE TEST — NOT FOR PUBLICATION")
    assert paths.paper_figures_dir.resolve() == (tmp_path / "results" / "smoke" / "paper" / "figures").resolve()
    assert c_pdf.exists()
    assert c_png.exists()


def test_real_artifacts_destination(tmp_path: Path):
    """C8: In REAL mode, artifacts save to paper/figures/ and paper/tables/ without smoke watermark."""
    config = ExperimentConfig(execution_mode=ExecutionMode.REAL)
    paths = get_isolated_paths(config, repo_root=tmp_path)

    assert paths.paper_figures_dir.resolve() == (tmp_path / "paper" / "figures").resolve()
    assert paths.paper_tables_dir.resolve() == (tmp_path / "paper" / "tables").resolve()


def test_artifact_manifest_generation(tmp_path: Path):
    """C9: generate_all_paper_artifacts generates artifact_manifest.json with SHA-256 hashes and commit."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    paper_dir = tmp_path / "paper"
    meta_path = tmp_path / "paper_metadata.yaml"
    meta_path.write_text("title: Test Paper\n", encoding="utf-8")

    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)

    res = generate_all_paper_artifacts(
        results_dir=results_dir,
        paper_dir=paper_dir,
        metadata_path=meta_path,
        config=config,
    )

    manifest_path = res.get("manifest")
    assert manifest_path is not None
    assert Path(manifest_path).exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["execution_mode"] == "smoke"
    assert "timestamp_utc" in manifest
    assert "artifacts" in manifest
    assert len(manifest["artifacts"]) > 0
    # Every artifact entry has sha256
    for art in manifest["artifacts"]:
        assert "sha256" in art
        assert "file" in art


def test_compile_paper_missing_compiler_raises(tmp_path: Path):
    """C10: compile_paper raises RuntimeError if compiler engine is missing."""
    with pytest.raises((RuntimeError, FileNotFoundError)):
        compile_paper(paper_dir=tmp_path, engine="non_existent_latex_compiler_xyz_123")


def test_validate_artifact_completeness():
    """Verify validate_artifact_completeness checks valid schemas and rejects incomplete data."""
    # 1. Valid data from synthesizer passes all checks
    cfg = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    valid_data = load_or_synthesize_benchmark_results(config=cfg)

    for artifact in ("scenario1_table", "scenario2_table", "ablation_table", "convergence_plot", "visitation_heatmap", "sensitivity_plot"):
        validate_artifact_completeness(valid_data, artifact)

    # 2. Rejection of invalid types and missing sections
    with pytest.raises(ValueError, match="must be a dictionary"):
        validate_artifact_completeness("not_a_dict", "scenario1_table")  # type: ignore

    with pytest.raises(ValueError, match="Scenario 1 results dictionary is empty"):
        validate_artifact_completeness({}, "scenario1_table")

    with pytest.raises(ValueError, match="Scenario 2 results dictionary is empty"):
        validate_artifact_completeness({}, "scenario2_table")

    with pytest.raises(ValueError, match="Ablation results dictionary is empty"):
        validate_artifact_completeness({}, "ablation_table")

    with pytest.raises(ValueError, match="Convergence curve metrics missing"):
        validate_artifact_completeness({}, "convergence_plot")

    with pytest.raises(ValueError, match="visitation heatmap matrix missing"):
        validate_artifact_completeness({}, "visitation_heatmap")

    with pytest.raises(ValueError, match="sensitivity metrics .* missing"):
        validate_artifact_completeness({}, "sensitivity_plot")

    with pytest.raises(ValueError, match="Unknown artifact type"):
        validate_artifact_completeness(valid_data, "unknown_artifact")


def test_load_real_benchmark_results_rejects_missing_metrics(tmp_path):
    """METRIC-01: Missing metrics in raw JSON raise ValueError and are never substituted with 0.0."""
    res_dir = tmp_path / "results" / "real"
    res_dir.mkdir(parents=True)
    cfg = ExperimentConfig(execution_mode=ExecutionMode.REAL)

    # 1. Scenario 1 missing 'f1' in runs
    axis1_file = res_dir / "tier1_axis1_samplers.json"
    bad_data = {
        "runs": [
            {
                "strategy": "uniform",
                "final_metrics": {
                    # Missing "f1"!
                    "auprc": 0.45,
                    "fpr_at_95_recall": 0.30,
                },
            }
        ]
    }
    axis1_file.write_text(json.dumps(bad_data), encoding="utf-8")
    save_provenance_metadata(axis1_file, config=cfg)

    with pytest.raises(ValueError, match="Missing required metric 'f1' for strategy 'uniform'"):
        load_real_benchmark_results(res_dir, expected_mode=ExecutionMode.REAL)

    # 2. Scenario 2 missing 'top1_accuracy' in runs
    good_axis1 = {
        "scenario1": {
            "uniform": {
                "10": {
                    "f1": {"mean": 0.42, "std": 0.01, "values": [0.42]},
                    "auprc": {"mean": 0.40, "std": 0.01, "values": [0.40]},
                    "fpr95": {"mean": 0.35, "std": 0.01, "values": [0.35]},
                }
            }
        }
    }
    axis1_file.write_text(json.dumps(good_axis1), encoding="utf-8")
    save_provenance_metadata(axis1_file, config=cfg)

    scen2_file = res_dir / "tier1_scenario2_curriculum.json"
    bad_scen2 = {
        "runs": [
            {
                "strategy": "uniform",
                "final_metrics": {
                    # Missing 'top1_accuracy'!
                    "macro_f1": 0.50,
                    "worst_10_class_recall": 0.20,
                },
            }
        ]
    }
    scen2_file.write_text(json.dumps(bad_scen2), encoding="utf-8")
    save_provenance_metadata(scen2_file, config=cfg)

    with pytest.raises(ValueError, match="Missing required metric 'top1_accuracy' for strategy 'uniform'"):
        load_real_benchmark_results(res_dir, expected_mode=ExecutionMode.REAL)


def test_load_real_benchmark_results_rejects_incomplete_suite(tmp_path):
    """COMP-01: Incomplete benchmark results in REAL mode raise FileNotFoundError detailing missing files."""
    res_dir = tmp_path / "results" / "real"
    res_dir.mkdir(parents=True)
    cfg = ExperimentConfig(execution_mode=ExecutionMode.REAL)

    # Provide only Scenario 1, omit Scenario 2 and ablation
    axis1_file = res_dir / "tier1_axis1_samplers.json"
    good_axis1 = {
        "scenario1": {
            "uniform": {
                "10": {
                    "f1": {"mean": 0.42, "std": 0.01, "values": [0.42]},
                    "auprc": {"mean": 0.40, "std": 0.01, "values": [0.40]},
                    "fpr95": {"mean": 0.35, "std": 0.01, "values": [0.35]},
                }
            }
        }
    }
    axis1_file.write_text(json.dumps(good_axis1), encoding="utf-8")
    save_provenance_metadata(axis1_file, config=cfg)

    with pytest.raises(FileNotFoundError, match="missing required benchmark dependency files"):
        load_real_benchmark_results(res_dir, expected_mode=ExecutionMode.REAL)


def test_load_real_benchmark_results_re_raises_parsing_errors(tmp_path):
    """COMP-01: Corrupted or invalid JSON benchmark files fail fast and are not swallowed as warnings."""
    res_dir = tmp_path / "results" / "real"
    res_dir.mkdir(parents=True)
    cfg = ExperimentConfig(execution_mode=ExecutionMode.REAL)

    axis1_file = res_dir / "tier1_axis1_samplers.json"
    axis1_file.write_text("{ corrupted invalid json ...", encoding="utf-8")
    save_provenance_metadata(axis1_file, config=cfg)

    with pytest.raises(json.JSONDecodeError):
        load_real_benchmark_results(res_dir, expected_mode=ExecutionMode.REAL)


def test_load_real_benchmark_results_extracts_imbalance_ratio(tmp_path):
    """LOAD-01: load_real_benchmark_results normalizes imbalance_ratio from runs without hardcoding 1:10."""
    res_dir = tmp_path / "results" / "real"
    res_dir.mkdir(parents=True)
    cfg = ExperimentConfig(execution_mode=ExecutionMode.REAL)

    axis1_file = res_dir / "tier1_axis1_samplers.json"
    axis1_data = {
        "runs": [
            {
                "strategy": "uniform",
                "imbalance_ratio": "1:50",
                "final_metrics": {"f1": 0.35, "auprc": 0.32, "fpr_at_95_recall": 0.45},
            },
            {
                "strategy": "uniform",
                "imbalance_ratio": "1:100",
                "final_metrics": {"f1": 0.25, "auprc": 0.22, "fpr_at_95_recall": 0.55},
            },
            {
                "strategy": "mcts_enhanced",
                "imbalance_ratio": "1:50",
                "final_metrics": {"f1": 0.45, "auprc": 0.42, "fpr_at_95_recall": 0.35},
            },
            {
                "strategy": "mcts_enhanced",
                "imbalance_ratio": "1:100",
                "final_metrics": {"f1": 0.35, "auprc": 0.32, "fpr_at_95_recall": 0.42},
            },
        ]
    }
    axis1_file.write_text(json.dumps(axis1_data), encoding="utf-8")
    save_provenance_metadata(axis1_file, config=cfg)

    # Provide minimal scenario2 and ablation files to satisfy completeness in REAL mode
    scen2_file = res_dir / "tier1_scenario2_curriculum.json"
    scen2_file.write_text(json.dumps({
        "scenario2": {
            "uniform": {
                "top1_accuracy": {"mean": 0.60, "std": 0.01},
                "macro_f1": {"mean": 0.58, "std": 0.01},
                "worst_10_recall": {"mean": 0.35, "std": 0.01},
            },
            "mcts_enhanced": {
                "top1_accuracy": {"mean": 0.70, "std": 0.01},
                "macro_f1": {"mean": 0.68, "std": 0.01},
                "worst_10_recall": {"mean": 0.45, "std": 0.01},
            },
        }
    }), encoding="utf-8")
    save_provenance_metadata(scen2_file, config=cfg)

    ablation_file = res_dir / "tier1_axis2_tree_ablation.json"
    ablation_file.write_text(json.dumps({
        "ablation": {
            "semantic_tree": {"f1": 0.50, "auprc": 0.48},
            "confusion_ward_tree": {"f1": 0.54, "auprc": 0.52},
        }
    }), encoding="utf-8")
    save_provenance_metadata(ablation_file, config=cfg)

    parsed = load_real_benchmark_results(res_dir, expected_mode=ExecutionMode.REAL)
    assert "scenario1" in parsed
    scen1 = parsed["scenario1"]
    assert "50" in scen1["uniform"]
    assert "100" in scen1["uniform"]
    assert scen1["uniform"]["50"]["f1"]["mean"] == 0.35
    assert scen1["uniform"]["100"]["f1"]["mean"] == 0.25


def test_generate_scenario1_table_adapts_to_single_ratio(tmp_path):
    """TAB-01: generate_scenario1_table adapts to single executed ratio (1:50) with clean formatting."""
    out_dir = tmp_path / "tables"
    data = {
        "is_smoke": False,
        "scenario1": {
            strat: {
                "50": {
                    "f1": {"mean": 0.40, "std": 0.01, "values": [0.39, 0.41]},
                    "auprc": {"mean": 0.38, "std": 0.01, "values": [0.37, 0.39]},
                    "fpr95": {"mean": 0.30, "std": 0.01, "values": [0.29, 0.31]},
                }
            }
            for strat in ("uniform", "ohem", "mcts_strict", "mcts_enhanced")
        }
    }

    tex_path = generate_scenario1_table(data, output_dir=out_dir)
    assert tex_path.exists()
    content = tex_path.read_text(encoding="utf-8")

    assert "at imbalance ratio $\\rho = 1:50$" in content
    assert "1:50" in content
    assert "1:10" not in content
    assert "1:100" not in content
    # Ensure there is no trailing \midrule immediately before \bottomrule
    assert "\\midrule\n\\bottomrule" not in content
    assert "\\bottomrule" in content

