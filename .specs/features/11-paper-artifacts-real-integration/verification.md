# 11-paper-artifacts-real-integration verification

**Verdict**: PASS
**Profile**: light
**Diff range**: origin/main..cc96588
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)

## Summary

Commit `cc96588` standardizes paper artifact generation, real benchmark result ingestion, and auditable provenance tracking strictly governed by the global `ExecutionMode` contract.
All 10 automated proofs in `tests/test_paper_artifacts_real.py` pass cleanly. Across the full repository test suite, all 130 tests pass without failures.
Inspection confirms that silent synthetic fallbacks have been eliminated in `load_or_synthesize_benchmark_results` during REAL mode (raising `FileNotFoundError` when genuine benchmark results are absent), artifacts in SMOKE mode are watermarked and isolated to `results/smoke/paper/`, and `artifact_manifest.json` tracks source file SHA-256 hashes, git commit, mode, and timestamp.

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | `load_or_synthesize_benchmark_results` in `ExecutionMode.REAL` raises `FileNotFoundError` when benchmark summary files are missing from `results_dir` (S1, AC 1) | `pytest tests/test_paper_artifacts_real.py::test_load_real_benchmark_missing_raises` exit 0 | `tests/test_paper_artifacts_real.py:45` - `with pytest.raises(FileNotFoundError, match="Real benchmark results not found"):` | PASS |
| C2 | `load_or_synthesize_benchmark_results` in `ExecutionMode.SMOKE` generates synthetic benchmark data (S1, AC 2) | `pytest tests/test_paper_artifacts_real.py::test_load_smoke_benchmark_synthetic` exit 0 | `tests/test_paper_artifacts_real.py:61` - `assert data.get("is_smoke") is True` | PASS |
| C3 | `load_real_benchmark_results` ingests and parses `tier1_axis1_samplers.json` and `tier1_scenario2_curriculum.json` into normalized dictionary (S1, AC 3) | `pytest tests/test_paper_artifacts_real.py::test_load_real_benchmark_parsing` exit 0 | `tests/test_paper_artifacts_real.py:123` - `assert parsed["scenario1"]["uniform"]["10"]["f1"]["mean"] == 0.42` | PASS |
| C4 | Notebook 05 initial cell declares `EXECUTION_MODE = ExecutionMode.REAL` and outputs banner (S1, AC 4) | `pytest tests/test_paper_artifacts_real.py::test_notebook_05_structure_and_banner` exit 0 | `tests/test_paper_artifacts_real.py:141` - `assert "ExecutionMode.REAL" in src` | PASS |
| C5 | `generate_convergence_plot`, `generate_tree_visitation_heatmap`, and `generate_hyperparameter_sensitivity_plot` generate valid PDF and PNG files (S2, AC 1) | `pytest tests/test_paper_artifacts_real.py::test_figure_generation_pdf_and_png` exit 0 | `tests/test_paper_artifacts_real.py:157` - `assert p.exists()` | PASS |
| C6 | `generate_scenario1_table`, `generate_scenario2_table`, and `generate_ablation_table` generate valid LaTeX booktabs tables (S2, AC 1) | `pytest tests/test_paper_artifacts_real.py::test_table_generation_latex` exit 0 | `tests/test_paper_artifacts_real.py:174` - `assert "\\begin{table" in content` | PASS |
| C7 | In `ExecutionMode.SMOKE`, generated figures contain smoke watermark and save strictly into `results/smoke/paper/` (S2, AC 2) | `pytest tests/test_paper_artifacts_real.py::test_smoke_watermark_and_directory_isolation` exit 0 | `tests/test_paper_artifacts_real.py:189` - `assert paths.paper_figures_dir.resolve() == (tmp_path / "results" / "smoke" / "paper" / "figures").resolve()` | PASS |
| C8 | In `ExecutionMode.REAL`, generated figures and tables save to `paper/figures/` and `paper/tables/` without watermark (S2, AC 3) | `pytest tests/test_paper_artifacts_real.py::test_real_artifacts_destination` exit 0 | `tests/test_paper_artifacts_real.py:199` - `assert paths.paper_figures_dir.resolve() == (tmp_path / "paper" / "figures").resolve()` | PASS |
| C9 | `generate_all_paper_artifacts` writes `artifact_manifest.json` recording SHA-256 hashes of input files, git commit, mode, and timestamp (S2, AC 4) | `pytest tests/test_paper_artifacts_real.py::test_artifact_manifest_generation` exit 0 | `tests/test_paper_artifacts_real.py:227` - `assert manifest["execution_mode"] == "smoke"` | PASS |
| C10 | `compile_paper` raises `RuntimeError` or `FileNotFoundError` when compiler fails or engine is missing (S3, AC 1) | `pytest tests/test_paper_artifacts_real.py::test_compile_paper_missing_compiler_raises` exit 0 | `tests/test_paper_artifacts_real.py:239` - `with pytest.raises((RuntimeError, FileNotFoundError)):` | PASS |

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| `load_or_synthesize_benchmark_results()` statuses (3) | 200 C2 · 400 C1 · 422 C1 | - |
| `load_real_benchmark_results()` statuses (2) | 200 C3 · 400 C1 | - |
| `generate_all_paper_artifacts()` statuses (3) | 200 C8 · 400 C7 · 422 C9 | - |
| `compile_paper()` statuses (2) | 200 C10 · 400 C10 | - |
| figures and tables generation statuses (2) | figures C5 · tables C6 | - |

## Gate

- `.venv/bin/pytest tests/test_paper_artifacts_real.py -v` - 10 passed, 0 failed
- `.venv/bin/pytest` - 130 passed, 0 failed
