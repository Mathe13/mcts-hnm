# Paper Artifacts and Real Integration - Checks

Profile: light
Plan: `.specs/features/11-paper-artifacts-real-integration/plan.md`

## Intent

10 checks in 3 slices · 3 one-way doors · 0 open

## Checks

### S1 - Execution Mode Enforcement and Real Benchmark Ingestion · 3 files · 25 KB · ~8k

**C1** - `load_or_synthesize_benchmark_results` in `ExecutionMode.REAL` raises `FileNotFoundError` when benchmark summary files are missing from `results_dir` (S1, AC 1)
Proof: `pytest tests/test_paper_artifacts_real.py::test_load_real_benchmark_missing_raises`

**C2** - `load_or_synthesize_benchmark_results` in `ExecutionMode.SMOKE` generates synthetic benchmark data (S1, AC 2)
Proof: `pytest tests/test_paper_artifacts_real.py::test_load_smoke_benchmark_synthetic`

**C3** - `load_real_benchmark_results` ingests and parses `tier1_axis1_samplers.json` and `tier1_scenario2_curriculum.json` into normalized dictionary (S1, AC 3)
Proof: `pytest tests/test_paper_artifacts_real.py::test_load_real_benchmark_parsing`

**C4** - Notebook 05 initial cell declares `EXECUTION_MODE = ExecutionMode.REAL` and outputs banner (S1, AC 4)
Proof: `pytest tests/test_paper_artifacts_real.py::test_notebook_05_structure_and_banner`

### S2 - Publication Figures, Tables, and Provenance Manifest · 3 files · 30 KB · ~10k

**C5** - `generate_convergence_plot`, `generate_tree_visitation_heatmap`, and `generate_hyperparameter_sensitivity_plot` generate valid PDF and PNG files (S2, AC 1)
Proof: `pytest tests/test_paper_artifacts_real.py::test_figure_generation_pdf_and_png`

**C6** - `generate_scenario1_table`, `generate_scenario2_table`, and `generate_ablation_table` generate valid LaTeX booktabs tables (S2, AC 1)
Proof: `pytest tests/test_paper_artifacts_real.py::test_table_generation_latex`

**C7** - In `ExecutionMode.SMOKE`, generated figures contain smoke watermark and save strictly into `results/smoke/paper/` (S2, AC 2)
Proof: `pytest tests/test_paper_artifacts_real.py::test_smoke_watermark_and_directory_isolation`

**C8** - In `ExecutionMode.REAL`, generated figures and tables save to `paper/figures/` and `paper/tables/` without watermark (S2, AC 3)
Proof: `pytest tests/test_paper_artifacts_real.py::test_real_artifacts_destination`

**C9** - `generate_all_paper_artifacts` writes `artifact_manifest.json` recording SHA-256 hashes of input files, git commit, mode, and timestamp (S2, AC 4)
Proof: `pytest tests/test_paper_artifacts_real.py::test_artifact_manifest_generation`

### S3 - Manuscript Compilation and Notebook 05 Standard · 2 files · 20 KB · ~6k

**C10** - `compile_paper` raises `RuntimeError` or `FileNotFoundError` when compiler fails or engine is missing (S3, AC 1)
Proof: `pytest tests/test_paper_artifacts_real.py::test_compile_paper_missing_compiler_raises`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| `load_or_synthesize_benchmark_results()` statuses (3) | 200 C2 · 400 C1 · 422 C1 | - |
| `load_real_benchmark_results()` statuses (2) | 200 C3 · 400 C1 | - |
| `generate_all_paper_artifacts()` statuses (3) | 200 C8 · 400 C7 · 422 C9 | - |
| `compile_paper()` statuses (2) | 200 C10 · 400 C10 | - |
| figures and tables generation statuses (2) | figures C5 · tables C6 | - |

- Claims naming a status code, route or response shape: C1, C2, C3, C8, C9, C10 - each has a proof that crosses the boundary
- No other check claims more than the single case its proof exercises

## Swept

- validation: C1, C3, C4
- failure modes: C1, C10
- idempotency: C5, C6, C9
- authorization: n/a - local research experimentation without user auth roles
- concurrency: C9
- data lifecycle: C7, C8, C9
- dependency failure: C1, C10
- state transitions: C1, C2
- observability: C4, C7, C9

## Handoff

- S1-S3 = 75 KB total across paper artifacts generation, metadata ingestion, compilation, and tests; estimate ~24k tokens, well under the 150k budget - one builder
