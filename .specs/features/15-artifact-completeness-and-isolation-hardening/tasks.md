# Tasks: F15 Artifact Completeness, Metric Integrity, and Isolation Hardening

Status: COMPLETED

## User Stories / Task Breakdown

### Phase 1: Test Modernization & Legacy Migration Default (P1)

- [x] Task 1.1: Modernize `tests/test_paper_artifacts.py` to replace `FAST_MODE` assertion with `ExecutionMode` (TEST-01)
  - Removed `"FAST_MODE"` from `required_ref` in `test_notebook_00_execution()`.
  - Added `"ExecutionMode"` and `"EXECUTION_MODE"` to `required_ref`.
  - Updated comments and execution environment to remove references to `FAST_MODE`.
  - Verified `test_notebook_00_execution` passes cleanly.

- [x] Task 1.2: Set `migrate_legacy: bool = False` by default in `get_isolated_paths` (MIG-01)
  - Updated `get_isolated_paths()` signature in `src/utils/experiment_config.py`.
  - Added unit test in `tests/test_experiment_config.py` verifying that default call does NOT move files, while explicit `migrate_legacy_results()` moves them.

### Phase 2: Mandatory Custom Output Directory Isolation (P1)

- [x] Task 2.1: Enforce `validate_mode_isolated_path` on custom `output_dir` in `benchmark_runner.py` (ISO-01)
  - Updated `run_benchmark()` and `run_sensitivity_ablation()` to validate custom `output_dir`.
  - Strengthened `validate_mode_isolated_path()` to ensure any path under `repo / results` belongs to `results/<mode>` or `results/legacy`.
  - Added unit test in `tests/test_tier1_benchmarks_real.py` verifying that unsegregated custom paths (`results/` or cross-mode `results/real` in SMOKE) raise `ValueError`.

### Phase 3: Eliminate Zero Defaults in Benchmark Loader & Fail-Fast Parsing (P0)

- [x] Task 3.1: Replace all `.get(..., 0.0)` with strict key validation in `src/utils/paper_artifacts.py` (METRIC-01)
  - For Scenario 1: validate `"f1"`, `"auprc"`, `"fpr_at_95_recall"`/`"fpr95"`. Raise `ValueError` if missing or None.
  - For Scenario 2: validate `"top1_accuracy"`, `"macro_f1"`/`"f1"`, `"worst_10_class_recall"`/`"worst_10_recall"`. Raise `ValueError` if missing or None.
  - For Ablation: validate `"f1"`, `"auprc"`. Raise `ValueError` if missing or None.
  - For Sensitivity: validate `"f1_mean"`. Raise `ValueError` if missing or None.
  - Removed all fallback `0.0` assignments.

- [x] Task 3.2: Enforce completeness and eliminate silent warning suppresses in `load_real_benchmark_results` (COMP-01)
  - Stopped catching JSON/parsing exceptions with `logger.warning(...)` when loading real benchmark files; re-raise immediately.
  - Replaced `if found_any and (merged["scenario1"] or merged["scenario2"])` with strict completeness verification: require `scenario1`, `scenario2`, and `ablation` for complete benchmark loading in REAL mode.
  - Added comprehensive unit tests in `tests/test_paper_artifacts_real.py` verifying that missing metrics and missing scenarios in JSON files raise `ValueError` and `FileNotFoundError`.

### Phase 4: Full Verification & Clean-Slate Auditing

- [x] Task 4.1: Run full test suite `python -m pytest tests/ -q --tb=short` (160/160 pass in 38.26s).
- [x] Task 4.2: Audit repository commands requested by user:
  - `git grep -n "FAST_MODE" -- notebooks src tests` (0 active references)
  - `git grep -n 'metrics.get(' -- src/utils/paper_artifacts.py` (0 occurrences)
  - Verified SMOKE notebook execution does not write to `results/real/` (0 files in `results/real/`).
- [x] Task 4.3: Generated `proofs.md` and `verification.md`.
- [x] Task 4.4: Record Architectural Decision AD-018 and update Handoff in `.specs/STATE.md`.
