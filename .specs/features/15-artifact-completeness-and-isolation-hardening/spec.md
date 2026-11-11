# F15: Artifact Completeness, Metric Integrity, and Isolation Hardening Specification

**Approved**: 2026-11-10 - user: "usando tlc-spec-driven: Pendências encontradas..."

## Problem Statement

Following the initial implementation of the `ExecutionMode` contract and provenance sidecars in Features `07`-`14`, a comprehensive audit identified five concrete vulnerabilities and technical debts in artifact loading, isolation, and testing:

1. **P0 — Missing metrics silently converted to zero**:
   In `src/utils/paper_artifacts.py`, the loader and normalizer continue using conversions like `float(metrics.get("f1", 0.0))`, `float(metrics.get("auprc", 0.0))`, and `float(metrics.get("fpr_at_95_recall", metrics.get("fpr95", 0.0)))` across Scenario 1, Scenario 2, and tree ablation runs and summaries. Even though downstream table generators check for missing fields, this defense is defeated because the loader pre-populates missing values with `0.0`. The original JSON must be strictly validated before and during normalization, raising a fast error if any required metric is missing or null, never substituting zero.

2. **P0 — Incomplete result sets accepted silently**:
   In `src/utils/paper_artifacts.py`, `load_real_benchmark_results()` allows partial datasets with `if found_any and (merged["scenario1"] or merged["scenario2"]): return merged`. A dataset containing only Scenario 1 without Scenario 2 (or without ablation) is silently returned. Furthermore, JSON parsing exceptions are caught as `logger.warning(...)`, allowing incomplete or corrupted execution to proceed. For scientific publication, strict completeness validation per artifact/suite and explicit failure on any missing dependency is required.

3. **P1 — Notebook 00 test uses outdated FAST_MODE control**:
   In `tests/test_paper_artifacts.py`, `test_notebook_00_execution()` still requires the string `"FAST_MODE"` in `required_ref` and references it in comments, rather than checking modern `ExecutionMode` controls. The test suite must be fully updated to reflect the `ExecutionMode` migration.

4. **P1 — Automatic migration of legacy results on path resolution**:
   In `src/utils/experiment_config.py`, `get_isolated_paths()` has default `migrate_legacy: bool = True`. Resolving directories can unintentionally alter filesystem state by moving files to `results/legacy/`. The default must be changed to `False`, restricting migrations to explicit invocations of `migrate_legacy_results()`.

5. **P1 — Custom output directories bypass isolation validation**:
   In `src/training/benchmark_runner.py`, custom `output_dir` paths are passed directly (`else: out_path = Path(output_dir)`) without invoking `validate_mode_isolated_path()`. Custom paths must be validated prior to any disk writes to prevent cross-mode pollution or unsegregated writes under `results/`.

---

## Out of Scope

- Modifying the core RL MCTS search algorithms or model architectures.
- Altering the mathematical definitions of evaluation metrics.
- Running long multi-seed full-scale GPU benchmarks (all tests run in SMOKE or labeled fast REAL fixtures under the <60s SLA).

---

## Assumptions & Open Questions

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| Zero tolerance for fabricated metric defaults | All occurrences of `.get(..., 0.0)` in `src/utils/paper_artifacts.py` loader are replaced with strict key checks raising `ValueError` | Fabricated 0.0 metrics distort scientific tables and mask execution bugs | y |
| Strict benchmark completeness | `load_real_benchmark_results()` requires complete scenarios (`scenario1`, `scenario2`, `ablation`) when loading full suite, and halts immediately on parsing errors | Partial runs must not silently generate scientific paper deliverables | y |
| Explicit legacy migration | `get_isolated_paths(..., migrate_legacy=False)` by default; migration occurs only when explicitly opted in | Directory resolution must be a side-effect-free path query | y |
| Mandatory custom path validation | `benchmark_runner.py` validates custom `output_dir` via `validate_mode_isolated_path` | Ensures execution mode confinement regardless of user-specified output paths | y |
| Modernize test assertions | Update `tests/test_paper_artifacts.py` to assert `ExecutionMode` and remove legacy `FAST_MODE` | Ensures test assertions match production codebase contracts | y |

---

## User Stories

### P0: Eliminate Zero Defaults in Benchmark Loader (METRIC-01)

**User Story**: As a researcher, I want `load_real_benchmark_results` to reject missing metrics in benchmark JSON outputs rather than defaulting to `0.0`, so that missing experimental data is immediately caught.

**Acceptance Criteria**:
1. WHEN a benchmark JSON in Scenario 1 is missing `"f1"`, `"auprc"`, or `"fpr_at_95_recall"`/`"fpr95"` (in `runs` or `aggregated_summary`) THEN the loader SHALL raise `ValueError` detailing the missing metric and strategy, never substituting `0.0`.
2. WHEN a benchmark JSON in Scenario 2 is missing `"top1_accuracy"`, `"macro_f1"`/`"f1"`, or `"worst_10_class_recall"`/`"worst_10_recall"` THEN the loader SHALL raise `ValueError`, never substituting `0.0`.
3. WHEN a benchmark JSON in tree ablation is missing `"f1"` or `"auprc"` THEN the loader SHALL raise `ValueError`, never substituting `0.0`.
4. WHEN hyperparameter sensitivity grid entries miss `"f1_mean"` for any parameter coordinate THEN the loader SHALL raise `ValueError`, never substituting `0.0`.

### P0: Enforce Result Set Completeness & Fail-Fast Parsing (COMP-01)

**User Story**: As a researcher, I want `load_real_benchmark_results` to require all necessary benchmark components and fail fast on corrupted or partial files, so that partial experimental runs are never accepted.

**Acceptance Criteria**:
1. WHEN any benchmark JSON file exists but fails to parse or validate schema THEN the system SHALL raise the exception immediately rather than logging a warning and continuing.
2. WHEN `load_real_benchmark_results()` is invoked with `expected_mode=ExecutionMode.REAL` THEN the returned dictionary SHALL contain complete `scenario1`, `scenario2`, and `ablation` sections; if any required section or individual file is absent, it SHALL raise `FileNotFoundError` or `ValueError`.
3. WHEN `load_real_benchmark_results()` is invoked with `expected_mode=ExecutionMode.SMOKE` on an existing results directory THEN it SHALL require complete schemas if files are present, or fall back to synthesis via `load_or_synthesize_benchmark_results()`.

### P1: Modernize Notebook 00 Test Contract (TEST-01)

**User Story**: As a developer, I want `tests/test_paper_artifacts.py` to assert the presence of `ExecutionMode` and remove requirements for `FAST_MODE`, so that tests accurately reflect the current execution contract.

**Acceptance Criteria**:
1. WHEN `test_notebook_00_execution()` is executed THEN it SHALL verify `ExecutionMode` and `EXECUTION_MODE` in the code cells and SHALL NOT require `FAST_MODE`.
2. WHEN `git grep -n "FAST_MODE"` is executed across `src/` and `tests/` THEN no active test or code requirements shall depend on `FAST_MODE`.

### P1: Explicit-Only Migration of Legacy Results (MIG-01)

**User Story**: As a researcher, I want `get_isolated_paths()` to default `migrate_legacy=False`, so that querying path locations does not unexpectedly move files on disk.

**Acceptance Criteria**:
1. WHEN `get_isolated_paths()` is called with default arguments THEN legacy files directly under `results/` SHALL NOT be moved to `results/legacy/`.
2. WHEN `migrate_legacy_results()` is explicitly called (or `migrate_legacy=True` is explicitly passed) THEN legacy files under `results/` SHALL be migrated to `results/legacy/`.

### P1: Mandatory Path Isolation for Custom Output Directories (ISO-01)

**User Story**: As a researcher, I want `benchmark_runner.py` to validate any custom `output_dir` with `validate_mode_isolated_path()`, so that custom directories adhere to mode boundaries.

**Acceptance Criteria**:
1. WHEN `run_benchmark_multi_seed_and_export()` or `run_sensitivity_ablation()` receives a custom `output_dir` THEN it SHALL call `validate_mode_isolated_path(output_dir, mode=active_config.execution_mode)` before writing files.
2. WHEN a SMOKE benchmark specifies an unsegregated or REAL directory (e.g. `results/` or `results/real`) THEN the runner SHALL raise `ValueError`.

---

## Requirement Traceability

| Requirement ID | User Story | Acceptance Criteria |
| -------------- | ---------- | ------------------- |
| METRIC-01 | P0: Eliminate Zero Defaults | AC 1–4 |
| COMP-01 | P0: Result Completeness & Fail-Fast | AC 1–3 |
| TEST-01 | P1: Modernize Notebook 00 Test | AC 1–2 |
| MIG-01 | P1: Explicit Legacy Migration | AC 1–2 |
| ISO-01 | P1: Mandatory Custom Path Isolation | AC 1–2 |
