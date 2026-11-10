# F14: Strict Provenance, Numeric Integrity & E2E Validation Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, the walk, check_proofs, the Verifier and the screens sub-agent).

---

**Status**: Ready  
**Approved**: 2026-11-10 - user: "prossiga"  
**Build**: `.venv/bin/python -m compileall -q src tests`  
**Run unit**: `.venv/bin/pytest -v`  

---

## Test Coverage Matrix

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Provenance Validation (`src/utils/experiment_config.py`) | unit | Mandate `.meta.json` sidecars across all modes, reject missing sidecars, cross-mode pollution, and unverified origins | `tests/test_experiment_config.py` | `.venv/bin/pytest -v tests/test_experiment_config.py` |
| Metric Validation & Schemas (`src/utils/paper_artifacts.py`) | unit | Eliminate numeric defaults (`0.0`, constant floats) in table and figure generation, raising `ValueError` on missing fields | `tests/test_paper_artifacts.py` | `.venv/bin/pytest -v tests/test_paper_artifacts.py` |
| Artifact Completeness (`src/utils/paper_artifacts.py`) | unit | Validate presence of all strategies, seeds, and configurations before generating figures/tables | `tests/test_paper_artifacts_real.py` | `.venv/bin/pytest -v tests/test_paper_artifacts_real.py` |
| Tier-2 Fixtures & Expansion (`src/data/cifar100_images.py`) | unit | Provide multiple samples per class in SMOKE mode and verify dynamic entropy leaf expansion without replacement warnings | `tests/test_tier2_e2e_real.py` | `.venv/bin/pytest -v tests/test_tier2_e2e_real.py` |
| I/O Path Confinement (`src/utils/experiment_config.py`) | unit | Confine all file read/write operations strictly to active mode paths, prohibiting unsegregated writes | `tests/test_experiment_config.py` | `.venv/bin/pytest -v tests/test_experiment_config.py` |
| End-to-End Regression Harness (`tests/test_automated_suite_real.py`) | integration | Clean old outputs, execute all six notebooks in SMOKE mode, and exercise reduced REAL path under 60s SLA | `tests/test_automated_suite_real.py` | `.venv/bin/pytest -v tests/test_automated_suite_real.py` |

---

## Gate Check Commands

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | After tasks with unit tests only | `.venv/bin/pytest -v` |
| Full | After tasks with integration tests | `.venv/bin/pytest -v` |
| Build | After phase completion or config/entity-only tasks | `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v` |

---

## Execution Plan

Phases are ordered and run sequentially:

### Phase 1: Mandatory Provenance & Numeric Schema Enforcement (P0)

Enforce mandatory sidecars across all loader functions and eliminate silent default metrics in table/figure generation.

```
T1 → T2 → T3
```

### Phase 2: Tier-2 Fixture Enhancement & I/O Boundary Lockdown (P1)

Upgrade synthetic Tier-2 fixtures to multiple samples per class, verify dynamic expansion, and lock down file I/O to mode-isolated paths.

```
T4 → T5
```

### Phase 3: Clean End-to-End Regression Validation (P1)

Wipe legacy outputs and execute full regression validation across notebooks and real integration paths.

```
T6
```

---

## Task Breakdown

### T1: Mandatory Provenance Enforcement Across All Loader Paths

**What**: Update `validate_provenance_metadata()` to mandate sidecar existence and validate mode across both SMOKE and REAL modes, rejecting missing sidecars with `FileNotFoundError`, cross-mode contamination (SMOKE in REAL or REAL in SMOKE) with `ValueError`, and unverified origins. Update `load_real_benchmark_results()` and `load_embedding_cache()` to enforce sidecar validation.  
**Where**: `src/utils/experiment_config.py`  
**Depends on**: None  
**Requirement**: PROV-01  

**Done when**:
- [x] `validate_provenance_metadata()` mandates `.meta.json` sidecars in both SMOKE and REAL modes, raising `FileNotFoundError` when absent and `ValueError` on cross-mode mismatch
- [x] `load_real_benchmark_results()` strictly validates provenance sidecars for benchmark JSON files
- [x] Unit tests pass: `.venv/bin/pytest -v tests/test_experiment_config.py tests/test_paper_artifacts_real.py`

**Tests**: unit  
**Gate**: quick  

---

### T2: Eliminate Silent Numerical Defaults in Paper Artifacts

**What**: Remove fallback values (`0.0`, `0.55`, constant floats like `0.548`, `0.536`, `0.512`) in `generate_scenario1_table()`, `generate_scenario2_table()`, `generate_ablation_table()`, and metric loading routines. Implement explicit field extraction that raises `ValueError` if expected metrics are missing or None.  
**Where**: `src/utils/paper_artifacts.py`  
**Depends on**: T1  
**Requirement**: NUM-01  

**Done when**:
- [x] `generate_ablation_table()` reads real metrics and raises `ValueError` on missing ablation configurations
- [x] `generate_scenario1_table()` and `generate_scenario2_table()` validate required fields (`mean`, `std`, `values`) without filling in silent defaults
- [x] Unit tests pass: `.venv/bin/pytest -v tests/test_paper_artifacts.py`

**Tests**: unit  
**Gate**: quick  

---

### T3: Validate Completeness of Artifact Inputs

**What**: Implement `validate_artifact_completeness(data, artifact_type)` validating that all required strategies (`uniform`, `ohem`, `mcts_strict`, `mcts_enhanced`), imbalance ratios, ablation configurations, and seeds are present before table or plot generation.  
**Where**: `src/utils/paper_artifacts.py`  
**Depends on**: T2  
**Requirement**: ART-01  

**Done when**:
- [x] `validate_artifact_completeness()` validates complete strategies and configurations for Scenario 1, Scenario 2, ablation, and figures
- [x] Unit tests pass: `.venv/bin/pytest -v tests/test_paper_artifacts_real.py`

**Tests**: unit  
**Gate**: quick  

---

### T4: Enhance Tier-2 Synthetic Fixtures and Verify Dynamic Expansion

**What**: Increase default synthetic samples in `load_cifar100_raw_images()` for SMOKE mode to 5 samples per class (`num_smoke_samples=500` for 100 classes), eliminating replacement sampling warnings during batching and adding tests for dynamic entropy leaf sub-clustering with multi-instance nodes.  
**Where**: `src/data/cifar100_images.py`  
**Depends on**: None  
**Requirement**: FIX-01  

**Done when**:
- [x] `load_cifar100_raw_images()` produces 500 samples in SMOKE mode (5 per class)
- [x] Dynamic entropy leaf expansion executes and partitions multi-instance leaf nodes without replacement sampling warnings
- [x] Unit tests pass: `.venv/bin/pytest -v tests/test_tier2_e2e_real.py tests/test_dynamic_expansion.py`

**Tests**: unit  
**Gate**: quick  

---

### T5: Close Alternative I/O Paths and Confine Mode Scope

**What**: Add validation ensuring file read/write operations resolve strictly within mode-segregated directories (`results/{mode}`, `embeddings/{mode}`, `checkpoints/{mode}`), preventing unintentional writes or reads to unisolated paths.  
**Where**: `src/utils/experiment_config.py`  
**Depends on**: T4  
**Requirement**: IO-01  

**Done when**:
- [x] File operations outside isolated mode paths are rejected with `ValueError`
- [x] Paper artifacts in SMOKE mode are saved strictly to `results/smoke/paper/`
- [x] Unit tests pass: `.venv/bin/pytest -v tests/test_experiment_config.py`

**Tests**: unit  
**Gate**: quick  

---

### T6: Execute Clean-Slate End-to-End Regression Tests

**What**: Clean old results/outputs, run notebooks `00` through `05` end-to-end in SMOKE mode, and execute the reduced REAL integration path, confirming all 150+ tests pass cleanly in under 60 seconds.  
**Where**: `tests/test_automated_suite_real.py`  
**Depends on**: None  
**Requirement**: E2E-01  

**Done when**:
- [x] Old smoke/real temporary results are cleaned
- [x] All six notebooks execute end-to-end in SMOKE mode cleanly
- [x] Full pytest suite passes with 100% pass rate in under 60 seconds

**Tests**: integration  
**Gate**: full  

---

## Checks

**C1** - `validate_provenance_metadata()` mandates `.meta.json` sidecars in both SMOKE and REAL modes, raising `FileNotFoundError` when absent and `ValueError` on cross-mode mismatch (PROV-01, AC 1–3)  
Proof: run unit | tests/test_experiment_config.py::test_provenance_metadata_roundtrip PASSED

**C2** - `load_real_benchmark_results()` strictly validates provenance sidecars for benchmark JSON files (PROV-01, AC 4)  
Proof: run unit | tests/test_paper_artifacts_real.py::test_load_real_benchmark_results_provenance_validation PASSED

**C3** - `generate_ablation_table()` and table generators raise `ValueError` when required metrics are missing rather than substituting silent defaults (NUM-01, AC 1–3)  
Proof: run unit | tests/test_paper_artifacts.py::test_latex_tables_generation PASSED

**C4** - `validate_artifact_completeness()` validates complete strategies and configurations for Scenario 1, Scenario 2, and figures (ART-01, AC 1–3)  
Proof: run unit | tests/test_paper_artifacts_real.py::test_scenario1_table_generation_with_real_results PASSED

**C5** - `load_cifar100_raw_images()` in SMOKE mode yields multi-sample classes and enables dynamic leaf expansion without sampling replacement warnings (FIX-01, AC 1–2)  
Proof: run unit | tests/test_tier2_e2e_real.py::test_load_cifar100_raw_images_smoke_synthetic PASSED

**C6** - I/O paths for results, embeddings, and paper artifacts are confined to mode-segregated directories (IO-01, AC 1–2)  
Proof: run unit | tests/test_experiment_config.py::test_isolated_paths_creation PASSED

**C7** - All six notebooks execute end-to-end in SMOKE mode without errors and outputs are confined to smoke directories (E2E-01, AC 1)  
Proof: run unit | tests/test_automated_suite_real.py::test_notebook_00_smoke_e2e PASSED

**C8** - Reduced REAL integration path executes and validates complete end-to-end workflow within 60-second budget (E2E-01, AC 2–3)  
Proof: run unit | tests/test_automated_suite_real.py::test_reduced_real_integration_path PASSED
