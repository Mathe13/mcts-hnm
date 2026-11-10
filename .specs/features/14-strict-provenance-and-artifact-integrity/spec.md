# F14: Strict Provenance, Numeric Integrity & E2E Validation Specification

**Approved**: 2026-11-10 - user: "prossiga"

## Problem Statement

While Features `07` through `13` established the `ExecutionMode` contract (`SMOKE` vs `REAL`), directory segregation, and basic provenance sidecars (`.meta.json`), an audit reveals remaining vulnerabilities in data and artifact integrity:
1. Provenance sidecars are optional or skipped in several read paths (e.g. in `src/utils/paper_artifacts.py`, where `_validate_meta` ignores missing sidecars, and `validate_provenance_metadata` returns empty dictionaries for non-REAL modes). Unverified origins or cross-mode artifacts can be ingested without explicit rejection.
2. Artifact generation functions in `src/utils/paper_artifacts.py` use silent fallback numeric defaults (e.g., `0.0`, `0.55`, hardcoded ablation numbers `0.548`, `0.536`, `0.512`), which risk masking missing metrics or producing partial/distorted scientific tables and figures.
3. Figures and tables lack schema and completeness validation, allowing partial benchmark runs (e.g. missing strategies or seeds) to generate incomplete artifacts without failing fast.
4. Tier-2 synthetic image fixtures in `src/data/cifar100_images.py` only provide 1 sample per class in SMOKE mode (`num_smoke_samples=100`), causing batch sampling warnings and preventing thorough multi-sample dynamic leaf expansion testing.
5. Alternative and unsegregated I/O paths remain open where functions can read or write to unisolated directories (`results/`, `paper/`) without strict mode boundary checks.
6. A clean-slate end-to-end execution of all six notebooks in SMOKE mode and the reduced REAL integration path must be performed to guarantee regression-free reproducibility.

Feature `F14` closes these vulnerabilities, enforcing mandatory provenance, eliminating silent defaults, validating artifact schema completeness, enhancing Tier-2 fixtures, locking down I/O scopes, and executing clean-slate end-to-end regression tests.

## Out of Scope

Explicitly excluded to maintain focus and test execution SLA:
- Running multi-hour full REAL training runs across all 5 seeds (full scientific benchmarks are scheduled on cluster infrastructure; local validation uses labeled reduced fixtures under 60s).
- Altering the mathematical equations of MCTS-HNM or baseline loss functions.
- Modifying LaTeX typesetting templates beyond table/figure generation logic.

---

## Assumptions & Open Questions

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| Mandatory provenance validation | Every input artifact (`.pt`, `.json`, `.npz`, `.pth`) loaded by benchmark runner, tree loader, or paper artifact generator SHALL have a matching `.meta.json` sidecar verifying execution mode and origin; missing or mismatched sidecars raise `FileNotFoundError` or `ValueError` | Guarantees scientific auditability and prevents silent ingestion of unverified data | y |
| Metric completeness and schema validation | Replace all numeric default getters (`.get("f1", 0.0)`, hardcoded constants) with strict schema validation; missing fields raise `ValueError` | Prevents fabricated or distorted numbers in scientific publications | y |
| Tier-2 synthetic fixture size | Increase SMOKE synthetic images to at least 5 samples per class (500 samples across 100 classes) | Eliminates replacement sampling warnings and enables robust dynamic leaf clustering tests | y |
| I/O boundary lockdown | Direct unsegregated file operations outside mode directories (`results/{mode}`, `embeddings/{mode}`, `checkpoints/{mode}`) are prohibited and raise `ValueError` | Prevents cross-mode pollution and guarantees filesystem clean separation | y |
| End-to-end regression execution | Wipe legacy/stray outputs before running all 6 notebooks in SMOKE mode and the labeled REAL integration path under the 60s SLA | Verifies clean-slate reproducibility | y |
| Implicit dimensions (auth, network, external API, TTL) | N/A because all execution is self-contained local Python/PyTorch code | Self-contained computational pipeline | y |

**Open questions:** none - all requirements resolved and confirmed above.

---

## User Stories

### P0: Mandatory Provenance Enforcement (PROV-01)

**User Story**: As a researcher, I want all data loading operations to mandate valid `.meta.json` sidecars matching the active execution mode, so that files without sidecars, incompatible modes, or unverified origins are immediately rejected.

**Acceptance Criteria**:
1. WHEN `validate_provenance_metadata()` is invoked for any target file or directory THEN the system SHALL verify that the corresponding `.meta.json` sidecar exists and contains valid `execution_mode`, `dataset`, and `git_commit` fields, raising `FileNotFoundError` if the sidecar is absent.
2. WHEN an artifact is loaded under `ExecutionMode.REAL` whose sidecar indicates `ExecutionMode.SMOKE` or `synthetic_data=True` THEN the system SHALL raise `ValueError`.
3. WHEN an artifact is loaded under `ExecutionMode.SMOKE` whose sidecar indicates `ExecutionMode.REAL` THEN the system SHALL raise `ValueError` to prevent cross-mode cache poisoning.
4. WHEN `load_real_benchmark_results()` or `load_embedding_cache()` is called THEN the system SHALL enforce sidecar validation for every loaded file before parsing its content.

### P0: Eliminate Silent Numerical Defaults & Enforce Field Schemas (NUM-01)

**User Story**: As a researcher, I want all paper artifact generators to reject incomplete metric dictionaries instead of silently substituting defaults (`0.0`, constant floats), so that scientific figures and tables never display fabricated numbers.

**Acceptance Criteria**:
1. WHEN `load_real_benchmark_results()` processes benchmark runs THEN the system SHALL validate that all expected metric fields (`f1`, `auprc`, `fpr95` for Scenario 1; `top1_accuracy`, `macro_f1`, `worst_10_recall` for Scenario 2) are present and non-empty, raising `ValueError` if any metric is missing.
2. WHEN `generate_ablation_table()` is called THEN the system SHALL read real numerical metrics from the `ablation` dictionary without falling back to hardcoded constants, raising `ValueError` if any ablation configuration or metric is missing.
3. WHEN `generate_scenario1_table()` or `generate_scenario2_table()` is called THEN the system SHALL validate that all required metric summary fields (`mean`, `std`, `values`) exist for every strategy, raising `ValueError` if any field is missing.

### P0: Validate Completeness of Artifact Inputs (ART-01)

**User Story**: As a researcher, I want explicit schema validation verifying that all required strategies, seeds, and configurations are present before generating each figure or table, so that partial experimental outputs fail fast.

**Acceptance Criteria**:
1. WHEN `validate_artifact_completeness()` is invoked for Scenario 1 THEN the system SHALL verify that all 4 strategies (`uniform`, `ohem`, `mcts_strict`, `mcts_enhanced`) are present for all evaluated imbalance ratios, raising `ValueError` if any strategy is missing.
2. WHEN `validate_artifact_completeness()` is invoked for Scenario 2 THEN the system SHALL verify that all 4 strategies are present with valid multi-class metrics, raising `ValueError` if any strategy is missing.
3. WHEN `generate_convergence_plot()`, `generate_tree_visitation_heatmap()`, or `generate_hyperparameter_sensitivity_plot()` is invoked THEN the system SHALL verify that complete trajectory arrays and matrices are present, raising `ValueError` if data is incomplete.

### P1: Enhance Tier-2 Synthetic Fixtures & Dynamic Expansion Tests (FIX-01)

**User Story**: As a developer running tests in SMOKE mode, I want `load_cifar100_raw_images()` to provide multiple samples per class so that Tier-2 training batches and dynamic leaf expansion can be thoroughly tested without replacement sampling warnings.

**Acceptance Criteria**:
1. WHEN `load_cifar100_raw_images()` is called in `ExecutionMode.SMOKE` THEN the system SHALL generate synthetic 32x32 image tensors with at least 5 samples per class (defaulting to `num_smoke_samples=500` for 100 classes).
2. WHEN dynamic leaf expansion is tested in SMOKE mode THEN the system SHALL verify that leaf nodes containing multiple samples are successfully partitioned into sub-clusters by prediction entropy while maintaining tree structural invariants.

### P1: Close Alternative I/O Paths & Confine Modes (IO-01)

**User Story**: As a researcher, I want all file reading and writing operations strictly confined to mode-segregated directories (`results/{mode}`, `embeddings/{mode}`, `checkpoints/{mode}`), so that unisolated paths cannot be read or written.

**Acceptance Criteria**:
1. WHEN any pipeline stage writes results, checkpoints, embeddings, or paper artifacts THEN the system SHALL restrict the target destination to the active mode path defined by `IsolatedPaths`, raising `ValueError` if an output path attempts to escape the mode root.
2. WHEN `generate_all_paper_artifacts()` is executed in SMOKE mode THEN the system SHALL save all figures, tables, and manifests under `results/smoke/paper/` and never write to the root `paper/` directory.

### P1: Execute Clean End-to-End Regression Tests (E2E-01)

**User Story**: As a contributor, I want to wipe old outputs and execute all six notebooks in SMOKE mode and the reduced REAL integration path, verifying that all tests pass cleanly within the 60-second budget.

**Acceptance Criteria**:
1. WHEN the regression test suite is executed THEN the system SHALL wipe old test outputs and execute notebooks `00` through `05` in SMOKE mode without errors.
2. WHEN the reduced REAL integration path is executed THEN the system SHALL verify that the end-to-end pipeline succeeds with valid `.meta.json` sidecars.
3. WHEN the full pytest suite is executed THEN the system SHALL complete all unit and integration tests in under 60 seconds with 100% pass rate.

---

## Requirement Traceability

| Requirement ID | User Story | Acceptance Criteria |
| -------------- | ---------- | ------------------- |
| PROV-01 | P0: Mandatory Provenance Enforcement | AC 1, AC 2, AC 3, AC 4 |
| NUM-01 | P0: Eliminate Silent Numerical Defaults | AC 1, AC 2, AC 3 |
| ART-01 | P0: Validate Completeness of Artifact Inputs | AC 1, AC 2, AC 3 |
| FIX-01 | P1: Enhance Tier-2 Synthetic Fixtures | AC 1, AC 2 |
| IO-01 | P1: Close Alternative I/O Paths | AC 1, AC 2 |
| E2E-01 | P1: Execute Clean End-to-End Regression Tests | AC 1, AC 2, AC 3 |
