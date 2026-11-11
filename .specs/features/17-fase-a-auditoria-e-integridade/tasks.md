# F17: Fase A — Auditoria e Integridade do Protocolo Experimental Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, the walk, check_proofs, the Verifier and the screens sub-agent).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Design**: `.specs/features/17-fase-a-auditoria-e-integridade/design.md`
**Status**: In Progress
**Approved**: 2026-11-11 - user: "sim"
**Build**: `.venv/bin/python -m compileall -q src tests`
**Run unit**: `.venv/bin/pytest -v tests/test_phase_a_audit.py tests/test_phase_a_splits.py tests/test_phase_a_data_segregation.py tests/test_phase_a_runner_equivalence.py tests/test_phase_a_persistence.py tests/test_phase_a_tree_validation.py tests/test_phase_a_gate.py`

---

## Test Coverage Matrix

> Generated from codebase, project guidelines, and spec - confirm before Execute. Guidelines found: `AGENTS.md`, `ROADMAP.md`, `roadmap_mcts_hnm_proxima_run_v2.md`.

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Documentation / Audit (`docs/experimental_audit.md`) | unit | File exists, catalog is complete, non-empty, and historical paths are documented | `tests/test_phase_a_audit.py` | `.venv/bin/pytest -v tests/test_phase_a_audit.py` |
| Config / Manifest (`src/utils/experiment_config.py`) | unit | Manifest contains commit, hardware, dependencies, embedding/split hashes; legacy immutability | `tests/test_phase_a_audit.py` | `.venv/bin/pytest -v tests/test_phase_a_audit.py` |
| Domain / Splits (`src/data/splits.py`) | unit | Real ratios 50/100/500, fixed 80 positives, nested negatives $N_{50} \subset N_{100} \subset N_{500}$, invalid ratio/counts fail-stop | `tests/test_phase_a_splits.py` | `.venv/bin/pytest -v tests/test_phase_a_splits.py` |
| Training Loop (`src/training/trainer.py`) | unit/integration | Periodic evaluation strictly on validation; test data never accessed during training; ValueError when val missing | `tests/test_phase_a_data_segregation.py` | `.venv/bin/pytest -v tests/test_phase_a_data_segregation.py` |
| Tree Calibration (`src/trees/confusion_tree.py`) | unit | Matrix estimation strictly on val/train calibration; ValueError if test passed | `tests/test_phase_a_data_segregation.py` | `.venv/bin/pytest -v tests/test_phase_a_data_segregation.py` |
| Runner Governance (`src/training/runner_config.py`) | unit | ResolvedTrainingConfig immutable, frozen, complete parameters; identical execution paths | `tests/test_phase_a_runner_equivalence.py` | `.venv/bin/pytest -v tests/test_phase_a_runner_equivalence.py` |
| Persistence & Stats (`src/training/benchmark_runner.py`, `src/utils/statistics.py`) | unit | Atomic run_id records; strict key pairing (seed, split, class, budget); rejection of unmatched keys | `tests/test_phase_a_persistence.py` | `.venv/bin/pytest -v tests/test_phase_a_persistence.py` |
| Hierarchy Clustering (`src/trees/confusion_tree.py`) | unit | Ward on Euclidean row-normalized confusion vectors; Average linkage on dissimilarity; no inversions | `tests/test_phase_a_tree_validation.py` | `.venv/bin/pytest -v tests/test_phase_a_tree_validation.py` |
| Verifier Gate A (`src/utils/gate_a_verifier.py`) | integration | Full evaluation of A1-A6 invariants; exit 0 on pass, non-zero on failure | `tests/test_phase_a_gate.py` | `.venv/bin/pytest -v tests/test_phase_a_gate.py` |

## Gate Check Commands

> Generated from codebase - confirm before Execute.

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | After tasks with unit tests only | `.venv/bin/pytest -v tests/test_phase_a_*.py` |
| Full | After tasks with integration tests | `.venv/bin/pytest -v tests/test_phase_a_*.py` |
| Build | After phase completion or config/entity-only tasks | `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_phase_a_*.py` |

---

## Execution Plan

Phases are ordered and run sequentially - each phase completes before the next begins, and tasks within a phase execute in order.

### Phase 1: Audit and Provenance Manifest (A1)

Creates `docs/experimental_audit.md` and implements run manifest generation with immutability guarantees.

```
T1 → T2
```

### Phase 2: Operational Imbalance Splitter with Nested Negatives (A2)

Implements `StratifiedImbalancedSplit` and `build_nested_imbalanced_splits` with 80 fixed positives and nested negatives ($N_{50} \subset N_{100} \subset N_{500}$).

```
T3
```

### Phase 3: Strict Data Segregation and Test Leak Prevention (A3)

Refactors `trainer.py` and `confusion_tree.py` to isolate validation data and prohibit test-set access during training or tree building.

```
T4 → T5
```

### Phase 4: Runner Configuration and Governance (A4)

Establishes `ResolvedTrainingConfig` and ensures deterministic equivalence between `run_benchmark` and `run_sensitivity_ablation`.

```
T6
```

### Phase 5: Granular Persistence and Paired Statistics (A5)

Serializes individual results per `run_id` and enforces strict key pairing in statistical hypothesis testing.

```
T7
```

### Phase 6: Mathematically Valid Hierarchies (A6)

Applies Ward clustering to Euclidean row-normalized confusion vectors and supports Average Linkage for general dissimilarities.

```
T8
```

### Phase 7: Gate A Verification (Gate A)

Implements `verify_gate_a()` validating all Phase A invariants.

```
T9
```

---

## Task Breakdown

### T1: Create `docs/experimental_audit.md` Catalog

**What**: Document the audit of historical runs, directory paths, embedding checksums, baseline parameters, and immutability rules.
**Where**: `docs/experimental_audit.md`
**Depends on**: None
**Reuses**: `ROADMAP_Improvements.md`, `roadmap_mcts_hnm_proxima_run_v2.md`
**Requirement**: AUD-01

**Done when**:

- [x] `docs/experimental_audit.md` exists and catalogs historical experiments, paths, and embedding hashes
- [x] Unit test in `tests/test_phase_a_audit.py::test_audit_documentation_present_and_complete` passes
- [x] Gate check passes: `.venv/bin/pytest -v tests/test_phase_a_audit.py`

**Tests**: unit
**Gate**: quick
**Commit**: `docs(audit): create experimental audit catalog and immutability rules`

---

### T2: Implement Run Manifest Generator and Historical Immutability

**What**: Implement `generate_run_manifest()` in `src/utils/experiment_config.py` capturing commit, hardware, dependencies, embedding/split hashes, and enforce immutability of `results/legacy/`.
**Where**: `src/utils/experiment_config.py`
**Depends on**: T1
**Reuses**: `src/utils/experiment_config.py`
**Requirement**: AUD-02

**Done when**:

- [x] `generate_run_manifest()` produces dictionary with commit, timestamp, hardware, dependencies, and split/embedding hashes
- [x] Attempts to overwrite files in `results/legacy/` raise `PermissionError`
- [x] Unit tests in `tests/test_phase_a_audit.py::test_generate_run_manifest` and `test_legacy_immutability` pass
- [x] Gate check passes: `.venv/bin/pytest -v tests/test_phase_a_audit.py`

**Tests**: unit
**Gate**: quick
**Commit**: `feat(config): implement run manifest generator and legacy immutability`

---

### T3: Implement Stratified Imbalanced Splitter with Nested Negatives

**What**: Implement `StratifiedImbalancedSplit`, `build_stratified_imbalanced_split()` and `build_nested_imbalanced_splits()` in `src/data/splits.py` supporting 80 fixed positives and nested negative sampling ($N_{50} \subset N_{100} \subset N_{500}$).
**Where**: `src/data/splits.py`
**Depends on**: T2
**Reuses**: `src/data/splits.py`
**Requirement**: SPLIT-01, SPLIT-02, SPLIT-03, SPLIT-04

**Done when**:

- [x] `build_stratified_imbalanced_split()` produces exact ratio $N_{\text{neg}}/N_{\text{pos}}$ and exact positive count
- [x] `build_nested_imbalanced_splits()` guarantees identical positives across 50, 100, 500 and strict negative subset containment $N_{50} \subset N_{100} \subset N_{500}$
- [x] Infeasible counts raise explicit `ValueError`
- [x] Official test split is un-resampled
- [x] Unit tests in `tests/test_phase_a_splits.py` pass
- [x] Gate check passes: `.venv/bin/pytest -v tests/test_phase_a_splits.py`

**Tests**: unit
**Gate**: quick
**Commit**: `feat(data): implement stratified split with nested negative sampling`

---

### T4: Segregate Training Evaluation to Validation Data in Trainer

**What**: Refactor `train_scenario1()` and `train_scenario2()` in `src/training/trainer.py` so periodic evaluation at `step % eval_every == 0` is strictly computed on `val_features`/`val_labels`, raising `ValueError` if validation data is missing.
**Where**: `src/training/trainer.py`
**Depends on**: T3
**Reuses**: `src/training/trainer.py`
**Requirement**: DATA-01, DATA-02

**Done when**:

- [x] `train_scenario1()` and `train_scenario2()` require `val_features` and `val_labels`
- [x] Periodic history logging evaluates exclusively on validation data
- [x] Test features are evaluated strictly once at final evaluation
- [x] Unit tests in `tests/test_phase_a_data_segregation.py::test_periodic_eval_uses_validation_strictly` pass
- [x] Gate check passes: `.venv/bin/pytest -v tests/test_phase_a_data_segregation.py`

**Tests**: unit
**Gate**: quick
**Commit**: `refactor(trainer): restrict periodic evaluation to validation data`

---

### T5: Prevent Test Data Leakage in Tree Calibration

**What**: Update `estimate_validation_confusion_matrix()` and tree construction routines in `src/trees/confusion_tree.py` to raise `ValueError` if test data is passed, requiring validation or out-of-fold calibration data.
**Where**: `src/trees/confusion_tree.py`
**Depends on**: T4
**Reuses**: `src/trees/confusion_tree.py`
**Requirement**: DATA-03

**Done when**:

- [x] `estimate_validation_confusion_matrix()` rejects test data and accepts validation or calibration data
- [x] Fallback from validation to test in `benchmark_runner.py` is removed, failing fast with `ValueError`
- [x] Unit tests in `tests/test_phase_a_data_segregation.py::test_tree_calibration_rejects_test_data` pass
- [x] Gate check passes: `.venv/bin/pytest -v tests/test_phase_a_data_segregation.py`

**Tests**: unit
**Gate**: quick
**Commit**: `refactor(trees): prevent test data leakage in tree calibration`

---

### T6: Implement Unified Resolved Training Config and Runner Equivalence

**What**: Implement immutable `ResolvedTrainingConfig` in `src/training/runner_config.py` and unify parameter resolution between `run_benchmark()` and `run_sensitivity_ablation()`.
**Where**: `src/training/runner_config.py`
**Depends on**: T5
**Reuses**: `src/training/benchmark_runner.py`
**Requirement**: RUN-01, RUN-02

**Done when**:

- [x] `ResolvedTrainingConfig` encapsulates and validates all training hyperparameters
- [x] `run_benchmark` and `run_sensitivity_ablation` share config resolution and propagate identical parameters
- [x] Deterministic equivalence test confirms matching loss/metrics on identical configs
- [x] Unit tests in `tests/test_phase_a_runner_equivalence.py` pass
- [x] Gate check passes: `.venv/bin/pytest -v tests/test_phase_a_runner_equivalence.py`

**Tests**: unit
**Gate**: quick
**Commit**: `feat(training): implement unified resolved training config`

---

### T7: Implement Fine-Grained Run ID Persistence and Paired Statistics

**What**: Serialize each benchmark execution to an atomic `results/{mode}/runs/{run_id}.json` file and enforce strict key pairing (seed, split, target class, budget) in `compute_paired_statistical_tests()`.
**Where**: `src/training/benchmark_runner.py`
**Depends on**: T6
**Reuses**: `src/training/benchmark_runner.py`, `src/utils/statistics.py`
**Requirement**: STORE-01, STORE-02

**Done when**:

- [ ] Each run produces an isolated `run_id` file with configuration and step history
- [ ] Paired statistical tests check matching seeds, split hashes, target classes, and budgets, raising `ValueError` on mismatch
- [ ] Unit tests in `tests/test_phase_a_persistence.py` pass
- [ ] Gate check passes: `.venv/bin/pytest -v tests/test_phase_a_persistence.py`

**Tests**: unit
**Gate**: quick
**Commit**: `feat(persistence): implement fine-grained run_id serialization and strictly paired statistics`

---

### T8: Mathematically Validate Ward Clustering over Euclidean Confusion Vectors

**What**: Update `build_confusion_tree()` in `src/trees/confusion_tree.py` to compute Ward linkage over row-normalized confusion vectors in $\mathbb{R}^K$ and Average Linkage over confusion distance matrices.
**Where**: `src/trees/confusion_tree.py`
**Depends on**: T7
**Reuses**: `src/trees/confusion_tree.py`
**Requirement**: TREE-01, TREE-02

**Done when**:

- [ ] Ward clustering operates directly on row-normalized confusion vectors $p_i \in \mathbb{R}^K$
- [ ] Average Linkage (UPGMA) operates directly on condensed confusion distance matrices
- [ ] Tree structural invariants (depth, leaf counts, monotonicity) are verified with zero Lance-Williams inversions
- [ ] Unit tests in `tests/test_phase_a_tree_validation.py` pass
- [ ] Gate check passes: `.venv/bin/pytest -v tests/test_phase_a_tree_validation.py`

**Tests**: unit
**Gate**: quick
**Commit**: `feat(trees): implement Euclidean row-normalized Ward clustering and Average Linkage`

---

### T9: Implement Gate A Verifier and Stop Verification

**What**: Implement `verify_gate_a()` in `src/utils/gate_a_verifier.py` evaluating all Phase A invariants and returning 0 on pass.
**Where**: `src/utils/gate_a_verifier.py`
**Depends on**: T8
**Reuses**: All Phase A modules
**Requirement**: GATE-01

**Done when**:

- [ ] `verify_gate_a()` evaluates splits, data segregation, runner equivalence, persistence granularity, and tree geometry
- [ ] Integration test `tests/test_phase_a_gate.py` passes with exit code 0
- [ ] Gate check passes: `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_phase_a_*.py`

**Tests**: integration
**Gate**: build
**Commit**: `feat(verification): implement Gate A verifier and end-to-end integrity check`

---

## Checks

**C1** - `docs/experimental_audit.md` exists and catalogs historical execution runs and embedding hashes (AUD-01, AC 1)
Proof: run unit | tests/test_phase_a_audit.py::test_audit_documentation_present_and_complete PASSED

**C2** - `generate_run_manifest()` generates an auditable dictionary with commit, hardware, dependencies, and split/embedding hashes (AUD-02, AC 2)
Proof: run unit | tests/test_phase_a_audit.py::test_generate_run_manifest PASSED

**C3** - Attempts to modify files in `results/legacy/` raise `PermissionError` (AUD-02, AC 3)
Proof: run unit | tests/test_phase_a_audit.py::test_legacy_immutability PASSED

**C4** - `build_stratified_imbalanced_split()` produces exact ratio and positive counts (SPLIT-01, AC 1)
Proof: run unit | tests/test_phase_a_splits.py::test_build_stratified_imbalanced_split_exact_counts PASSED

**C5** - Nested splits for ratios 50, 100, 500 share identical positives and satisfy $N_{50} \subset N_{100} \subset N_{500}$ (SPLIT-02, AC 2)
Proof: run unit | tests/test_phase_a_splits.py::test_nested_negative_subset_containment PASSED

**C6** - Infeasible positive or negative counts raise explicit `ValueError` without silent adjustment (SPLIT-03, AC 3)
Proof: run unit | tests/test_phase_a_splits.py::test_infeasible_counts_raise_value_error PASSED

**C7** - Official CIFAR-100 test split remains un-resampled with natural One-vs-Rest prevalence (SPLIT-04, AC 4)
Proof: run unit | tests/test_phase_a_splits.py::test_official_test_split_unmodified PASSED

**C8** - Periodic training evaluation evaluates strictly on validation features and labels (DATA-01, AC 1)
Proof: run unit | tests/test_phase_a_data_segregation.py::test_periodic_eval_uses_validation_strictly PASSED

**C9** - Missing validation data raises `ValueError` immediately without falling back to test data (DATA-02, AC 2)
Proof: run unit | tests/test_phase_a_data_segregation.py::test_missing_validation_raises_value_error PASSED

**C10** - Tree confusion estimation strictly rejects test data (DATA-03, AC 3)
Proof: run unit | tests/test_phase_a_data_segregation.py::test_tree_calibration_rejects_test_data PASSED

**C11** - `ResolvedTrainingConfig` resolves and validates all training hyperparameters immutably (RUN-01, AC 1)
Proof: run unit | tests/test_phase_a_runner_equivalence.py::test_resolved_training_config_immutability PASSED

**C12** - Runners execute identical pipelines and yield matching metrics under identical configs (RUN-02, AC 2)
Proof: run unit | tests/test_phase_a_runner_equivalence.py::test_runners_deterministic_equivalence PASSED

**C13** - Benchmark results are serialized individually per `run_id` with full metadata (STORE-01, AC 1)
Proof: run unit | tests/test_phase_a_persistence.py::test_atomic_run_id_persistence PASSED

**C14** - Paired statistical tests enforce strict matching of seeds, splits, classes, and budgets (STORE-02, AC 2)
Proof: run unit | tests/test_phase_a_persistence.py::test_paired_statistics_key_validation PASSED

**C15** - Ward clustering is computed over Euclidean row-normalized confusion vectors and Average Linkage over distance matrices (TREE-01, TREE-02, AC 1, 2)
Proof: run unit | tests/test_phase_a_tree_validation.py::test_ward_and_average_linkage_properties PASSED

**C16** - `verify_gate_a()` checks all Phase A invariants and exits 0 on pass (GATE-01, AC 1)
Proof: run unit | tests/test_phase_a_gate.py::test_verify_gate_a_passes PASSED

---

## Handoff

The feature's live state, written as it happens - a compaction gives no warning.

**Settled mid-build:** none
**Dispatched:** none
**Walked:** none

---

## Phase Execution Map

Visual representation of task ordering. Phases run in sequence, and tasks within a phase run in order:

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7

T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8 → T9
```

---

## Task Granularity Check

| Task | Scope | Status |
| ---- | ----- | ------ |
| T1: Create `docs/experimental_audit.md` Catalog | 1 documentation artifact (`docs/experimental_audit.md`) | ✅ Granular |
| T2: Implement Run Manifest Generator and Historical Immutability | 1 utility module (`src/utils/experiment_config.py`) | ✅ Granular |
| T3: Implement Stratified Imbalanced Splitter with Nested Negatives | 1 splitter module (`src/data/splits.py`) | ✅ Granular |
| T4: Segregate Training Evaluation to Validation Data in Trainer | 1 trainer module (`src/training/trainer.py`) | ✅ Granular |
| T5: Prevent Test Data Leakage in Tree Calibration | 1 tree builder module (`src/trees/confusion_tree.py`) | ✅ Granular |
| T6: Implement Unified Resolved Training Config and Runner Equivalence | 1 config module (`src/training/runner_config.py`) | ✅ Granular |
| T7: Implement Fine-Grained Run ID Persistence and Paired Statistics | 2 persistence/stats modules (`src/training/benchmark_runner.py`, `src/utils/statistics.py`) | ✅ Granular |
| T8: Mathematically Validate Ward Clustering over Euclidean Confusion Vectors | 1 tree module (`src/trees/confusion_tree.py`) | ✅ Granular |
| T9: Implement Gate A Verifier and Stop Verification | 1 verifier module (`src/utils/gate_a_verifier.py`) | ✅ Granular |

---

## Diagram-Definition Cross-Check

| Task | Depends On (task body) | Diagram Shows | Status |
| ---- | ---------------------- | ------------- | ------ |
| T1 | None | Phase 1 start (`T1`) | ✅ Match |
| T2 | T1 | `T1 → T2` | ✅ Match |
| T3 | T2 | Phase 1 (`T2`) → Phase 2 (`T3`) | ✅ Match |
| T4 | T3 | Phase 2 (`T3`) → Phase 3 (`T4`) | ✅ Match |
| T5 | T4 | `T4 → T5` | ✅ Match |
| T6 | T5 | Phase 3 (`T5`) → Phase 4 (`T6`) | ✅ Match |
| T7 | T6 | Phase 4 (`T6`) → Phase 5 (`T7`) | ✅ Match |
| T8 | T7 | Phase 5 (`T7`) → Phase 6 (`T8`) | ✅ Match |
| T9 | T8 | Phase 6 (`T8`) → Phase 7 (`T9`) | ✅ Match |

---

## Test Co-location Validation

| Task | Code Layer Created/Modified | Matrix Requires | Task Says | Status |
| ---- | --------------------------- | --------------- | --------- | ------ |
| T1: Create `docs/experimental_audit.md` Catalog | Documentation / Audit | unit | unit | ✅ OK |
| T2: Implement Run Manifest Generator and Historical Immutability | Config / Manifest | unit | unit | ✅ OK |
| T3: Implement Stratified Imbalanced Splitter with Nested Negatives | Domain / Splits | unit | unit | ✅ OK |
| T4: Segregate Training Evaluation to Validation Data in Trainer | Training Loop | unit | unit | ✅ OK |
| T5: Prevent Test Data Leakage in Tree Calibration | Tree Calibration | unit | unit | ✅ OK |
| T6: Implement Unified Resolved Training Config and Runner Equivalence | Runner Governance | unit | unit | ✅ OK |
| T7: Implement Fine-Grained Run ID Persistence and Paired Statistics | Persistence & Stats | unit | unit | ✅ OK |
| T8: Mathematically Validate Ward Clustering over Euclidean Confusion Vectors | Hierarchy Clustering | unit | unit | ✅ OK |
| T9: Implement Gate A Verifier and Stop Verification | Verifier Gate A | integration | integration | ✅ OK |
