# F20: Fase D — Campanha Científica Final e Artefatos Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, the walk, check_proofs, the Verifier and the screens sub-agent).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Design**: `.specs/features/20-fase-d-campanha-cientifica-e-artefatos/design.md`  
**Status**: In Progress  
**Approved**: 2026-11-11 - user: "prosseguir"  
**Build**: `.venv/bin/python -m compileall -q src tests`  
**Run unit**: `.venv/bin/pytest -v tests/test_phase_d_metrics.py tests/test_phase_d_statistics.py tests/test_phase_d_campaign.py tests/test_phase_d_artifacts.py tests/test_phase_d_gate.py`  

---

## Test Coverage Matrix

> Generated from codebase, project guidelines, and spec - confirm before Execute. Guidelines found: `pyproject.toml`, `.specs/features/20-fase-d-campanha-cientifica-e-artefatos/spec.md`.

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Metrics & Costs (`src/training/trainer.py`) | unit | AUPRC, calibrated F1, computational efficiency costs | `tests/test_phase_d_metrics.py` | `.venv/bin/pytest -v tests/test_phase_d_metrics.py` |
| Paired Statistics (`src/utils/statistics.py`) | unit | Paired t-test, Cohen's d, ratio interaction analysis | `tests/test_phase_d_statistics.py` | `.venv/bin/pytest -v tests/test_phase_d_statistics.py` |
| Campaign Runner (`src/training/campaign_runner.py`) | integration | Frozen matrix orchestration under smoke mode | `tests/test_phase_d_campaign.py` | `.venv/bin/pytest -v tests/test_phase_d_campaign.py` |
| Paper Artifacts (`src/utils/paper_artifacts.py`) | integration | 2x2 ablation plot, exposure heatmap, LaTeX tables | `tests/test_phase_d_artifacts.py` | `.venv/bin/pytest -v tests/test_phase_d_artifacts.py` |
| Gate D Verifier (`src/utils/gate_d_verifier.py`) | integration | Full matrix verification and signed audit manifest | `tests/test_phase_d_gate.py` | `.venv/bin/pytest -v tests/test_phase_d_gate.py` |

---

## Gate Check Commands

> Generated from codebase - confirm before Execute.

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | After tasks with unit tests only | `.venv/bin/pytest -v tests/test_phase_d_metrics.py` |
| Full | After tasks with integration tests | `.venv/bin/pytest -v tests/test_phase_d_*.py` |
| Build | After phase completion or config/entity-only tasks | `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_phase_d_*.py` |

---

## Execution Plan

Phases are ordered and run sequentially - each phase completes before the next begins, and tasks within a phase execute in order.

### Phase 1: Computational Cost Telemetry & Dual Checkpoint Metrics (D2)

Instruments trainer output to record detailed computational costs (updates, gradient sample counts, mining forwards, wall-clock seconds) and dual checkpoint records (`last` and `best_validation`).

```
T1
```

### Phase 2: Inferential Statistics & Interaction Analysis (D3)

Implements paired method differences across identical seeds, Cohen's d effect sizes, confidence intervals, and method $\times$ ratio interaction statistics.

```
T2
```

### Phase 3: Scientific Campaign Runner & Manifest Orchestration (D1)

Implements `run_scientific_campaign()` executing the frozen protocol matrix across Scenarios 1 and 2 with atomic JSON and `.meta.json` serialization.

```
T3
```

### Phase 4: Publication Artifacts, Visualizations & LaTeX Tables (D4)

Enhances artifact generators to produce validation curves with variability bands, exposure distribution heatmaps, 2x2 ablation charts, and consolidated LaTeX tables.

```
T4
```

### Phase 5: Gate D Verifier & Audit Manifest (Gate D)

Implements `verify_gate_d()` to audit complete matrix presence, verify absence of synthetic data in REAL mode, and generate `campaign_audit_manifest.json`.

```
T5
```

---

## Task Breakdown

### T1: Implement Computational Cost Telemetry and Dual Checkpoint Metrics

**What**: Instrument `src/training/trainer.py` to record computational costs (updates, gradient sample count, additional mining forwards, wall-clock time) and preserve both `last` and `best_validation` checkpoint metrics with AUPRC and calibrated F1.  
**Where**: `src/training/trainer.py`  
**Depends on**: none  
**Reuses**: `CheckpointManager`, `src/utils/metrics.py`  
**Requirement**: METR-01  

**Done when**:

- [x] `train_scenario1()` records computational costs (`total_updates`, `gradient_samples`, `mining_forwards`, `elapsed_seconds`)
- [x] `train_scenario2()` records computational costs and maintains `best_validation` and `last` evaluations
- [x] AUPRC and calibrated F1 are reported for both checkpoint evaluations in Scenario 1
- [x] Unit tests in `tests/test_phase_d_metrics.py` pass
- [x] Gate check passes: `.venv/bin/pytest -v tests/test_phase_d_metrics.py`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `feat(metrics): computational cost telemetry and dual checkpoints`  

---

### T2: Implement Inferential Statistics and Method-by-Ratio Interaction

**What**: Implement `compute_paired_method_differences()` and `compute_method_by_ratio_interaction()` in `src/utils/statistics.py`, computing paired differences, Cohen's d, t-tests, and interaction effects across ratios.  
**Where**: `src/utils/statistics.py`  
**Depends on**: T1  
**Reuses**: `scipy.stats`  
**Requirement**: STAT-01  

**Done when**:

- [x] Computes paired differences across identical seeds with mean, std, and 95% CI
- [x] Computes Cohen's d effect size for paired samples
- [x] Computes interaction statistics (difference of differences) between imbalance ratios
- [x] Unit tests in `tests/test_phase_d_statistics.py` pass
- [x] Gate check passes: `.venv/bin/pytest -v tests/test_phase_d_statistics.py`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `feat(statistics): paired comparisons and ratio interaction tests`  

---

### T3: Implement Scientific Campaign Runner and Manifest Orchestration

**What**: Implement `run_scientific_campaign()` in `src/training/campaign_runner.py` orchestrating the frozen protocol matrix across Scenarios 1 and 2 with 5 paired seeds and generating `campaign_manifest.json`.  
**Where**: `src/training/campaign_runner.py`  
**Depends on**: T2  
**Reuses**: `src/training/trainer.py`, `src/data/splits.py`  
**Requirement**: CAMP-01  

**Done when**:

- [x] Executes Scenario 1 across 4 methods and ratios 1:50, 1:100, 1:500 with 5 paired seeds
- [x] Executes Scenario 2 across 6 ablation methods with 5 paired seeds
- [x] Serializes atomic `.json` and `.meta.json` per run with mode and git commit provenance
- [x] Generates consolidated `campaign_manifest.json` indexing all runs
- [x] Integration tests in `tests/test_phase_d_campaign.py` pass
- [x] Gate check passes: `.venv/bin/pytest -v tests/test_phase_d_campaign.py`  

**Tests**: integration  
**Gate**: quick  
**Commit**: `feat(campaign): scientific campaign runner and manifest generator`  

---

### T4: Implement Publication Artifacts, Visualizations and LaTeX Tables

**What**: Enhance `src/utils/paper_artifacts.py` to generate validation curves with variability bands, exposure distribution heatmaps, 2x2 ablation charts, and consolidated LaTeX tables distinguishing checkpoints.  
**Where**: `src/utils/paper_artifacts.py`  
**Depends on**: T3  
**Reuses**: `matplotlib`, `src/training/campaign_runner.py`  
**Requirement**: ARTF-01  

**Done when**:

- [x] Generates validation curve figures with shaded variability bands across paired seeds
- [x] Generates class exposure heatmaps and 2x2 QNorm x DualStream ablation plots
- [x] Generates publication-ready LaTeX tables displaying both `last` and `best_validation` metrics
- [x] Rejects missing or synthetic runs in REAL mode with `FileNotFoundError`
- [x] Integration tests in `tests/test_phase_d_artifacts.py` pass
- [x] Gate check passes: `.venv/bin/pytest -v tests/test_phase_d_artifacts.py`  

**Tests**: integration  
**Gate**: quick  
**Commit**: `feat(artifacts): publication figures and consolidated LaTeX tables`  

---

### T5: Implement Gate D Verifier and Signed Audit Manifest

**What**: Create `src/utils/gate_d_verifier.py` implementing `verify_gate_d()` to audit campaign completeness, verify absence of synthetic data in REAL mode, check finite metrics, and generate `campaign_audit_manifest.json`.  
**Where**: `src/utils/gate_d_verifier.py`  
**Depends on**: T4  
**Reuses**: `src/utils/gate_c_verifier.py`  
**Requirement**: GATE-04  

**Done when**:

- [ ] `verify_gate_d()` audits that all required scenarios, ratios, methods, and paired seeds are present
- [ ] Rejects runs with NaN/Inf metrics or unverified synthetic smoke data in REAL mode
- [ ] Verifies that generated LaTeX tables and figures match empirical JSON outputs
- [ ] Generates signed `campaign_audit_manifest.json` certifying full reproducibility
- [ ] Integration tests in `tests/test_phase_d_gate.py` pass with exit code 0
- [ ] Gate check passes: `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_phase_d_*.py`  

**Tests**: integration  
**Gate**: build  
**Commit**: `feat(verification): implement Gate D scientific audit and manifest verifier`  

---

## Checks

**C1** - Scenario 1 runs executed across 4 methods and ratios 1:50, 1:100, 1:500 with paired seeds (CAMP-01, AC 1)  
Proof: run unit | tests/test_phase_d_campaign.py::test_campaign_runner_scenario1_execution PASSED  

**C2** - Scenario 2 runs executed across 6 ablation methods with paired seeds (CAMP-01, AC 2)  
Proof: run unit | tests/test_phase_d_campaign.py::test_campaign_runner_scenario2_execution PASSED  

**C3** - Atomic JSON and `.meta.json` serialized with git commit and mode provenance (CAMP-01, AC 3)  
Proof: run unit | tests/test_phase_d_campaign.py::test_campaign_runner_provenance_and_manifest PASSED  

**C4** - Abort and record diagnostic error on unhandled exceptions or NaN metrics (CAMP-01, AC 4)  
Proof: run unit | tests/test_phase_d_campaign.py::test_campaign_runner_handles_failure_gracefully PASSED  

**C5** - Scenario 1 records AUPRC, precision, recall, and calibrated F1 at $\tau_{\text{val}}$ (METR-01, AC 1)  
Proof: run unit | tests/test_phase_d_metrics.py::test_scenario1_computes_auprc_and_calibrated_f1 PASSED  

**C6** - Scenario 2 records macro-F1, top-1 accuracy, balanced accuracy, and Worst-10 recall (METR-01, AC 2)  
Proof: run unit | tests/test_phase_d_metrics.py::test_scenario2_computes_macro_f1_and_worst10 PASSED  

**C7** - Computational costs (updates, gradient samples, mining forwards, wall-clock time) recorded (METR-01, AC 3)  
Proof: run unit | tests/test_phase_d_metrics.py::test_computational_cost_telemetry_recording PASSED  

**C8** - Distinct metric records maintained for `last` and `best_validation` checkpoints (METR-01, AC 4)  
Proof: run unit | tests/test_phase_d_metrics.py::test_dual_checkpoint_metrics_distinction PASSED  

**C9** - Paired differences across seeds computed with mean, std, and Cohen's d (STAT-01, AC 1)  
Proof: run unit | tests/test_phase_d_statistics.py::test_paired_method_differences_and_cohens_d PASSED  

**C10** - Method-by-ratio interaction statistics computed preserving seed pairing (STAT-01, AC 2)  
Proof: run unit | tests/test_phase_d_statistics.py::test_method_by_ratio_interaction_statistics PASSED  

**C11** - Statistical summary tables report confidence intervals, sample size $n$, and seed values (STAT-01, AC 3)  
Proof: run unit | tests/test_phase_d_statistics.py::test_statistical_summary_confidence_intervals PASSED  

**C12** - Validation curves with bands, exposure heatmaps, and 2x2 ablation charts generated (ARTF-01, AC 1)  
Proof: run unit | tests/test_phase_d_artifacts.py::test_figure_generation_validation_and_ablation PASSED  

**C13** - Publication LaTeX tables display both checkpoints, AUPRC, and costs without placeholders (ARTF-01, AC 2)  
Proof: run unit | tests/test_phase_d_artifacts.py::test_latex_table_generation_with_checkpoints_and_auprc PASSED  

**C14** - Reject missing results or synthetic smoke artifacts during REAL generation (ARTF-01, AC 3)  
Proof: run unit | tests/test_phase_d_artifacts.py::test_artifacts_reject_synthetic_or_missing_real_data PASSED  

**C15** - `verify_gate_d()` audits presence of all planned scenarios, ratios, methods, and seeds (GATE-04, AC 1)  
Proof: run unit | tests/test_phase_d_gate.py::test_verify_gate_d_passes_complete_campaign PASSED  

**C16** - Reject missing runs, unrecorded metrics, or synthetic data in REAL mode (GATE-04, AC 2)  
Proof: run unit | tests/test_phase_d_gate.py::test_verify_gate_d_rejects_incomplete_or_synthetic_campaign PASSED  

**C17** - Generate signed `campaign_audit_manifest.json` certifying scientific reproducibility (GATE-04, AC 3)  
Proof: run unit | tests/test_phase_d_gate.py::test_verify_gate_d_generates_signed_audit_manifest PASSED  

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
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5

T1 → T2 → T3 → T4 → T5
```

---

## Task Granularity Check

| Task | Scope | Status |
| ---- | ----- | ------ |
| T1 | `src/training/trainer.py` | pending |
| T2 | `src/utils/statistics.py` | pending |
| T3 | `src/training/campaign_runner.py` | pending |
| T4 | `src/utils/paper_artifacts.py` | pending |
| T5 | `src/utils/gate_d_verifier.py` | pending |
