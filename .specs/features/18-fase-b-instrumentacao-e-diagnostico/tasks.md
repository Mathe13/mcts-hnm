# F18: Fase B — Instrumentação, Observabilidade e Piloto Diagnóstico Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, the walk, check_proofs, the Verifier and the screens sub-agent).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Design**: `.specs/features/18-fase-b-instrumentacao-e-diagnostico/design.md`  
**Status**: In Progress  
**Approved**: 2026-11-11 - user: "sim"  
**Build**: `.venv/bin/python -m compileall -q src tests`  
**Run unit**: `.venv/bin/pytest -v tests/test_phase_b_observability.py tests/test_phase_b_mcts_diagnostics.py tests/test_phase_b_checkpoints.py tests/test_phase_b_gate.py`  

---

## Test Coverage Matrix

> Generated from codebase, project guidelines, and spec - confirm before Execute. Guidelines found: `pyproject.toml`, `.specs/features/18-fase-b-instrumentacao-e-diagnostico/spec.md`.

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Exposure & Observability (`src/training/exposure.py`) | unit | Full branch coverage; zero & non-zero entropy; Worst-10 recall | `tests/test_phase_b_observability.py` | `.venv/bin/pytest -v tests/test_phase_b_observability.py` |
| Training Loop Logging (`src/training/trainer.py`) | unit/integration | Step history completeness across Scenarios 1 & 2 | `tests/test_phase_b_observability.py` | `.venv/bin/pytest -v tests/test_phase_b_observability.py` |
| MCTS Policy Diagnostics (`src/mcts/mcts_agent.py`) | unit | State snapshot schema; exploration positivity ($U > 0$); micro-tree backpropagation | `tests/test_phase_b_mcts_diagnostics.py` | `.venv/bin/pytest -v tests/test_phase_b_mcts_diagnostics.py` |
| Checkpoints Management (`src/training/checkpoints.py`) | unit | Deterministic roundtrip; model, optimizer, tree, and RNG resumability | `tests/test_phase_b_checkpoints.py` | `.venv/bin/pytest -v tests/test_phase_b_checkpoints.py` |
| Threshold Calibration (`src/utils/metrics.py`) | unit | Threshold calibration grid over validation; test F1 and AUPRC reporting | `tests/test_phase_b_checkpoints.py` | `.venv/bin/pytest -v tests/test_phase_b_checkpoints.py` |
| Pilot Runner (`src/training/pilot_runner.py`) | integration | Multi-scenario pilot execution under smoke mode | `tests/test_phase_b_gate.py` | `.venv/bin/pytest -v tests/test_phase_b_gate.py` |
| Verifier Gate B (`src/utils/gate_b_verifier.py`) | integration | Gate B viability rules & failure cases for 1:500 | `tests/test_phase_b_gate.py` | `.venv/bin/pytest -v tests/test_phase_b_gate.py` |

---

## Gate Check Commands

> Generated from codebase - confirm before Execute.

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | After tasks with unit tests only | `.venv/bin/pytest -v tests/test_phase_b_observability.py` |
| Full | After tasks with integration tests | `.venv/bin/pytest -v tests/test_phase_b_*.py` |
| Build | After phase completion or config/entity-only tasks | `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_phase_b_*.py` |

---

## Execution Plan

Phases are ordered and run sequentially - each phase completes before the next begins, and tasks within a phase execute in order.

### Phase 1: Exposure Tracking & Sampling Diagnostics (B1)

Implements `ExposureTracker` tracking minibatch class distributions, node visits, unique instance counts, normalized exposure entropy $H_{\text{norm}}$, and Worst-10 recall.

```
T1
```

### Phase 2: Uniform Observability & Step History Logging (B1)

Instruments `train_scenario1()` and `train_scenario2()` to record comprehensive `step_history` dictionaries and attach exposure diagnostics to the run record.

```
T2
```

### Phase 3: MCTS Policy State Inspection & Starvation Audit (B2)

Adds `get_mcts_policy_diagnostics()` to `src/mcts/mcts_agent.py` and validates starvation invariants and backpropagation on micro-hierarchies.

```
T3
```

### Phase 4: Checkpoints & Threshold Calibration (B3)

Implements `CheckpointManager` persisting `last` and `best_validation` checkpoints, and `calibrate_binary_threshold()` for out-of-test threshold optimization in Scenario 1.

```
T4 → T5
```

### Phase 5: Diagnostic Pilot Runner & Gate B Verification (B4, Gate B)

Implements `run_diagnostic_pilot()` running Scenarios 1 and 2 in pilot mode, and `verify_gate_b()` conducting automated viability audits for 1:500.

```
T6 → T7
```

---

## Task Breakdown

### T1: Implement ExposureTracker and Normalized Entropy Diagnostics

**What**: Create `src/training/exposure.py` implementing `ExposureTracker` to record batch sampling events, compute normalized exposure entropy $H_{\text{norm}} \in [0, 1]$, count unexposed classes $C_{\text{unexp}}$, and compute Worst-10 recall.  
**Where**: `src/training/exposure.py`  
**Depends on**: none  
**Reuses**: `FastHierarchyTree`  
**Requirement**: OBS-02  

**Done when**:

- [x] `ExposureTracker` records leaf selections, instance IDs, and class IDs per batch
- [x] `compute_diagnostics()` computes $N_v$, $N_l$, $N_s$, $N_u$, $H_{\text{norm}}$, $C_{\text{unexp}}$, and Worst-10 class recall
- [x] Returns $0.0$ for $H_{\text{norm}}$ when total selections are zero
- [x] Unit tests in `tests/test_phase_b_observability.py` pass
- [x] Gate check passes: `.venv/bin/pytest -v tests/test_phase_b_observability.py`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `feat(exposure): implement ExposureTracker and normalized entropy diagnostics`  

---

### T2: Integrate Step History and Batch Exposure Logging into Trainer

**What**: Update `train_scenario1()` and `train_scenario2()` in `src/training/trainer.py` to record `step_history` with loss, metrics, learning rate, wall time, and class exposure diagnostics in the return dictionary.  
**Where**: `src/training/trainer.py`  
**Depends on**: T1  
**Reuses**: `src/training/exposure.py`  
**Requirement**: OBS-01  

**Done when**:

- [x] `train_scenario1()` logs `train_loss`, `val_loss`, `val_F1`, `val_AUPRC`, `learning_rate`, `step`, and time in `step_history`
- [x] `train_scenario2()` logs `macro_f1` and top-1 `accuracy` in `step_history`
- [x] Training return dictionary contains complete `step_history` and `exposure_diagnostics`
- [x] Integration tests in `tests/test_phase_b_observability.py` pass
- [x] Gate check passes: `.venv/bin/pytest -v tests/test_phase_b_observability.py`  

**Tests**: integration  
**Gate**: quick  
**Commit**: `feat(trainer): integrate step history and batch exposure logging`  

---

### T3: Implement MCTS Policy Diagnostics and Micro-Tree Validation

**What**: Add `get_mcts_policy_diagnostics()` in `src/mcts/mcts_agent.py` to extract per-node Q-values, rewards, exploration term $U$, UCB scores, and counters, and verify starvation and propagation invariants on synthetic trees.  
**Where**: `src/mcts/mcts_agent.py`  
**Depends on**: T2  
**Reuses**: `src/mcts/mcts_agent.py`, `src/trees/base_tree.py`  
**Requirement**: MCTS-01, MCTS-02  

**Done when**:

- [x] `get_mcts_policy_diagnostics()` extracts per-node reward, Q, U, UCB, and visit metrics
- [x] Traversal guarantees $U > 0$ for unvisited/low-visit siblings when parent visits $> 0$ and $c_{\text{puct}} > 0$
- [x] Traversal on synthetic 3-class tree visits all leaves without premature starvation
- [x] Backpropagation updates visits and Q-values along full path to root
- [x] Unit tests in `tests/test_phase_b_mcts_diagnostics.py` pass
- [x] Gate check passes: `.venv/bin/pytest -v tests/test_phase_b_mcts_diagnostics.py`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `feat(mcts): implement policy state diagnostics and micro-tree validation`  

---

### T4: Implement CheckpointManager for Last and Best-Validation States

**What**: Create `src/training/checkpoints.py` implementing `CheckpointManager` to persist and restore `last` and `best_validation` checkpoints containing model, optimizer, tree, and RNG states.  
**Where**: `src/training/checkpoints.py`  
**Depends on**: T3  
**Reuses**: `torch.save`, `torch.load`  
**Requirement**: CKPT-01  

**Done when**:

- [ ] `CheckpointManager.save_checkpoint()` saves `last.pt` and `best_validation.pt`
- [ ] Checkpoint dictionary stores model state dict, optimizer state dict, tree state, step, metrics, and RNG states
- [ ] `CheckpointManager.load_checkpoint()` restores state accurately and deterministically resumes training
- [ ] Unit tests in `tests/test_phase_b_checkpoints.py` pass
- [ ] Gate check passes: `.venv/bin/pytest -v tests/test_phase_b_checkpoints.py`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `feat(checkpoints): implement CheckpointManager for last and best-validation`  

---

### T5: Implement Validation Threshold Calibration and Reporting

**What**: Implement `calibrate_binary_threshold()` and evaluation utilities in `src/utils/metrics.py`, integrating out-of-test threshold tuning into Scenario 1.  
**Where**: `src/utils/metrics.py`  
**Depends on**: T4  
**Reuses**: `sklearn.metrics.f1_score`, `src/training/trainer.py`  
**Requirement**: CKPT-02  

**Done when**:

- [ ] `calibrate_binary_threshold(val_probs, val_targets)` finds $\tau_{\text{val}} \in [0.01, 0.99]$ maximizing validation F1
- [ ] Scenario 1 test evaluation reports default F1 ($\tau=0.5$), calibrated F1 ($\tau=\tau_{\text{val}}$), and AUPRC
- [ ] Unit tests in `tests/test_phase_b_checkpoints.py` pass
- [ ] Gate check passes: `.venv/bin/pytest -v tests/test_phase_b_checkpoints.py`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `feat(metrics): implement validation threshold calibration and reporting`  

---

### T6: Implement Diagnostic Pilot Runner for Scenarios 1 and 2

**What**: Create `src/training/pilot_runner.py` implementing `run_diagnostic_pilot()` to execute short diagnostic runs (2 seeds, $\le 500$ steps) covering Scenario 1 (50, 100, 500) and Scenario 2 (Uniform, Uniform Grouped, Enhanced).  
**Where**: `src/training/pilot_runner.py`  
**Depends on**: T5  
**Reuses**: `src/training/trainer.py`, `src/data/splits.py`, `src/training/runner_config.py`  
**Requirement**: PILOT-01  

**Done when**:

- [ ] `run_diagnostic_pilot()` runs Scenarios 1 and 2 under configurable small update budgets
- [ ] Outputs are isolated to `results/pilot/` or `results/smoke/` without modifying legacy files
- [ ] Serializes per-run JSON records with step histories and diagnostics
- [ ] Integration tests in `tests/test_phase_b_gate.py` pass
- [ ] Gate check passes: `.venv/bin/pytest -v tests/test_phase_b_gate.py`  

**Tests**: integration  
**Gate**: quick  
**Commit**: `feat(pilot): implement diagnostic pilot runner for Scenarios 1 and 2`  

---

### T7: Implement Gate B and 1:500 Viability Verifier

**What**: Create `src/utils/gate_b_verifier.py` implementing `verify_gate_b()` to audit pilot run records, verifying 1:500 viability, finite metrics, and complete diagnostics.  
**Where**: `src/utils/gate_b_verifier.py`  
**Depends on**: T6  
**Reuses**: `src/utils/gate_b_verifier.py`  
**Requirement**: GATE-02  

**Done when**:

- [ ] `verify_gate_b()` verifies complete `step_history` and `exposure_diagnostics` across all pilot records
- [ ] `verify_gate_b()` validates no NaN/Inf in losses, gradients, or metrics
- [ ] `verify_gate_b()` validates exact 80 positives and 40,000 negatives without replacement for 1:500
- [ ] `verify_gate_b()` returns 1 when any record violates invariants
- [ ] Integration test `tests/test_phase_b_gate.py::test_verify_gate_b_passes` passes with exit code 0
- [ ] Gate check passes: `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_phase_b_*.py`  

**Tests**: integration  
**Gate**: build  
**Commit**: `feat(verification): implement Gate B and 1:500 viability verifier`  

---

## Checks

**C1** - `compute_exposure_diagnostics()` returns 0.0 entropy for zero total selections (OBS-02, AC 4)  
Proof: run unit | tests/test_phase_b_observability.py::test_exposure_entropy_zero_selections PASSED  

**C2** - `compute_exposure_diagnostics()` computes normalized entropy in $[0, 1]$ and Worst-10 recall on non-zero selections (OBS-02, AC 3, 5)  
Proof: run unit | tests/test_phase_b_observability.py::test_exposure_entropy_and_worst10_calculation PASSED  

**C3** - `train_scenario1()` logs complete `step_history` with loss, metrics, learning rate, and elapsed time (OBS-01, AC 1, 3)  
Proof: run unit | tests/test_phase_b_observability.py::test_scenario1_step_history_logging PASSED  

**C4** - `train_scenario2()` logs `macro_f1` and `accuracy` in `step_history` (OBS-01, AC 2)  
Proof: run unit | tests/test_phase_b_observability.py::test_scenario2_step_history_logging PASSED  

**C5** - `get_mcts_policy_diagnostics()` returns per-node Q-values, rewards, exploration term U, UCB scores, and visit counts (MCTS-01, AC 1)  
Proof: run unit | tests/test_phase_b_mcts_diagnostics.py::test_mcts_policy_diagnostics_structure PASSED  

**C6** - Tree traversal maintains $U > 0$ for unvisited/low-visit siblings when $N_{\text{parent}} > 0$ and $c_{\text{puct}} > 0$ (MCTS-01, AC 2)  
Proof: run unit | tests/test_phase_b_mcts_diagnostics.py::test_mcts_exploration_positivity PASSED  

**C7** - MCTS visits all leaves on synthetic micro-tree without premature starvation (MCTS-02, AC 3)  
Proof: run unit | tests/test_phase_b_mcts_diagnostics.py::test_micro_tree_no_starvation PASSED  

**C8** - Backpropagation updates visit counts and Q values along full ancestor path to root (MCTS-02, AC 3)  
Proof: run unit | tests/test_phase_b_mcts_diagnostics.py::test_micro_tree_backpropagation_to_root PASSED  

**C9** - `CheckpointManager` saves and restores `last` and `best_validation` checkpoints deterministically (CKPT-01, AC 1)  
Proof: run unit | tests/test_phase_b_checkpoints.py::test_checkpoint_save_and_restore_roundtrip PASSED  

**C10** - `calibrate_binary_threshold()` selects $\tau_{\text{val}}$ maximizing validation F1 (CKPT-02, AC 2)  
Proof: run unit | tests/test_phase_b_checkpoints.py::test_binary_threshold_calibration_optimizes_f1 PASSED  

**C11** - Scenario 1 evaluation reports default F1 ($\tau=0.5$), calibrated F1 ($\tau=\tau_{\text{val}}$), and AUPRC (CKPT-02, AC 3)  
Proof: run unit | tests/test_phase_b_checkpoints.py::test_scenario1_reports_calibrated_f1_and_auprc PASSED  

**C12** - `run_diagnostic_pilot()` runs 2 seeds for Scenario 1 (50, 100, 500) and Scenario 2 in smoke mode (PILOT-01, AC 1)  
Proof: run unit | tests/test_phase_b_gate.py::test_pilot_runner_smoke_execution PASSED  

**C13** - Pilot writes isolated records to target output directory without polluting legacy (PILOT-01, AC 1)  
Proof: run unit | tests/test_phase_b_gate.py::test_pilot_output_isolation PASSED  

**C14** - `verify_gate_b()` validates complete diagnostics, finite metrics, and 1:500 viability (GATE-02, AC 2)  
Proof: run unit | tests/test_phase_b_gate.py::test_verify_gate_b_passes PASSED  

**C15** - `verify_gate_b()` returns 1 when any run record contains NaNs or violates split invariants (GATE-02, AC 3)  
Proof: run unit | tests/test_phase_b_gate.py::test_verify_gate_b_rejects_nan_and_invariants PASSED  

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

T1 → T2 → T3 → T4 → T5 → T6 → T7
```

---

## Task Granularity Check

| Task | Scope | Status |
| ---- | ----- | ------ |
| T1 | `src/training/exposure.py` | pending |
| T2 | `src/training/trainer.py` | pending |
| T3 | `src/mcts/mcts_agent.py` | pending |
| T4 | `src/training/checkpoints.py` | pending |
| T5 | `src/utils/metrics.py` | pending |
| T6 | `src/training/pilot_runner.py` | pending |
| T7 | `src/utils/gate_b_verifier.py` | pending |
