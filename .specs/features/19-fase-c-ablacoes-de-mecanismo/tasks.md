# F19: Fase C — Ablações de Mecanismo com Variantes Imutáveis Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, the walk, check_proofs, the Verifier and the screens sub-agent).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Design**: `.specs/features/19-fase-c-ablacoes-de-mecanismo/design.md`  
**Status**: Complete  
**Approved**: 2026-11-11 - user: "prossiga"  
**Build**: `.venv/bin/python -m compileall -q src tests`  
**Run unit**: `.venv/bin/pytest -v tests/test_phase_c_ablations.py tests/test_phase_c_selectors.py tests/test_phase_c_hierarchies.py tests/test_phase_c_gate.py`  

---

## Test Coverage Matrix

> Generated from codebase, project guidelines, and spec - confirm before Execute. Guidelines found: `pyproject.toml`, `.specs/features/19-fase-c-ablacoes-de-mecanismo/spec.md`.

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Batch Diversity (`src/training/batch_diversity.py`) | unit | Shannon entropy of batch labels; unique class counts | `tests/test_phase_c_ablations.py` | `.venv/bin/pytest -v tests/test_phase_c_ablations.py` |
| Uniform Grouped Sampler (`src/mcts/sampler.py`) | unit | Focal 50/50 composition; parameter validation | `tests/test_phase_c_ablations.py` | `.venv/bin/pytest -v tests/test_phase_c_ablations.py` |
| Factorial 2x2 Trainer (`src/training/trainer.py`) | unit/integration | 4 orthogonal variants; invariant parameter control | `tests/test_phase_c_ablations.py` | `.venv/bin/pytest -v tests/test_phase_c_ablations.py` |
| Simple Selectors (`src/training/selectors.py`) | unit | Greedy Loss EMA & Flat UCB zero-visit priority | `tests/test_phase_c_selectors.py` | `.venv/bin/pytest -v tests/test_phase_c_selectors.py` |
| Structural Metrics (`src/trees/hierarchy_metrics.py`) | unit | Max/mean depth; branching factor; node counts | `tests/test_phase_c_hierarchies.py` | `.venv/bin/pytest -v tests/test_phase_c_hierarchies.py` |
| Random & UPGMA Trees (`src/trees/random_tree.py`) | unit | Semantic topology matching; UPGMA dissimilarity | `tests/test_phase_c_hierarchies.py` | `.venv/bin/pytest -v tests/test_phase_c_hierarchies.py` |
| Gate C Verifier (`src/utils/gate_c_verifier.py`) | integration | Strict isolation check; causal classification | `tests/test_phase_c_gate.py` | `.venv/bin/pytest -v tests/test_phase_c_gate.py` |

---

## Gate Check Commands

> Generated from codebase - confirm before Execute.

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | After tasks with unit tests only | `.venv/bin/pytest -v tests/test_phase_c_ablations.py` |
| Full | After tasks with integration tests | `.venv/bin/pytest -v tests/test_phase_c_*.py` |
| Build | After phase completion or config/entity-only tasks | `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_phase_c_*.py` |

---

## Execution Plan

Phases are ordered and run sequentially - each phase completes before the next begins, and tasks within a phase execute in order.

### Phase 1: Batch Diversity and Uniform Grouped Control (C1)

Implements `compute_batch_diversity()` and `UniformGroupedBatchSampler` to measure batch diversity and isolate minibatch homogeneity effects in Scenario 2.

```
T1 → T2
```

### Phase 2: Factorial 2x2 Grid (Q-normalization x Dual-Stream) (C2)

Implements the 4 orthogonal variants (`mcts_strict`, `mcts_qnorm_only`, `mcts_dualstream_only`, `mcts_enhanced`) in `trainer.py` and `runner_config.py` with invariant non-ablated parameters.

```
T3
```

### Phase 3: Simple Selection Controls (C3)

Implements `GreedyLossEMASelector` and `FlatUCBSelector` to benchmark tree search against greedy selection and a flat multi-armed bandit.

```
T4
```

### Phase 4: Structural Topology of Hierarchies and Random Baseline (C5)

Implements tree topological metrics (`max_depth`, `mean_leaf_depth`, `mean_branching_factor`) and a balanced Random Tree baseline matching CIFAR-100 superclass topology.

```
T5 → T6
```

### Phase 5: Gate C Verifier (Variable Isolation) (Gate C)

Implements `verify_gate_c()` to audit strict parameter isolation across paired ablation runs, check telemetry completeness, and classify performance degradation causes.

```
T7
```

---

## Task Breakdown

### T1: Implement Batch Diversity Telemetry and Exposure Integration

**What**: Create `src/training/batch_diversity.py` implementing `compute_batch_diversity()` to calculate unique classes $N_{\text{unique}}$ and Shannon entropy $H_{\text{batch}}$ from minibatch labels. Integrate metrics into `ExposureTracker` and `step_history`.  
**Where**: `src/training/batch_diversity.py`  
**Depends on**: none  
**Reuses**: `ExposureTracker`  
**Requirement**: ABLA-01  

**Done when**:

- [x] `compute_batch_diversity()` returns `unique_classes` and `batch_entropy` for arbitrary label arrays
- [x] Returns 0.0 entropy when batch has 1 class or is empty
- [x] Integrates diversity tracking into `ExposureTracker.record_batch()`
- [x] Unit tests in `tests/test_phase_c_ablations.py` pass
- [x] Gate check passes: `.venv/bin/pytest -v tests/test_phase_c_ablations.py`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `feat(telemetry): implement batch diversity and entropy calculation`  

---

### T2: Implement Uniform Grouped Sampler and Strategy Integration

**What**: Implement `UniformGroupedBatchSampler` in `src/mcts/sampler.py` sampling $B/2$ instances from a uniform random focal class and $B/2$ from the full training pool. Add strategy `"uniform_grouped"` to `SUPPORTED_STRATEGIES` in `src/training/trainer.py`.  
**Where**: `src/mcts/sampler.py`  
**Depends on**: T1  
**Reuses**: `DualStreamBatch`, `src/training/trainer.py`  
**Requirement**: ABLA-01  

**Done when**:

- [x] `UniformGroupedBatchSampler` draws $B/2$ focal instances from a random class and $B/2$ uniform instances
- [x] Raises `ValueError` for invalid `dual_stream_ratio` outside $[0.0, 1.0]$
- [x] `train_scenario2()` executes with `strategy="uniform_grouped"`
- [x] Unit tests in `tests/test_phase_c_ablations.py` pass
- [x] Gate check passes: `.venv/bin/pytest -v tests/test_phase_c_ablations.py`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `feat(samplers): implement UniformGroupedBatchSampler strategy`  

---

### T3: Implement 2x2 Factorial Grid (Q-normalization x Dual-Stream)

**What**: Support `mcts_strict`, `mcts_qnorm_only`, `mcts_dualstream_only`, and `mcts_enhanced` in `trainer.py` and `runner_config.py`, enforcing invariant non-ablated parameters (weights, LR, momentum, exploration constant, step budget).  
**Where**: `src/training/trainer.py`  
**Depends on**: T2  
**Reuses**: `MCTSHNMAgent`, `DualStreamBatchSampler`  
**Requirement**: ABLA-02  

**Done when**:

- [x] `mcts_strict` sets `q_norm=False` and `dual_stream_ratio=0.0`
- [x] `mcts_qnorm_only` sets `q_norm=True` and `dual_stream_ratio=0.0`
- [x] `mcts_dualstream_only` sets `q_norm=False` and `dual_stream_ratio=0.5`
- [x] `mcts_enhanced` sets `q_norm=True` and `dual_stream_ratio=0.5`
- [x] Non-ablated factors are held strictly invariant for identical seeds
- [x] Unit tests in `tests/test_phase_c_ablations.py` pass
- [x] Gate check passes: `.venv/bin/pytest -v tests/test_phase_c_ablations.py`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `feat(trainer): implement 2x2 factorial grid of Q-norm and Dual-Stream`  

---

### T4: Implement Greedy Loss EMA and Flat UCB Baseline Selectors

**What**: Create `src/training/selectors.py` implementing `GreedyLossEMASelector` (greedy selection on class loss EMA, $c=0$) and `FlatUCBSelector` (flat 100-arm multi-armed bandit with unvisited priority). Integrate `"greedy_loss_ema"` and `"flat_ucb"` into `src/training/trainer.py`.  
**Where**: `src/training/selectors.py`  
**Depends on**: T3  
**Reuses**: `src/training/trainer.py`  
**Requirement**: ABLA-03  

**Done when**:

- [x] `GreedyLossEMASelector` updates loss EMAs and greedily selects the class with highest loss
- [x] `FlatUCBSelector` assigns infinite priority to unvisited classes and applies UCB1 over flat classes
- [x] Both selectors update statistics dynamically from minibatch loss
- [x] Strategies `"greedy_loss_ema"` and `"flat_ucb"` train downstream models in Scenario 2
- [x] Unit tests in `tests/test_phase_c_selectors.py` pass
- [x] Gate check passes: `.venv/bin/pytest -v tests/test_phase_c_selectors.py`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `feat(selectors): implement GreedyLossEMA and FlatUCB baseline selectors`  

---

### T5: Implement Tree Structural Topology Metrics

**What**: Create `src/trees/hierarchy_metrics.py` implementing `compute_tree_structural_metrics()` to calculate `max_depth`, `mean_leaf_depth`, `total_nodes`, `num_leaves`, and `mean_branching_factor`.  
**Where**: `src/trees/hierarchy_metrics.py`  
**Depends on**: T4  
**Reuses**: `FastHierarchyTree`  
**Requirement**: ABLA-05  

**Done when**:

- [x] `compute_tree_structural_metrics()` returns complete structural dictionary for any `FastHierarchyTree`
- [x] Correctly computes maximum depth, mean leaf depth, total nodes, and branching factor
- [x] Validates input is a valid `FastHierarchyTree` instance
- [x] Unit tests in `tests/test_phase_c_hierarchies.py` pass
- [x] Gate check passes: `.venv/bin/pytest -v tests/test_phase_c_hierarchies.py`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `feat(trees): implement tree structural metrics computation`  

---

### T6: Implement Balanced Random Hierarchy Baseline and UPGMA Validation

**What**: Implement `build_random_tree()` in `src/trees/random_tree.py` constructing a balanced 20-superclass random tree matching the CIFAR-100 semantic topology. Verify Average Linkage (UPGMA) with non-negative confusion dissimilarities in `confusion_tree.py`.  
**Where**: `src/trees/random_tree.py`  
**Depends on**: T5  
**Reuses**: `FastHierarchyTree`, `build_confusion_tree`  
**Requirement**: ABLA-05  

**Done when**:

- [x] `build_random_tree()` constructs a valid `FastHierarchyTree` with 100 leaves and 20 groups of 5
- [x] Structural invariants (root=0, contiguous IDs, parent-child links) hold
- [x] Average Linkage (UPGMA) produces valid hierarchy from confusion dissimilarity
- [x] Unit tests in `tests/test_phase_c_hierarchies.py` pass
- [x] Gate check passes: `.venv/bin/pytest -v tests/test_phase_c_hierarchies.py`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `feat(trees): implement random balanced hierarchy and UPGMA`  

---

### T7: Implement Gate C Variable Isolation Verifier and Causal Classifier

**What**: Create `src/utils/gate_c_verifier.py` implementing `verify_gate_c()` to audit pairwise parameter invariance, finite metrics, telemetry completeness, and produce a causal diagnostic classification between batch homogeneity and policy starvation.  
**Where**: `src/utils/gate_c_verifier.py`  
**Depends on**: T6  
**Reuses**: `src/utils/gate_b_verifier.py`  
**Requirement**: GATE-03    

**Done when**:

- [x] `verify_gate_c()` verifies that paired ablation runs differ only in targeted factors
- [x] Rejects runs with NaN/Inf or missing batch diversity telemetry
- [x] Evaluates Uniform vs Uniform Grouped vs Enhanced and outputs diagnostic cause classification
- [x] Integration tests in `tests/test_phase_c_gate.py` pass with exit code 0
- [x] Gate check passes: `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_phase_c_*.py`  

**Tests**: integration  
**Gate**: build  
**Commit**: `feat(verification): implement Gate C variable isolation verifier`  

---

## Checks

**C1** - `uniform_grouped` samples B/2 focal and B/2 uniform instances (ABLA-01, AC 1)  
Proof: run unit | tests/test_phase_c_ablations.py::test_uniform_grouped_batch_composition PASSED  

**C2** - Batch diversity metrics $N_{\text{unique}}$ and $H_{\text{batch}}$ computed (ABLA-01, AC 2)  
Proof: run unit | tests/test_phase_c_ablations.py::test_batch_diversity_calculation PASSED  

**C3** - Mean batch diversity and cumulative coverage recorded in summary (ABLA-01, AC 3)  
Proof: run unit | tests/test_phase_c_ablations.py::test_batch_diversity_integration_with_exposure PASSED  

**C4** - Reject invalid `dual_stream_ratio` in `uniform_grouped` (ABLA-01, AC 4)  
Proof: run unit | tests/test_phase_c_ablations.py::test_uniform_grouped_parameter_validation PASSED  

**C5** - `mcts_strict` sets `q_norm=False` and `dual_stream_ratio=0.0` (ABLA-02, AC 1)  
Proof: run unit | tests/test_phase_c_ablations.py::test_factorial_variants_configuration PASSED  

**C6** - `mcts_qnorm_only` sets `q_norm=True` and `dual_stream_ratio=0.0` (ABLA-02, AC 2)  
Proof: run unit | tests/test_phase_c_ablations.py::test_factorial_variants_configuration PASSED  

**C7** - `mcts_dualstream_only` sets `q_norm=False` and `dual_stream_ratio=0.5` (ABLA-02, AC 3)  
Proof: run unit | tests/test_phase_c_ablations.py::test_factorial_variants_configuration PASSED  

**C8** - `mcts_enhanced` sets `q_norm=True` and `dual_stream_ratio=0.5` (ABLA-02, AC 4)  
Proof: run unit | tests/test_phase_c_ablations.py::test_factorial_variants_configuration PASSED  

**C9** - Factorial variants hold non-ablated parameters strictly invariant (ABLA-02, AC 5)  
Proof: run unit | tests/test_phase_c_ablations.py::test_factorial_controlled_factors_invariance PASSED  

**C10** - `greedy_loss_ema` greedily selects highest EMA loss with $c=0$ (ABLA-03, AC 1)  
Proof: run unit | tests/test_phase_c_selectors.py::test_greedy_loss_ema_selector_behavior PASSED  

**C11** - `flat_ucb` maintains counts and Q-values without tree hierarchy (ABLA-03, AC 2)  
Proof: run unit | tests/test_phase_c_selectors.py::test_flat_ucb_unvisited_priority_and_selection PASSED  

**C12** - Class statistics updated at each step from batch loss (ABLA-03, AC 3)  
Proof: run unit | tests/test_phase_c_selectors.py::test_greedy_loss_ema_selector_behavior PASSED  

**C13** - Zero-visit classes in `flat_ucb` assigned infinite priority (ABLA-03, AC 4)  
Proof: run unit | tests/test_phase_c_selectors.py::test_flat_ucb_unvisited_priority_and_selection PASSED  

**C14** - `compute_tree_structural_metrics()` returns depth, node, branching stats (ABLA-05, AC 1)  
Proof: run unit | tests/test_phase_c_hierarchies.py::test_tree_structural_metrics_calculation PASSED  

**C15** - Random hierarchy preserves 100 leaves and semantic depth profile (ABLA-05, AC 2)  
Proof: run unit | tests/test_phase_c_hierarchies.py::test_random_tree_construction_and_invariants PASSED  

**C16** - Average Linkage (UPGMA) produces strictly valid hierarchy (ABLA-05, AC 3)  
Proof: run unit | tests/test_phase_c_hierarchies.py::test_upgma_average_linkage_tree_validity PASSED  

**C17** - `verify_gate_c()` audits strict parameter invariance in paired runs (GATE-03, AC 1)  
Proof: run unit | tests/test_phase_c_gate.py::test_verify_gate_c_passes_valid_ablations PASSED  

**C18** - Reject non-finite metrics or missing batch diversity telemetry (GATE-03, AC 2)  
Proof: run unit | tests/test_phase_c_gate.py::test_verify_gate_c_rejects_divergent_parameters PASSED  

**C19** - Diagnostic classification of batch homogeneity vs policy starvation (GATE-03, AC 3)  
Proof: run unit | tests/test_phase_c_gate.py::test_verify_gate_c_passes_valid_ablations PASSED  

---

## Handoff

The feature's live state, written as it happens - a compaction gives no warning.

**Settled mid-build:** none  
**Dispatched:** independent verifier sub-agent 57bd5593-57ec-4701-9b1a-6e0e088e3a4c  
**Walked:** verified all T1-T7 tasks and proofs passed (20/20)  

---

## Phase Execution Map

Visual representation of task ordering. Phases run in sequence, and tasks within a phase run in order:

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5

T1 → T2 → T3 → T4 → T5 → T6 → T7
```

---

## Task Granularity Check

| Task | Scope | Status | Commit |
| ---- | ----- | ------ | ------ |
| T1 | `src/training/batch_diversity.py` | complete | `6a0d574` |
| T2 | `src/mcts/sampler.py` | complete | `039c8fc` |
| T3 | `src/training/trainer.py` | complete | `ef837d5` |
| T4 | `src/training/selectors.py` | complete | `c050f96` |
| T5 | `src/trees/hierarchy_metrics.py` | complete | `ecffb78` |
| T6 | `src/trees/random_tree.py` | complete | `d465f0a` |
| T7 | `src/utils/gate_c_verifier.py` | complete | `ca4aeb4` |
