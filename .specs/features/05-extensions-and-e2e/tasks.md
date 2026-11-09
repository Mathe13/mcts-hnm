# F5: Dynamic Entropy Leaf Expansion & Tier-2 End-to-End Vision Pipeline Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, the walk, check_proofs, the Verifier and the screens sub-agent).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Design**: `.specs/features/05-extensions-and-e2e/design.md`  
**Status**: In Progress  
**Approved**: 2026-11-09 - user: "prossiga"  
**Build**: `.venv/bin/python -m compileall -q src tests`  
**Run unit**: `.venv/bin/pytest -v`  

---

## Test Coverage Matrix

> Generated from codebase, project guidelines, and spec - confirm before Execute. Guidelines found: `AGENTS.md`, `ROADMAP.md`, `ARCHITECTURE.md`.

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| MCTS / Dynamic Leaf Expansion (`src/mcts/expansion.py`) | unit | Binary & multi-class Shannon prediction entropy $H(p)$, in-place leaf splitting into $K_{\text{sub}}$ quantiles, `B_max` widening, instance count conservation, tie-breaking, and invalid-input `ValueError` guards | `tests/test_dynamic_expansion.py` | `.venv/bin/pytest -v tests/test_dynamic_expansion.py` |
| MCTS / Diagnostics (`src/mcts/diagnostics.py`) | unit | Superclass visit aggregation on dynamically expanded trees via virtual sub-cluster class ID resolution (`sub_cls_id % 1000`) | `tests/test_dynamic_expansion.py` | `.venv/bin/pytest -v tests/test_dynamic_expansion.py` |
| Training / Tier-1 Trainer Expansion Hook (`src/training/trainer.py`) | integration | `train_scenario1()` and `train_scenario2()` with `enable_leaf_expansion=True`, `split_threshold`, and `k_sub` | `tests/test_dynamic_expansion.py` | `.venv/bin/pytest -v tests/test_dynamic_expansion.py` |
| Training / Tier-2 E2E ResNet-18 Pipeline (`src/training/e2e_trainer.py`) | unit / integration | `Tier2ResNet18` forward & feature extraction, `train_e2e_scenario1()`, `train_e2e_scenario2()`, periodic Ward confusion-tree refresh (`refresh_tree_every`), and `run_e2e_experiment()` atomic JSON/CSV persistence | `tests/test_dynamic_expansion.py` | `.venv/bin/pytest -v tests/test_dynamic_expansion.py` |
| Notebook Orchestration (`notebooks/04_tier2_e2e_and_expansion.ipynb`) | integration | End-to-end execution of all cells via `nbformat` / `nbclient` running dynamic leaf expansion ablation and Tier-2 E2E validation without errors | `tests/test_dynamic_expansion.py` | `.venv/bin/pytest -v tests/test_dynamic_expansion.py` |

## Gate Check Commands

> Generated from codebase - confirm before Execute.

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | After tasks with unit tests only | `.venv/bin/pytest -v` |
| Full | After tasks with integration tests | `.venv/bin/pytest -v` |
| Build | After phase completion or config/entity-only tasks | `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v` |

---

## Execution Plan

Phases are ordered and run sequentially - each phase completes before the next begins, and tasks within a phase execute in order.

### Phase 1: Dynamic Entropy Leaf Expansion (`ISS-13`)

Implement per-instance prediction entropy $H(p)$, in-place `FastHierarchyTree` leaf sub-clustering, virtual sub-cluster superclass mapping in diagnostics, and expansion hooks in the Tier-1 trainer.

```
T1 → T2 → T3
```

### Phase 2: Tier-2 End-to-End ResNet-18 Vision Pipeline (`ISS-14`)

Implement `Tier2ResNet18`, periodic Ward confusion-tree refresh, end-to-end Scenario 1 & Scenario 2 training loops, and the multi-seed `run_e2e_experiment()` runner.

```
T4 → T5
```

### Phase 3: Versioned Notebook 04 Orchestration (`ISS-14`)

Create versioned notebook `notebooks/04_tier2_e2e_and_expansion.ipynb` running the P2 dynamic leaf expansion ablation and Tier-2 end-to-end ResNet-18 validation.

```
T6
```

---

## Task Breakdown

### T1: Implement Dynamic Prediction-Entropy Leaf Expansion in `src/mcts/expansion.py`

**What**: Implement `SUBCLUSTER_ID_STRIDE`, `resolve_base_class_id()`, `compute_prediction_entropy()`, `expand_leaf_by_entropy()`, and `maybe_expand_leaf()` in `src/mcts/expansion.py` and unit tests in `tests/test_dynamic_expansion.py`.
**Where**: `src/mcts/expansion.py`
**Depends on**: None
**Reuses**: `src/trees/base_tree.py`
**Requirement**: EXP-01

**Done when**:

- [x] `compute_prediction_entropy(logits, is_binary)` returns finite non-negative 1D `float64` entropies for both binary (`[N]` / `[N, 1]`) and multi-class (`[N, C]`) logits
- [x] `expand_leaf_by_entropy()` and `maybe_expand_leaf()` split an eligible leaf node ($n_l \ge N_{\text{split}}$, `0 <= leaf_to_class_id < 1000`, `instance_counts[l] >= k_sub`) into `k_sub` child leaves sorted by ascending entropy (with deterministic index tie-breaking), widening `tree.children` if `k_sub > B_max` and preserving `tree.validate_tree()`
- [x] Calling `maybe_expand_leaf()` on an already-expanded sub-cluster leaf (`leaf_to_class_id >= 1000`), a leaf with `< split_threshold` visits, or `< k_sub` instances returns `None` without modifying the tree
- [x] Invalid arguments (`k_sub < 2`, `split_threshold < 1`, internal node ID, out-of-range node ID, or mismatched entropy array length) raise `ValueError`
- [x] Gate check passes: `.venv/bin/pytest -v`

**Tests**: unit
**Gate**: quick
**Commit**: `feat(mcts): implement dynamic prediction-entropy leaf expansion`

---

### T2: Support Virtual Sub-Cluster Class IDs in `src/mcts/diagnostics.py`

**What**: Update `compute_superclass_visit_distribution()` in `src/mcts/diagnostics.py` to resolve virtual sub-cluster class IDs (`cls_id % 1000`) back to their canonical CIFAR-100 fine class (`0..99`) and add unit tests in `tests/test_dynamic_expansion.py`.
**Where**: `src/mcts/diagnostics.py`
**Depends on**: T1
**Reuses**: `src/mcts/expansion.py`, `src/data/cifar100_taxonomy.py`
**Requirement**: EXP-01

**Done when**:

- [x] `compute_superclass_visit_distribution()` and `MCTSDiagnosticsTracker.record_step()` / `.summary()` execute without error on dynamically expanded Semantic and Ward Confusion trees
- [x] Visit counts on expanded sub-cluster leaves aggregate into the parent fine class's canonical CIFAR-100 coarse superclass
- [x] Gate check passes: `.venv/bin/pytest -v`

**Tests**: unit
**Gate**: quick
**Commit**: `feat(diagnostics): map expanded sub-cluster leaves to canonical superclasses`

---

### T3: Integrate Dynamic Leaf Expansion into `src/training/trainer.py`

**What**: Add optional `enable_leaf_expansion: bool = False`, `split_threshold: int = 50`, and `k_sub: int = 3` parameters to `train_scenario1()` and `train_scenario2()` in `src/training/trainer.py` (and `BenchmarkConfig` in `src/training/benchmark_runner.py` if needed) so Tier-1 MCTS runs trigger `maybe_expand_leaf()` after `agent.update()`, with integration tests in `tests/test_dynamic_expansion.py`.
**Where**: `src/training/trainer.py`
**Depends on**: T2
**Reuses**: `src/mcts/expansion.py`, `src/mcts/agent.py`, `src/mcts/sampler.py`
**Requirement**: EXP-01

**Done when**:

- [x] `train_scenario1()` and `train_scenario2()` accept `enable_leaf_expansion`, `split_threshold`, and `k_sub`, expanding high-visit leaves dynamically during training and logging `num_expanded_leaves` and `final_num_leaves` in the returned dictionary
- [x] Existing `DualStreamBatchSampler` and `MCTSHNMAgent` instances seamlessly traverse and sample from newly created sub-cluster leaves
- [x] Integration tests pass in `tests/test_dynamic_expansion.py`
- [x] Gate check passes: `.venv/bin/pytest -v`

**Tests**: integration
**Gate**: full
**Commit**: `feat(training): integrate dynamic leaf expansion into Tier-1 trainers`

---

### T4: Implement `Tier2ResNet18` and End-to-End Trainers in `src/training/e2e_trainer.py`

**What**: Implement `Tier2ResNet18`, `evaluate_e2e_scenario1()`, `evaluate_e2e_scenario2()`, `refresh_confusion_tree_from_model()`, `train_e2e_scenario1()`, and `train_e2e_scenario2()` in `src/training/e2e_trainer.py` with unit and integration tests in `tests/test_dynamic_expansion.py`.
**Where**: `src/training/e2e_trainer.py`
**Depends on**: T3
**Reuses**: `src/mcts/agent.py`, `src/mcts/expansion.py`, `src/mcts/sampler.py`, `src/training/baselines.py`, `src/trees/confusion_tree.py`
**Requirement**: EXP-02

**Done when**:

- [ ] `Tier2ResNet18(num_outputs=K, width_multiplier=...)` processes 4D RGB images `[B, 3, 32, 32]` and returns logits `[B, K]` (or `[B]` when `squeeze_binary=True` and `K=1`) and feature vectors via `forward_features()`
- [ ] `train_e2e_scenario1()` and `train_e2e_scenario2()` optimize `Tier2ResNet18` end-to-end with `Adam` across `uniform`, `ohem`, `mcts_strict`, and `mcts_enhanced`, supporting both `enable_leaf_expansion=True` and periodic Ward confusion-tree rebuilds via `refresh_tree_every`
- [ ] Invalid image tensor shapes, unknown strategies, or non-positive step/eval counts raise `ValueError`
- [ ] Unit and integration tests pass in `tests/test_dynamic_expansion.py`
- [ ] Gate check passes: `.venv/bin/pytest -v`

**Tests**: integration
**Gate**: full
**Commit**: `feat(e2e): implement Tier2ResNet18 and end-to-end MCTS trainers`

---

### T5: Implement `E2EExperimentConfig` and `run_e2e_experiment()` in `src/training/e2e_trainer.py`

**What**: Add `E2EExperimentConfig` and `run_e2e_experiment()` to `src/training/e2e_trainer.py` to run multi-seed Tier-2 end-to-end benchmarks, compute paired statistical tests (`compute_paired_statistical_tests`) against `mcts_enhanced`, and atomically persist `.json` and `.csv` summaries to `output_dir`, with tests in `tests/test_dynamic_expansion.py`.
**Where**: `src/training/e2e_trainer.py`
**Depends on**: T4
**Reuses**: `src/utils/statistics.py`
**Requirement**: EXP-02

**Done when**:

- [ ] `run_e2e_experiment()` executes multi-seed runs deterministically across strategies, aggregates `mean ± std`, computes paired Student's $t$-test and Wilcoxon signed-rank statistics against `mcts_enhanced`, and atomically writes `.json` and `.csv` files to `output_dir`
- [ ] Invalid `E2EExperimentConfig` parameters raise `ValueError`
- [ ] Integration tests pass in `tests/test_dynamic_expansion.py`
- [ ] Gate check passes: `.venv/bin/pytest -v`

**Tests**: integration
**Gate**: full
**Commit**: `feat(e2e): add multi-seed run_e2e_experiment runner and atomic exporter`

---

### T6: Create Versioned Notebook 04 in `notebooks/04_tier2_e2e_and_expansion.ipynb`

**What**: Create versioned notebook `notebooks/04_tier2_e2e_and_expansion.ipynb` orchestrating the P2 dynamic entropy leaf expansion ablation and the Tier-2 end-to-end ResNet-18 validation experiment, plus an end-to-end execution test in `tests/test_dynamic_expansion.py`.
**Where**: `notebooks/04_tier2_e2e_and_expansion.ipynb`
**Depends on**: T5
**Reuses**: `src/mcts/expansion.py`, `src/training/trainer.py`, `src/training/e2e_trainer.py`
**Requirement**: EXP-03

**Done when**:

- [ ] `notebooks/04_tier2_e2e_and_expansion.ipynb` is a valid `nbformat` v4 notebook executing the dynamic leaf expansion ablation and Tier-2 end-to-end ResNet-18 experiment via `src.*` APIs, rendering summary plots/tables, and saving JSON/CSV artifacts under `results/`
- [ ] End-to-end notebook execution test in `tests/test_dynamic_expansion.py` passes with zero errors
- [ ] Gate check passes: `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v`

**Tests**: integration
**Gate**: build
**Commit**: `feat(notebooks): create versioned tier-2 e2e and leaf expansion notebook 04`

---

## Checks

**C1** - `compute_prediction_entropy()` returns finite non-negative 1D `float64` Shannon entropies for both binary (`[N]` / `[N, 1]`) and multi-class (`[N, C]`) logits (EXP-01, AC 1)  
Proof: run unit | tests/test_dynamic_expansion.py::test_compute_prediction_entropy_binary_and_multiclass PASSED

**C2** - `expand_leaf_by_entropy()` and `maybe_expand_leaf()` split an eligible leaf with `visits >= split_threshold` into `k_sub` entropy-quantile child leaves while preserving total instance counts and `tree.validate_tree()` (EXP-01, AC 2)  
Proof: run unit | tests/test_dynamic_expansion.py::test_expand_leaf_by_entropy_splits_and_preserves_invariants PASSED

**C3** - `MCTSHNMAgent`, `DualStreamBatchSampler`, and `MCTSDiagnosticsTracker` traverse, sample from, backpropagate through, and aggregate canonical superclass visits on dynamically expanded trees (EXP-01, AC 3)  
Proof: run unit | tests/test_dynamic_expansion.py::test_mcts_agent_sampler_and_diagnostics_on_expanded_tree PASSED

**C4** - `expand_leaf_by_entropy()` and `maybe_expand_leaf()` raise `ValueError` on invalid arguments (`k_sub < 2`, `split_threshold < 1`, internal node ID, out-of-range node ID, or mismatched entropy count) (EXP-01, AC 4)  
Proof: run unit | tests/test_dynamic_expansion.py::test_expansion_invalid_arguments_raise PASSED

**C5** - `maybe_expand_leaf()` safely returns `None` without mutating the tree when called on an already-expanded sub-cluster leaf (`leaf_to_class_id >= 1000`), a leaf with `< split_threshold` visits, or `< k_sub` instances, and handles `k_sub > B_max` and tied entropies deterministically (EXP-01, Edge Cases 1–3)  
Proof: run unit | tests/test_dynamic_expansion.py::test_expansion_edge_cases_noop_bmax_and_ties PASSED

**C6** - `train_scenario1()` and `train_scenario2()` with `enable_leaf_expansion=True` dynamically expand visited leaves during training and record expanded leaf counts (EXP-01, AC 2 & AC 3)  
Proof: run unit | tests/test_dynamic_expansion.py::test_tier1_trainers_with_dynamic_leaf_expansion PASSED

**C7** - `Tier2ResNet18` processes 4D RGB image batches `[B, 3, 32, 32]` and returns logits of shape `[B, K]` (or `[B]` when `squeeze_binary=True` and `K=1`) and feature embeddings via `forward_features()` (EXP-02, AC 1)  
Proof: run unit | tests/test_dynamic_expansion.py::test_tier2_resnet18_forward_and_features PASSED

**C8** - `train_e2e_scenario1()` and `train_e2e_scenario2()` fine-tune `Tier2ResNet18` end-to-end across strategies with dynamic leaf expansion and periodic Ward confusion-tree refresh (`refresh_tree_every`) (EXP-02, AC 2)  
Proof: run unit | tests/test_dynamic_expansion.py::test_train_e2e_scenario1_and_scenario2_with_refresh_and_expansion PASSED

**C9** - `run_e2e_experiment()` executes multi-seed Tier-2 runs deterministically, computes paired statistical tests against `mcts_enhanced`, and atomically writes JSON and CSV summaries to `output_dir` (EXP-02, AC 3)  
Proof: run unit | tests/test_dynamic_expansion.py::test_run_e2e_experiment_multi_seed_and_export PASSED

**C10** - `train_e2e_scenario1()`, `train_e2e_scenario2()`, and `run_e2e_experiment()` raise `ValueError` on invalid image shapes, unknown strategies, or non-positive step/eval intervals (EXP-02, AC 4)  
Proof: run unit | tests/test_dynamic_expansion.py::test_e2e_trainer_invalid_arguments_raise PASSED

**C11** - `notebooks/04_tier2_e2e_and_expansion.ipynb` executes end-to-end without errors and saves dynamic expansion and Tier-2 E2E benchmark artifacts under `results/` (EXP-03, AC 1)  
Proof: run unit | tests/test_dynamic_expansion.py::test_notebook_04_execution PASSED

---

## Handoff

The feature's live state, written as it happens - a compaction gives no warning.

**Settled mid-build:** none  
**Dispatched:** none  
**Walked:** pending  

---

## Phase Execution Map

Visual representation of task ordering. Phases run in sequence, and tasks within a phase run in order:

```
Phase 1 → Phase 2 → Phase 3

T1 → T2 → T3 → T4 → T5 → T6
```

---

## Task Granularity Check

| Task | Scope | Status |
| ---- | ----- | ------ |
| T1: Implement Dynamic Prediction-Entropy Leaf Expansion in `src/mcts/expansion.py` | 1 module (`src/mcts/expansion.py`) | ✅ Granular |
| T2: Support Virtual Sub-Cluster Class IDs in `src/mcts/diagnostics.py` | 1 module (`src/mcts/diagnostics.py`) | ✅ Granular |
| T3: Integrate Dynamic Leaf Expansion into `src/training/trainer.py` | 1 module (`src/training/trainer.py`) | ✅ Granular |
| T4: Implement `Tier2ResNet18` and End-to-End Trainers in `src/training/e2e_trainer.py` | 1 module (`src/training/e2e_trainer.py`) | ✅ Granular |
| T5: Implement `E2EExperimentConfig` and `run_e2e_experiment()` in `src/training/e2e_trainer.py` | 1 module (`src/training/e2e_trainer.py`) | ✅ Granular |
| T6: Create Versioned Notebook 04 in `notebooks/04_tier2_e2e_and_expansion.ipynb` | 1 notebook (`notebooks/04_tier2_e2e_and_expansion.ipynb`) | ✅ Granular |

---

## Diagram-Definition Cross-Check

| Task | Depends On (task body) | Diagram Shows | Status |
| ---- | ---------------------- | ------------- | ------ |
| T1 | None | Phase 1 start (`T1`) | ✅ Match |
| T2 | T1 | `T1 → T2` | ✅ Match |
| T3 | T2 | `T2 → T3` | ✅ Match |
| T4 | T3 | Phase 1 (`T3`) → Phase 2 (`T4`) | ✅ Match |
| T5 | T4 | `T4 → T5` | ✅ Match |
| T6 | T5 | Phase 2 (`T5`) → Phase 3 (`T6`) | ✅ Match |

---

## Test Co-location Validation

| Task | Code Layer Created/Modified | Matrix Requires | Task Says | Status |
| ---- | --------------------------- | --------------- | --------- | ------ |
| T1 | MCTS / Dynamic Leaf Expansion (`src/mcts/expansion.py`) | unit | unit | ✅ OK |
| T2 | MCTS / Diagnostics (`src/mcts/diagnostics.py`) | unit | unit | ✅ OK |
| T3 | Training / Tier-1 Trainer Expansion Hook (`src/training/trainer.py`) | integration | integration | ✅ OK |
| T4 | Training / Tier-2 E2E ResNet-18 Pipeline (`src/training/e2e_trainer.py`) | integration | integration | ✅ OK |
| T5 | Training / Tier-2 E2E ResNet-18 Pipeline (`src/training/e2e_trainer.py`) | integration | integration | ✅ OK |
| T6 | Notebook Orchestration (`notebooks/04_tier2_e2e_and_expansion.ipynb`) | integration | integration | ✅ OK |
