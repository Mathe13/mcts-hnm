# F4: Downstream Classifiers, Baselines & Tier-1 Benchmark Notebook Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, the walk, check_proofs, the Verifier and the screens sub-agent).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Design**: `.specs/features/04-tier1-benchmark-suite/design.md`  
**Status**: Ready  
**Approved**: 2026-11-09 - user: "prossiga"  
**Build**: `.venv/bin/python -m compileall -q src tests`  
**Run unit**: `.venv/bin/pytest -v`  

---

## Test Coverage Matrix

> Generated from codebase, project guidelines, and spec - confirm before Execute. Guidelines found: `AGENTS.md`, `ROADMAP.md`, `ARCHITECTURE.md`.

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Model / Downstream MLP (`src/models/mlp_head.py`) | unit | All shapes, binary vs multi-class forward passes, dropout, gradient flow, and parameter validation | `tests/test_training_loop.py` | `.venv/bin/pytest -v tests/test_training_loop.py` |
| Training / Baselines (`src/training/baselines.py`) | unit | `UniformSampler` and `OHEMSampler` batch assembly, top-loss selection, candidate ratio clamping, and parameter validation | `tests/test_training_loop.py` | `.venv/bin/pytest -v tests/test_training_loop.py` |
| Training / Trainer & Metrics (`src/training/trainer.py`) | unit / integration | Scenario 1 (`BCEWithLogitsLoss`, F1, AUPRC, FPR@95) & Scenario 2 (`CrossEntropyLoss`, Top-1, Macro-F1, Worst-10) training loops across 4 strategies (`uniform`, `ohem`, `mcts_strict`, `mcts_enhanced`) on synthetic data | `tests/test_training_loop.py` | `.venv/bin/pytest -v tests/test_training_loop.py` |
| Utils / Statistics & Runner (`src/utils/statistics.py`, `src/training/benchmark_runner.py`) | unit / integration | Paired Student's t-test, Wilcoxon signed-rank test, zero-difference safe fallback, multi-seed aggregation, and atomic JSON/CSV result saving | `tests/test_training_loop.py` | `.venv/bin/pytest -v tests/test_training_loop.py` |
| Notebook Orchestration (`notebooks/03_tier1_mcts_benchmarks.ipynb`) | integration | Full cell execution via `nbformat` / `nbclient` executing end-to-end multi-seed benchmark flow without errors | `tests/test_training_loop.py` | `.venv/bin/pytest -v tests/test_training_loop.py` |

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

### Phase 1: Models & Non-Tree Baseline Samplers (`ISS-10`)

Implement the Tier-1 2-layer MLP classification head and the `UniformSampler` and `OHEMSampler` classes.

```
T1 → T2
```

### Phase 2: Scenario 1 & 2 Unified Trainer & Metrics (`ISS-11`)

Implement the Scenario 1 and Scenario 2 training loops, metric evaluators (F1, AUPRC, FPR@95, Top-1, Macro-F1, Worst-10-Class Recall), and MCTS reward backpropagation.

```
T3 → T4
```

### Phase 3: Statistical Tests, Benchmark Runner & Notebook 03 (`ISS-12`)

Implement paired statistical significance tests (`ttest_rel`, `wilcoxon`), the programmatic multi-seed benchmark runner with atomic JSON/CSV export, and the versioned orchestration notebook `notebooks/03_tier1_mcts_benchmarks.ipynb`.

```
T5 → T6 → T7
```

---

## Task Breakdown

### T1: Implement Tier-1 MLP Head in `src/models/mlp_head.py`

**What**: Implement `Tier1MLPHead` and `create_mlp_optimizer()` in `src/models/mlp_head.py` (with exports in `src/models/__init__.py`) and unit tests in `tests/test_training_loop.py`.
**Where**: `src/models/mlp_head.py`
**Depends on**: None
**Reuses**: `torch.nn`
**Requirement**: BENCH-01

**Done when**:

- [x] `Tier1MLPHead(input_dim=512, hidden_dim=128, num_outputs=K, dropout=0.1)` outputs tensor of shape `[B, K]` (or `[B]` when `squeeze_binary=True` and `num_outputs=1`) and supports backpropagation
- [x] `create_mlp_optimizer(model, lr=1e-3)` returns an `Adam` optimizer configured with the model parameters
- [x] `Tier1MLPHead` raises `ValueError` on non-positive dimensions, `dropout < 0`, or `dropout >= 1`
- [x] Unit tests pass in `tests/test_training_loop.py`
- [x] Gate check passes: `.venv/bin/pytest -v`

**Tests**: unit
**Gate**: quick
**Commit**: `feat(models): implement Tier1MLPHead and Adam optimizer factory`

---

### T2: Implement `UniformSampler` and `OHEMSampler` in `src/training/baselines.py`

**What**: Implement `UniformSampler` and `OHEMSampler` in `src/training/baselines.py` supporting both Scenario 1 and Scenario 2 sampling pools, plus unit tests in `tests/test_training_loop.py`.
**Where**: `src/training/baselines.py`
**Depends on**: T1
**Reuses**: `src/utils/seed.py`
**Requirement**: BENCH-01

**Done when**:

- [x] `UniformSampler.sample_batch()` samples `pos_batch_size` positives and `neg_batch_size` negatives uniformly at random (or `neg_batch_size` across all classes for Scenario 2)
- [x] `OHEMSampler.sample_batch()` draws $M_{\text{cand}} = \min(|\text{pool}|, \, r_{\text{ohem}} \cdot B_{\text{neg}})$ candidate negative samples, computes per-sample loss with `torch.no_grad()`, and returns the top-$B_{\text{neg}}$ highest loss negative indices
- [x] `UniformSampler` and `OHEMSampler` raise `ValueError` on invalid batch sizes or `ohem_ratio < 1`
- [x] Unit tests verify that `OHEMSampler` selects the highest loss samples deterministically on synthetic features
- [x] Gate check passes: `.venv/bin/pytest -v`

**Tests**: unit
**Gate**: quick
**Commit**: `feat(training): implement UniformSampler and OHEMSampler baselines`

---

### T3: Implement Scenario 1 & 2 Evaluators in `src/training/trainer.py`

**What**: Implement `evaluate_scenario1_metrics()` (Loss, F1, Precision, Recall, AUPRC, FPR@95) and `evaluate_scenario2_metrics()` (Loss, Top-1 Accuracy, Macro-F1, Worst-10-Class Recall) in `src/training/trainer.py` with unit tests in `tests/test_training_loop.py`.
**Where**: `src/training/trainer.py`
**Depends on**: T2
**Reuses**: `sklearn.metrics`, `torch`
**Requirement**: BENCH-02

**Done when**:

- [x] `evaluate_scenario1_metrics()` returns finite metrics (`loss`, `f1`, `precision`, `recall`, `auprc`, `fpr_at_95_recall`) using `zero_division=0.0`
- [x] `evaluate_scenario2_metrics()` returns finite metrics (`loss`, `top1_accuracy`, `macro_f1`, `worst_10_class_recall`) using `zero_division=0.0`
- [x] Unit tests pass verifying metric bounds and exact values on synthetic predictions
- [x] Gate check passes: `.venv/bin/pytest -v`

**Tests**: unit
**Gate**: quick
**Commit**: `feat(training): add metrics for Scenario 1 and Scenario 2`

---

### T4: Implement `train_scenario1()` and `train_scenario2()` in `src/training/trainer.py`

**What**: Implement the training loops `train_scenario1()` and `train_scenario2()` supporting `uniform`, `ohem`, `mcts_strict`, and `mcts_enhanced` strategies, coupled with `agent.update()` and `MCTSDiagnosticsTracker` for MCTS runs, plus unit tests in `tests/test_training_loop.py`.
**Where**: `src/training/trainer.py`
**Depends on**: T3
**Reuses**: `src/models/mlp_head.py`, `src/mcts/agent.py`, `src/mcts/sampler.py`, `src/mcts/diagnostics.py`, `src/training/baselines.py`
**Requirement**: BENCH-02

**Done when**:

- [x] `train_scenario1()` optimizes `Tier1MLPHead(num_outputs=1)` with `BCEWithLogitsLoss()`, logging evaluation metrics and tree visitation entropy at `eval_every` steps
- [x] `train_scenario2()` optimizes `Tier1MLPHead(num_outputs=num_classes)` with `CrossEntropyLoss(label_smoothing=0.1)`, logging evaluation metrics and tree visitation entropy at `eval_every` steps
- [x] MCTS runs update leaf $Q$-values and visit counts using the MCTS leaf sub-batch reward and shift visits toward high-loss classes
- [x] Invalid parameters or unknown strategy names raise `ValueError`
- [x] Unit tests pass in `tests/test_training_loop.py`
- [x] Gate check passes: `.venv/bin/pytest -v`

**Tests**: integration
**Gate**: quick
**Commit**: `feat(training): implement train_scenario1 and train_scenario2 loops`

---

### T5: Implement Paired Statistical Tests in `src/utils/statistics.py`

**What**: Implement `compute_paired_statistical_tests()` in `src/utils/statistics.py` returning paired Student's t-test and Wilcoxon signed-rank statistics with zero-difference fallback, plus unit tests in `tests/test_training_loop.py`.
**Where**: `src/utils/statistics.py`
**Depends on**: T4
**Reuses**: `scipy.stats`
**Requirement**: BENCH-03

**Done when**:

- [ ] `compute_paired_statistical_tests(scores_a, scores_b)` returns `{"mean_a", "std_a", "mean_b", "std_b", "mean_diff", "t_stat", "t_pvalue", "wilcoxon_stat", "wilcoxon_pvalue"}` for $N \ge 2$ paired scores
- [ ] When all paired differences are zero, returns `t_stat=0.0`, `t_pvalue=1.0`, `wilcoxon_stat=0.0`, `wilcoxon_pvalue=1.0` without raising an exception
- [ ] Raises `ValueError` on unequal array lengths, length $< 2$, or non-finite values
- [ ] Unit tests pass in `tests/test_training_loop.py`
- [ ] Gate check passes: `.venv/bin/pytest -v`

**Tests**: unit
**Gate**: quick
**Commit**: `feat(utils): implement compute_paired_statistical_tests with zero-difference guard`

---

### T6: Implement Multi-Seed Benchmark Runner in `src/training/benchmark_runner.py`

**What**: Implement `BenchmarkConfig` and `run_benchmark()` in `src/training/benchmark_runner.py` executing multi-seed experiment grids, computing paired statistics against `mcts_enhanced`, and atomically exporting JSON and CSV results, plus tests in `tests/test_training_loop.py`.
**Where**: `src/training/benchmark_runner.py`
**Depends on**: T5
**Reuses**: `src/training/trainer.py`, `src/utils/statistics.py`
**Requirement**: BENCH-03

**Done when**:

- [ ] `run_benchmark()` executes multi-seed runs deterministically, aggregates `mean ± std`, runs paired statistical tests against `mcts_enhanced`, and saves atomic `.json` and `.csv` files
- [ ] Raises `ValueError` on invalid configuration options
- [ ] Unit/integration tests verify multi-seed execution and atomic file saving on a synthetic fixture
- [ ] Gate check passes: `.venv/bin/pytest -v`

**Tests**: integration
**Gate**: quick
**Commit**: `feat(benchmarks): implement multi-seed benchmark runner and atomic exporter`

---

### T7: Create Versioned Orchestration Notebook in `notebooks/03_tier1_mcts_benchmarks.ipynb`

**What**: Create versioned notebook `notebooks/03_tier1_mcts_benchmarks.ipynb` executing the 3 experimental axes (Sampler Comparison, Semantic vs. Ward Confusion Tree Comparison, and $c$/$\alpha$ Ablations) and add end-to-end execution test in `tests/test_training_loop.py`.
**Where**: `notebooks/03_tier1_mcts_benchmarks.ipynb`
**Depends on**: T6
**Reuses**: `src/training/benchmark_runner.py`, `src/trees/*`
**Requirement**: BENCH-03

**Done when**:

- [ ] `notebooks/03_tier1_mcts_benchmarks.ipynb` is a valid `nbformat` v4 notebook executing all 3 benchmark axes via high-level `src.*` APIs and rendering summary figures/tables
- [ ] End-to-end integration test executes the notebook cells without errors
- [ ] Gate check passes: `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v`

**Tests**: integration
**Gate**: full
**Commit**: `feat(notebooks): create versioned tier-1 mcts benchmark notebook 03`

---

## Checks

**C1** - `Tier1MLPHead` returns logits of shape `[B, K]` (or `[B]` when `squeeze_binary=True` and `num_outputs=1`) and `create_mlp_optimizer()` creates an `Adam` optimizer (BENCH-01, AC 1)  
Proof: run unit | tests/test_training_loop.py::test_tier1_mlp_head_forward_and_optimizer PASSED

**C2** - `UniformSampler` samples `pos_batch_size` positives and `neg_batch_size` negatives uniformly at random across valid index ranges (BENCH-01, AC 2)  
Proof: run unit | tests/test_training_loop.py::test_uniform_sampler_batch_generation PASSED

**C3** - `OHEMSampler` samples candidate negative instances and selects the top-$B_{\text{neg}}$ samples with the highest losses under `torch.no_grad()` (BENCH-01, AC 3)  
Proof: run unit | tests/test_training_loop.py::test_ohem_sampler_selects_highest_loss PASSED

**C4** - `Tier1MLPHead`, `UniformSampler`, and `OHEMSampler` raise `ValueError` on invalid initialization parameters (BENCH-01, AC 4)  
Proof: run unit | tests/test_training_loop.py::test_model_and_sampler_invalid_args_raise PASSED

**C5** - `train_scenario1()` optimizes `Tier1MLPHead` with `BCEWithLogitsLoss()` and records `f1`, `precision`, `recall`, `auprc`, `fpr_at_95_recall`, and visitation entropy at evaluation steps (BENCH-02, AC 1)  
Proof: run unit | tests/test_training_loop.py::test_train_scenario1_metrics_and_entropy PASSED

**C6** - `train_scenario2()` optimizes `Tier1MLPHead` with `CrossEntropyLoss(label_smoothing=0.1)` and records `top1_accuracy`, `macro_f1`, `worst_10_class_recall`, and visitation entropy at evaluation steps (BENCH-02, AC 2)  
Proof: run unit | tests/test_training_loop.py::test_train_scenario2_metrics_and_entropy PASSED

**C7** - In MCTS strategies, leaf $Q$-values and visit counts are updated using only the MCTS leaf sub-batch reward and concentrate on harder classes (BENCH-02, AC 3)  
Proof: run unit | tests/test_training_loop.py::test_mcts_updates_and_shifts_leaf_visits PASSED

**C8** - `train_scenario1()` and `train_scenario2()` raise `ValueError` on unknown strategies, mismatched data, or invalid step counts (BENCH-02, AC 4)  
Proof: run unit | tests/test_training_loop.py::test_trainer_invalid_arguments_raise PASSED

**C9** - `compute_paired_statistical_tests()` returns mean, std, paired difference, paired Student's t-test, and Wilcoxon signed-rank statistics across paired multi-seed scores (BENCH-03, AC 1)  
Proof: run unit | tests/test_training_loop.py::test_compute_paired_statistical_tests PASSED

**C10** - When all paired score differences are identically zero, `compute_paired_statistical_tests()` safely returns `p_value = 1.0` and `statistic = 0.0` without error (BENCH-03, Edge Case 1)  
Proof: run unit | tests/test_training_loop.py::test_compute_paired_statistical_tests_zero_diff_fallback PASSED

**C11** - `run_benchmark()` executes multi-seed benchmarks deterministically, computes comparisons, and atomically saves JSON and CSV results to disk (BENCH-03, AC 2)  
Proof: run unit | tests/test_training_loop.py::test_run_benchmark_multi_seed_and_export PASSED

**C12** - `notebooks/03_tier1_mcts_benchmarks.ipynb` runs end-to-end without errors and executes the benchmark suite (BENCH-03, AC 3)  
Proof: run unit | tests/test_training_loop.py::test_notebook_03_execution PASSED

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
Phase 1 → Phase 2 → Phase 3

T1 → T2 → T3 → T4 → T5 → T6 → T7
```

---

## Task Granularity Check

| Task | Scope | Status |
| ---- | ----- | ------ |
| T1: Implement Tier-1 MLP Head in `src/models/mlp_head.py` | 1 model module (`src/models/mlp_head.py`) | ✅ Granular |
| T2: Implement `UniformSampler` and `OHEMSampler` in `src/training/baselines.py` | 1 training baseline module (`src/training/baselines.py`) | ✅ Granular |
| T3: Implement Scenario 1 & 2 Evaluators in `src/training/trainer.py` | 1 evaluator module (`src/training/trainer.py`) | ✅ Granular |
| T4: Implement `train_scenario1()` and `train_scenario2()` in `src/training/trainer.py` | 1 training loop module (`src/training/trainer.py`) | ✅ Granular |
| T5: Implement Paired Statistical Tests in `src/utils/statistics.py` | 1 statistics module (`src/utils/statistics.py`) | ✅ Granular |
| T6: Implement Multi-Seed Benchmark Runner in `src/training/benchmark_runner.py` | 1 runner module (`src/training/benchmark_runner.py`) | ✅ Granular |
| T7: Create Versioned Orchestration Notebook in `notebooks/03_tier1_mcts_benchmarks.ipynb` | 1 notebook (`notebooks/03_tier1_mcts_benchmarks.ipynb`) | ✅ Granular |

---

## Diagram-Definition Cross-Check

| Task | Depends On (task body) | Diagram Shows | Status |
| ---- | ---------------------- | ------------- | ------ |
| T1 | None | Phase 1 start (`T1`) | ✅ Match |
| T2 | T1 | `T1 → T2` | ✅ Match |
| T3 | T2 | Phase 1 (`T2`) → Phase 2 (`T3`) | ✅ Match |
| T4 | T3 | `T3 → T4` | ✅ Match |
| T5 | T4 | Phase 2 (`T4`) → Phase 3 (`T5`) | ✅ Match |
| T6 | T5 | `T5 → T6` | ✅ Match |
| T7 | T6 | `T6 → T7` | ✅ Match |

---

## Test Co-location Validation

| Task | Code Layer Created/Modified | Matrix Requires | Task Says | Status |
| ---- | --------------------------- | --------------- | --------- | ------ |
| T1 | Model / Downstream MLP (`src/models/mlp_head.py`) | unit | unit | ✅ OK |
| T2 | Training / Baselines (`src/training/baselines.py`) | unit | unit | ✅ OK |
| T3 | Training / Trainer & Metrics (`src/training/trainer.py`) | unit | unit | ✅ OK |
| T4 | Training / Trainer & Metrics (`src/training/trainer.py`) | integration | integration | ✅ OK |
| T5 | Utils / Statistics (`src/utils/statistics.py`) | unit | unit | ✅ OK |
| T6 | Training / Benchmark Runner (`src/training/benchmark_runner.py`) | integration | integration | ✅ OK |
| T7 | Notebook Orchestration (`notebooks/03_tier1_mcts_benchmarks.ipynb`) | integration | integration | ✅ OK |
