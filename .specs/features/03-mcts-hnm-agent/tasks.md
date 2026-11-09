# F3: Core MCTS Hard Negative Mining Agent Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, the walk, check_proofs, the Verifier and the screens sub-agent).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Design**: `.specs/features/03-mcts-hnm-agent/design.md`
**Status**: In Progress
**Approved**: 2026-11-08 - user: "prossiga"
**Build**: `.venv/bin/python -m compileall -q src tests`
**Run unit**: `.venv/bin/pytest -v`

---

## Test Coverage Matrix

> Generated from codebase, project guidelines, and spec - confirm before Execute. Guidelines found: `AGENTS.md`, `ROADMAP.md`, `ARCHITECTURE.md`.

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Domain / MCTS Selection & Q-Normalization (`src/mcts/agent.py`) | unit | All branches; 1:1 to `MCTS-01` ACs 1–4; Strict (`q_norm=False`) vs. Enhanced (`q_norm=True`), equal-$Q$ zero-division guard, deterministic unvisited-child priority, and invalid parameter validation | `tests/test_mcts_agent.py` | `.venv/bin/pytest -v tests/test_mcts_agent.py` |
| Domain / MCTS Backpropagation & Reward (`src/mcts/agent.py`) | unit | All branches; 1:1 to `MCTS-02` ACs 1–4; leaf momentum ($\alpha=0.3$), bottom-up $M_c / M_p$ weighted ancestor updates on Semantic and Ward trees, binary/multi-class reward computation, and `ValueError` on invalid node/reward | `tests/test_mcts_agent.py` | `.venv/bin/pytest -v tests/test_mcts_agent.py` |
| Domain / Dual-Stream Batch Sampler (`src/mcts/sampler.py`) | unit | All branches; 1:1 to `MCTS-03` ACs 1–2; exact $K_{\text{mcts}} + K_{\text{uni}} == B_{\text{neg}}$ across $\beta \in [0, 1]$, positive sampling, replacement fallback with single warning on small leaves, and parameter validation | `tests/test_mcts_agent.py` | `.venv/bin/pytest -v tests/test_mcts_agent.py` |
| Domain / RL Visitation Diagnostics (`src/mcts/diagnostics.py`) | unit | All branches; 1:1 to `MCTS-03` ACs 3–4; Shannon visitation entropy $H_t$ and $H_t / \ln K$, zero-visit safety, 20-superclass visit distribution across Semantic and Ward trees, and `MCTSDiagnosticsTracker` | `tests/test_mcts_agent.py` | `.venv/bin/pytest -v tests/test_mcts_agent.py` |

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

### Phase 1: MCTS Selection, Q-Normalization & Momentum Backpropagation (`ISS-07`, `ISS-08`)

Tasks that implement the core `MCTSHNMAgent` tree traversal, Min-Max sibling $Q$-normalization, reward calculation, and bottom-up instance-weighted backpropagation.

```
T1 → T2
```

### Phase 2: Dual-Stream Batch Sampler & RL Visitation Diagnostics (`ISS-09`)

Tasks that implement hybrid MCTS + uniform negative batch sampling and Tree Visitation Entropy / superclass visitation trackers.

```
T3 → T4
```

---

## Task Breakdown

### T1: Implement UCB1 Selection & Min-Max Q-Normalization in `src/mcts/agent.py`

**What**: Implement `normalize_sibling_q_values()`, `compute_ucb1_scores()`, and `MCTSHNMAgent` (`__init__`, `select_child`, `select_leaf`, `reset`) in `src/mcts/agent.py` (with package exports in `src/mcts/__init__.py`) and unit tests in `tests/test_mcts_agent.py`.
**Where**: `src/mcts/agent.py`
**Depends on**: None
**Reuses**: `src/trees/base_tree.py` (`FastHierarchyTree`), `src/trees/semantic_tree.py`, `src/trees/confusion_tree.py`
**Requirement**: MCTS-01

**Done when**:

- [x] `normalize_sibling_q_values(q_values, eps=1e-5)` normalizes sibling $Q$-values into `[0, 1]` and returns `0.0` for all siblings when $Q_{\max} == Q_{\min}$ without `NaN`
- [x] `compute_ucb1_scores()` and `MCTSHNMAgent.select_leaf()` compute $U_i = \tilde{Q}_i + c \sqrt{\frac{\ln \max(N_s, 1)}{n_i + \epsilon}}$ for both `q_norm=False` (Strict GiantHunter) and `q_norm=True` (Enhanced MCTS-HNM) and deterministically prioritize unvisited children ($n_i = 0$) before visited siblings
- [x] `MCTSHNMAgent` raises `ValueError` on invalid hyperparameters (`exploration_c < 0`, `momentum_alpha <= 0` or `> 1`, `eps <= 0`)
- [x] Unit tests `test_ucb1_selection_strict_q_norm_false`, `test_ucb1_selection_enhanced_q_norm_true_and_equal_q_zero`, `test_select_leaf_prioritizes_unvisited_children`, and `test_agent_invalid_hyperparameters_raise` pass in `tests/test_mcts_agent.py`
- [x] Gate check passes: `.venv/bin/pytest -v`
- [x] Test count: 33 tests pass (29 from F1+F2 + 4 new in `tests/test_mcts_agent.py`)

**Tests**: unit
**Gate**: quick
**Commit**: `feat(mcts): implement UCB1 selection and Min-Max Q-normalization`

---

### T2: Implement Momentum Backpropagation, Hierarchical Aggregation & Reward Calculation

**What**: Implement `MCTSHNMAgent.update(leaf_node_id, reward)` and `compute_mcts_reward(logits, is_binary, target_labels)` in `src/mcts/agent.py`, plus unit tests in `tests/test_mcts_agent.py`.
**Where**: `src/mcts/agent.py` (modify)
**Depends on**: T1
**Reuses**: `src/trees/base_tree.py` (`FastHierarchyTree`)
**Requirement**: MCTS-02

**Done when**:

- [ ] `MCTSHNMAgent.update(leaf_node_id, reward)` increments `tree.visits[l]` by `1`, updates leaf $Q_l^{(t)} = \alpha r + (1 - \alpha) Q_l^{(t-1)}$, and updates every ancestor $p$ up to `root_id` via `tree.visits[p] += 1` and $Q_p = \frac{1}{M_p} \sum_{c \in \text{Children}(p)} M_c Q_c$
- [ ] `compute_mcts_reward()` computes finite mean binary cross-entropy loss on negative logits (`is_binary=True`) and multi-class cross-entropy loss (`is_binary=False`)
- [ ] `MCTSHNMAgent.update()` raises `ValueError` on out-of-range node IDs, internal nodes (`is_leaf == False`), or `NaN`/`Inf` rewards without mutating tree state
- [ ] Unit tests `test_backpropagation_leaf_momentum_update`, `test_backpropagation_weighted_ancestor_aggregation`, `test_compute_mcts_reward_binary_and_multiclass`, and `test_backpropagation_invalid_inputs_raise` pass in `tests/test_mcts_agent.py`
- [ ] Gate check passes: `.venv/bin/pytest -v`
- [ ] Test count: 37 tests pass (no silent deletions)

**Tests**: unit
**Gate**: quick
**Commit**: `feat(mcts): add momentum backpropagation and instance-weighted aggregation`

---

### T3: Implement `DualStreamBatchSampler` in `src/mcts/sampler.py`

**What**: Implement `DualStreamBatch` and `DualStreamBatchSampler` in `src/mcts/sampler.py` combining $\lfloor (1 - \beta) B_{\text{neg}} \rfloor$ MCTS-selected leaf instances with $\lceil \beta B_{\text{neg}} \rceil$ uniform negative anchor instances (and optional positive instances), plus unit tests in `tests/test_mcts_agent.py`.
**Where**: `src/mcts/sampler.py`
**Depends on**: T2
**Reuses**: `src/trees/base_tree.py` (`FastHierarchyTree`), `src/utils/seed.py`
**Requirement**: MCTS-03

**Done when**:

- [ ] `DualStreamBatchSampler` assembles negative batches with exact sizes $K_{\text{mcts}} = \lfloor (1 - \beta) B_{\text{neg}} \rfloor$ and $K_{\text{uni}} = \lceil \beta B_{\text{neg}} \rceil$ ($K_{\text{mcts}} + K_{\text{uni}} == B_{\text{neg}}$) across $\beta \in [0.0, 1.0]$ and optional positive samples
- [ ] When a leaf has fewer instances than $K_{\text{mcts}}$, `sample_batch()` samples with replacement and emits a warning once per sampler instance; invalid sampler arguments raise `ValueError`
- [ ] Unit tests `test_dual_stream_batch_sampler_ratios_and_positives` and `test_dual_stream_sampler_small_leaf_replacement_and_validation` pass in `tests/test_mcts_agent.py`
- [ ] Gate check passes: `.venv/bin/pytest -v`
- [ ] Test count: 39 tests pass (no silent deletions)

**Tests**: unit
**Gate**: quick
**Commit**: `feat(mcts): implement DualStreamBatchSampler with replacement fallback`

---

### T4: Implement Tree Visitation Entropy & Superclass Diagnostics in `src/mcts/diagnostics.py`

**What**: Implement `compute_visitation_entropy()`, `compute_leaf_visit_distribution()`, `compute_superclass_visit_distribution()`, `MCTSStepDiagnostic`, and `MCTSDiagnosticsTracker` in `src/mcts/diagnostics.py` with unit tests in `tests/test_mcts_agent.py`.
**Where**: `src/mcts/diagnostics.py`
**Depends on**: T3
**Reuses**: `src/data/cifar100_taxonomy.py` (`get_cifar100_taxonomy`), `src/trees/base_tree.py` (`FastHierarchyTree`)
**Requirement**: MCTS-03

**Done when**:

- [ ] `compute_visitation_entropy(tree)` computes Shannon entropy $H_t = -\sum_{n_i > 0} p_i \ln p_i$ and normalized entropy $H_t / \ln K \in [0.0, 1.0]$, returning `(0.0, 0.0)` when total leaf visits equal `0`
- [ ] `compute_superclass_visit_distribution(tree)` maps leaf visits to all 20 CIFAR-100 coarse superclass names on both Semantic and Ward Confusion trees, and `MCTSDiagnosticsTracker` logs step-level RL diagnostics
- [ ] Unit tests `test_visitation_entropy_zero_single_and_uniform` and `test_superclass_visit_distribution_and_tracker` pass in `tests/test_mcts_agent.py`
- [ ] Gate check passes: `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v`
- [ ] Test count: 41 tests pass (no silent deletions)

**Tests**: unit
**Gate**: quick
**Commit**: `feat(mcts): add tree visitation entropy and superclass diagnostics`

---

## Checks

**C1** - `MCTSHNMAgent` with `q_norm=False` (Strict GiantHunter mode) selects the child maximizing $U_i = Q_i + c \sqrt{\frac{\ln N_s}{n_i + \epsilon}}$ when all children at internal node $s$ have been visited (MCTS-01, AC 1)
Proof: run unit | tests/test_mcts_agent.py::test_ucb1_selection_strict_q_norm_false PASSED

**C2** - `MCTSHNMAgent` with `q_norm=True` (Enhanced MCTS-HNM mode) normalizes sibling $Q$-values into `[0, 1]` via $\bar{Q}_i^{\text{norm}} = \frac{Q_i - Q_{\min}}{Q_{\max} - Q_{\min} + \epsilon}$ and sets $\bar{Q}_i^{\text{norm}} = 0.0$ when $Q_{\max} == Q_{\min}$ without `NaN` (MCTS-01, AC 2)
Proof: run unit | tests/test_mcts_agent.py::test_ucb1_selection_enhanced_q_norm_true_and_equal_q_zero PASSED

**C3** - `MCTSHNMAgent.select_leaf()` deterministically prioritizes unvisited children ($n_i = 0$) before selecting any already-visited sibling ($n_j \ge 1$) (MCTS-01, AC 3)
Proof: run unit | tests/test_mcts_agent.py::test_select_leaf_prioritizes_unvisited_children PASSED

**C4** - `MCTSHNMAgent` raises `ValueError` when initialized with `exploration_c < 0`, `momentum_alpha <= 0` or `momentum_alpha > 1`, or `eps <= 0` (MCTS-01, AC 4)
Proof: run unit | tests/test_mcts_agent.py::test_agent_invalid_hyperparameters_raise PASSED

**C5** - `MCTSHNMAgent.update(leaf_node_id, reward)` increments `tree.visits[l]` by `1` and updates `tree.q_values[l]` via momentum $Q_l^{(t)} = \alpha r + (1 - \alpha) Q_l^{(t-1)}$ (MCTS-02, AC 1)
Proof: run unit | tests/test_mcts_agent.py::test_backpropagation_leaf_momentum_update PASSED

**C6** - `MCTSHNMAgent.update(leaf_node_id, reward)` increments `tree.visits[p]` by `1` and updates every ancestor $p$ up to `root_id` via instance-weighted aggregation $Q_p = \frac{1}{M_p} \sum_{c \in \text{Children}(p)} M_c Q_c$ on both Semantic and Ward Confusion trees (MCTS-02, AC 2)
Proof: run unit | tests/test_mcts_agent.py::test_backpropagation_weighted_ancestor_aggregation PASSED

**C7** - `compute_mcts_reward()` returns the finite mean cross-entropy loss scalar $r = -\frac{1}{K_{\text{mcts}}} \sum \log(1 - \sigma(z_j))$ in Scenario 1 (`is_binary=True`) and the mean multi-class cross-entropy loss in Scenario 2 (`is_binary=False`) (MCTS-02, AC 3)
Proof: run unit | tests/test_mcts_agent.py::test_compute_mcts_reward_binary_and_multiclass PASSED

**C8** - `MCTSHNMAgent.update()` raises `ValueError` when called with an out-of-range node ID, an internal node (`is_leaf == False`), or a `NaN` or infinite `reward` without mutating tree state (MCTS-02, AC 4)
Proof: run unit | tests/test_mcts_agent.py::test_backpropagation_invalid_inputs_raise PASSED

**C9** - `DualStreamBatchSampler` assembles negative mini-batches of size $B_{\text{neg}}$ from $K_{\text{mcts}} = \lfloor (1 - \beta) B_{\text{neg}} \rfloor$ MCTS leaf instances and $K_{\text{uni}} = \lceil \beta B_{\text{neg}} \rceil$ uniform negative instances ($K_{\text{mcts}} + K_{\text{uni}} == B_{\text{neg}}$) across $\beta \in [0.0, 1.0]$ (MCTS-03, AC 1)
Proof: run unit | tests/test_mcts_agent.py::test_dual_stream_batch_sampler_ratios_and_positives PASSED

**C10** - `DualStreamBatchSampler` samples with replacement and emits a warning once per sampler instance when a selected leaf contains fewer instances than $K_{\text{mcts}}$, and raises `ValueError` on invalid sampler parameters (MCTS-03, AC 2)
Proof: run unit | tests/test_mcts_agent.py::test_dual_stream_sampler_small_leaf_replacement_and_validation PASSED

**C11** - `compute_visitation_entropy(tree)` returns Shannon entropy $H_t = -\sum p_i \ln p_i$ and normalized entropy $H_t / \ln K \in [0.0, 1.0]$, returning `(0.0, 0.0)` when total leaf visits equal `0` (MCTS-03, AC 3)
Proof: run unit | tests/test_mcts_agent.py::test_visitation_entropy_zero_single_and_uniform PASSED

**C12** - `compute_superclass_visit_distribution(tree)` and `MCTSDiagnosticsTracker` return 20-superclass visit proportions summing to `1.0` (or `0.0` when unvisited) across both Semantic and Ward Confusion trees (MCTS-03, AC 4)
Proof: run unit | tests/test_mcts_agent.py::test_superclass_visit_distribution_and_tracker PASSED

---

## Handoff

The feature's live state, written as it happens - a compaction gives no warning.

**Settled mid-build:** none
**Dispatched:** none
**Walked:** pending execution

---

## Phase Execution Map

Visual representation of task ordering. Phases run in sequence, and tasks within a phase run in order:

```
Phase 1 → Phase 2

T1 → T2 → T3 → T4
```

---

## Task Granularity Check

| Task | Scope | Status |
| ---- | ----- | ------ |
| T1: Implement UCB1 Selection & Min-Max Q-Normalization in `src/mcts/agent.py` | 1 domain module (`src/mcts/agent.py`) | ✅ Granular |
| T2: Implement Momentum Backpropagation, Hierarchical Aggregation & Reward Calculation | 1 domain module (`src/mcts/agent.py`) | ✅ Granular |
| T3: Implement `DualStreamBatchSampler` in `src/mcts/sampler.py` | 1 domain module (`src/mcts/sampler.py`) | ✅ Granular |
| T4: Implement Tree Visitation Entropy & Superclass Diagnostics in `src/mcts/diagnostics.py` | 1 domain module (`src/mcts/diagnostics.py`) | ✅ Granular |

---

## Diagram-Definition Cross-Check

| Task | Depends On (task body) | Diagram Shows | Status |
| ---- | ---------------------- | ------------- | ------ |
| T1 | None | Phase 1 start (`T1`) | ✅ Match |
| T2 | T1 | `T1 → T2` | ✅ Match |
| T3 | T2 | Phase 1 (`T2`) → Phase 2 (`T3`) | ✅ Match |
| T4 | T3 | `T3 → T4` | ✅ Match |

---

## Test Co-location Validation

| Task | Code Layer Created/Modified | Matrix Requires | Task Says | Status |
| ---- | --------------------------- | --------------- | --------- | ------ |
| T1: Implement UCB1 Selection & Min-Max Q-Normalization in `src/mcts/agent.py` | Domain / MCTS Selection & Q-Normalization (`src/mcts/agent.py`) | unit | unit | ✅ OK |
| T2: Implement Momentum Backpropagation, Hierarchical Aggregation & Reward Calculation | Domain / MCTS Backpropagation & Reward (`src/mcts/agent.py`) | unit | unit | ✅ OK |
| T3: Implement `DualStreamBatchSampler` in `src/mcts/sampler.py` | Domain / Dual-Stream Batch Sampler (`src/mcts/sampler.py`) | unit | unit | ✅ OK |
| T4: Implement Tree Visitation Entropy & Superclass Diagnostics in `src/mcts/diagnostics.py` | Domain / RL Visitation Diagnostics (`src/mcts/diagnostics.py`) | unit | unit | ✅ OK |
