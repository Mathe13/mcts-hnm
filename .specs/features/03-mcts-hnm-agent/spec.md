# F3: Core MCTS Hard Negative Mining Agent Specification

**Approved**: 2026-11-08 - user: "prossiga"

## Problem Statement

While *GiantHunter* (*Bioinformatics*, 2025) introduced a Monte Carlo Tree Search (MCTS) framework to dynamically select hard negative classes from a taxonomic tree, its official repository (`FuchuanQu/GiantHunter`) omits the MCTS training implementation entirely. Moreover, the paper's raw UCB1 formulation ($U_i = Q_i + c \sqrt{\ln N_s / (n_i + \epsilon)}$) uses unnormalized cross-entropy loss values as action-values $Q_i$, which can distort the exploration-exploitation balance as loss scales shift during training, and samples 100% of each negative batch from a single selected leaf, risking catastrophic forgetting on previously learned classes (`AD-005`). Feature `F3` implements the modular 4-phase MCTS Hard Negative Mining engine in `src/mcts/` — comprising UCB1 selection with configurable Min-Max $Q$-normalization (`src/mcts/agent.py`), momentum leaf updates and instance-weighted ancestor backpropagation (`src/mcts/agent.py`), Dual-Stream hybrid batch sampling (`src/mcts/sampler.py`), and Tree Visitation Entropy and superclass visitation diagnostics (`src/mcts/diagnostics.py`).

## Goals

- [x] Implement `MCTSHNMAgent` (`src/mcts/agent.py`) supporting deterministic UCB1 tree traversal from root to leaf with unvisited-node priority ($n_i = 0$) and a configurable `q_norm: bool` toggle comparing Strict GiantHunter (raw $Q_i$) against Enhanced MCTS-HNM (Min-Max sibling normalization $\bar{Q}_i^{\text{norm}} \in [0, 1]$) (`ISS-07`, `AD-005`).
- [x] Implement MCTS backpropagation (`src/mcts/agent.py`) updating the selected leaf's action-value via exponential moving average momentum $Q_l^{(t)} = \alpha r + (1 - \alpha) Q_l^{(t-1)}$ (default $\alpha = 0.3$) and updating every ancestor node $p$ bottom-up via instance-weighted aggregation $Q_p = \frac{1}{M_p} \sum_{c \in \text{Children}(p)} M_c Q_c$ (`ISS-08`).
- [x] Implement `DualStreamBatchSampler` (`src/mcts/sampler.py`) assembling negative mini-batches of size $B_{\text{neg}}$ from $\lfloor (1 - \beta) B_{\text{neg}} \rfloor$ MCTS-selected leaf instances and $\lceil \beta B_{\text{neg}} \rceil$ uniformly sampled anchor instances for $\beta \in [0.0, 1.0]$, with optional positive class sampling for Scenario 1 (`ISS-09`, `AD-005`).
- [x] Implement RL diagnostic metrics (`src/mcts/diagnostics.py`) including Tree Visitation Entropy $H_t = -\sum_{i \in \text{Leaves}} p_i \ln p_i$, normalized entropy $H_t / \ln K$, and per-superclass visit distribution tracking across both Semantic and Ward Confusion trees (`ISS-09`).

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
| ------- | ------ |
| Neural network MLP classification heads and Uniform / OHEM baseline samplers | Covered in Feature `F4` (`04-tier1-benchmark-suite`). |
| Multi-seed training loops, statistical significance tests, and `notebooks/03_tier1_mcts_benchmarks.ipynb` | Covered in Feature `F4` (`04-tier1-benchmark-suite`). |
| Dynamic leaf sub-clustering by prediction entropy ($n_l \ge N_{\text{split}}$) | Scoped as P2 extension in Feature `F5` (`05-extensions-and-e2e`, `AD-006`). |
| Publication PDF figure generation and LaTeX manuscript tables | Covered in Feature `F6` (`06-paper-and-artifacts`). |

---

## Assumptions & Open Questions

Every ambiguity is resolved or recorded here.

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| UCB1 formula and unvisited-child priority ($n_i = 0$) | Compute $U_i = \tilde{Q}_i + c \sqrt{\frac{\ln \max(N_s, 1)}{n_i + \epsilon}}$ with $\epsilon = 10^{-5}$; when one or more children at internal node $s$ have $n_i == 0$, restrict selection to unvisited children ($n_i == 0$) and select the first unvisited child deterministically | Guarantees that unvisited children are always explored before visited siblings even when $N_s \in \{0, 1\}$ (where $\ln \max(N_s, 1) = 0$) or when raw $Q_j$ is large in `q_norm=False` mode, while using the exact GiantHunter Eq. (1) UCB1 score once all children at node $s$ have $n_i \ge 1$ | y |
| Min-Max $Q$-normalization formula (`q_norm=True`) | For sibling values $Q = [Q_1, \dots, Q_B]$ at node $s$, if $Q_{\max} == Q_{\min}$ then $\bar{Q}_i^{\text{norm}} = 0.0$ for all $i$; otherwise $\bar{Q}_i^{\text{norm}} = \frac{Q_i - Q_{\min}}{Q_{\max} - Q_{\min} + \epsilon}$ with $\epsilon = 10^{-5}$ | Matches `EPIC-03` Acceptance Criterion 2 and Edge Case 2, bounding normalized exploitation values in $[0, 1]$ and eliminating division-by-zero (`NaN`) when sibling $Q$-values are equal | y |
| Leaf momentum initialization and update rule | Initialize `tree.q_values` at `0.0` ($Q_l^{(0)} = 0.0$) and update on every step $t \ge 1$ via $Q_l^{(t)} = \alpha r + (1 - \alpha) Q_l^{(t-1)}$ with default $\alpha = 0.3$ | Matches GiantHunter Eq. (3) and `EPIC-03` Acceptance Criterion 3 | y |
| Dual-Stream ratio $\beta$ (`dual_stream_ratio`) convention | $\beta \in [0.0, 1.0]$ specifies the uniform anchor fraction: $K_{\text{mcts}} = \lfloor (1 - \beta) B_{\text{neg}} \rfloor$ MCTS-selected leaf instances and $K_{\text{uni}} = \lceil \beta B_{\text{neg}} \rceil$ uniform negative instances ($\beta = 0.0$ yields 100% MCTS batch; $\beta = 0.5$ yields 50/50 Dual-Stream batch) | Matches `EPIC-03` Acceptance Criterion 4 and `.specs/features/00-epic-roadmap/context.md` (`AD-005`) | y |
| Superclass visitation distribution on Ward Confusion Trees | Aggregate leaf visit counts $n_l$ by mapping each leaf's `leaf_to_class_id[l]` to its canonical CIFAR-100 coarse superclass via `get_cifar100_taxonomy().fine_id_to_coarse_id` | Allows direct comparison of superclass visitation distributions between Mode A (Semantic Tree) and Mode B (Ward Confusion Tree) even though Mode B internal nodes are binary Ward merges rather than the 20 superclasses | y |
| Remaining implicit-requirement dimensions (auth, rate limits, TTL/expiry, external services, concurrency) | N/A for in-memory NumPy MCTS tree traversal, backpropagation, and deterministic batch index sampling | Single-process scientific RL library with no network I/O or multi-user state | y |

**Open questions:** none - all resolved and recorded above.

---

## User Stories

### P1: UCB1 Selection & Min-Max Q-Normalization (`ISS-07`) ⭐ MVP

**User Story**: As a Reinforcement Learning researcher conducting the Strict vs. Enhanced MCTS ablation (`AD-005`), I want `MCTSHNMAgent` (`src/mcts/agent.py`) to traverse any `FastHierarchyTree` from root to leaf using UCB1 with configurable `q_norm: bool` and deterministic unvisited-node priority so that I can isolate the effect of sibling $Q$-normalization on exploration dynamics.

**Why P1**: Implements Phase 1 (Selection) of the MCTS-HNM algorithm (`EPIC-03`, `AD-005`).

**Acceptance Criteria**:

1. WHEN `select_leaf()` is called on an `MCTSHNMAgent` configured with `q_norm=False` (Strict GiantHunter mode) at an internal node $s$ where all children have been visited ($n_i \ge 1$) THEN the system SHALL select the child $i$ maximizing $U_i = Q_i + c \sqrt{\frac{\ln N_s}{n_i + \epsilon}}$.
2. WHERE `q_norm=True` (Enhanced MCTS-HNM mode) is enabled, the system SHALL normalize sibling $Q$-values at internal node $s$ into $[0, 1]$ via $\bar{Q}_i^{\text{norm}} = \frac{Q_i - Q_{\min}}{Q_{\max} - Q_{\min} + \epsilon}$ (setting $\bar{Q}_i^{\text{norm}} = 0.0$ when $Q_{\max} == Q_{\min}$) before adding the UCB1 exploration bonus $c \sqrt{\frac{\ln N_s}{n_i + \epsilon}}$.
3. WHEN an internal node $s$ has one or more unvisited children ($n_i = 0$) during `select_leaf()` THEN the system SHALL prioritize unvisited children deterministically before selecting any already-visited sibling ($n_j \ge 1$).
4. IF `MCTSHNMAgent` is initialized with `exploration_c < 0`, `momentum_alpha <= 0` or `momentum_alpha > 1`, or `eps <= 0` THEN the system SHALL raise a `ValueError`.

**Independent Test**: Run `pytest tests/test_mcts_agent.py -k "selection or q_norm"` to verify unvisited-child priority, exact UCB1 scores in Strict vs. Enhanced modes, and zero-division protection when $Q_{\max} == Q_{\min}$.

---

### P1: Momentum Backpropagation & Hierarchical Aggregation (`ISS-08`) ⭐ MVP

**User Story**: As an RL researcher, I want `MCTSHNMAgent.update(leaf_node_id, reward)` (`src/mcts/agent.py`) to update the selected leaf's $Q$-value via exponential moving average momentum and propagate instance-weighted $Q$-values and visit counts up to the root so that internal clade values accurately reflect non-stationary classifier loss.

**Why P1**: Implements Phase 4 (Backpropagation) of GiantHunter Equations (3) and (4) (`EPIC-03`).

**Acceptance Criteria**:

1. WHEN `update(leaf_node_id, reward)` is called for leaf node $l$ with scalar reward $r$ THEN the system SHALL increment `tree.visits[l]` by `1` and update `tree.q_values[l]` in-place via $Q_l^{(t)} = \alpha r + (1 - \alpha) Q_l^{(t-1)}$.
2. WHEN `update(leaf_node_id, reward)` updates leaf node $l$ THEN the system SHALL traverse every ancestor node $p$ along the path from `tree.parents[l]` to `tree.root_id`, incrementing `tree.visits[p]` by `1` and recomputing `tree.q_values[p]` via instance-weighted aggregation $Q_p = \frac{1}{M_p} \sum_{c \in \text{Children}(p)} M_c Q_c$ where $M_c = \text{tree.instance\_counts}[c]$ and $M_p = \text{tree.instance\_counts}[p]$.
3. WHEN `compute_mcts_reward(logits, is_binary=True, target_labels=None)` is called on the model's logits for the MCTS-mined sub-batch THEN the system SHALL return the finite mean cross-entropy loss scalar $r = -\frac{1}{K_{\text{mcts}}} \sum_{j=1}^{K_{\text{mcts}}} \log(1 - \sigma(z_j))$ in Scenario 1 (`is_binary=True`) or the mean multi-class cross-entropy loss in Scenario 2 (`is_binary=False`).
4. IF `update(leaf_node_id, reward)` is called with a `leaf_node_id` that is out of range or is an internal node (`tree.is_leaf[leaf_node_id] == False`), or with a `NaN` or infinite `reward` THEN the system SHALL raise a `ValueError`.

**Independent Test**: Run `pytest tests/test_mcts_agent.py -k "backprop or reward"` to verify multi-step momentum updates ($\alpha=0.3$), exact bottom-up $M_c / M_p$ weighted ancestor $Q$-values, visit count conservation ($N_p = \sum_{c \in \text{Children}(p)} n_c$), and `ValueError` on invalid node IDs or `NaN` rewards.

---

### P1: Dual-Stream Batch Sampler & RL Diagnostics (`ISS-09`) ⭐ MVP

**User Story**: As a researcher analyzing MCTS sample selection and preventing catastrophic forgetting (`AD-005`), I want `DualStreamBatchSampler` (`src/mcts/sampler.py`) and visitation diagnostics (`src/mcts/diagnostics.py`) so that each training step assembles hybrid MCTS + uniform negative mini-batches and tracks Tree Visitation Entropy $H_t$ and superclass visit distributions.

**Why P1**: Completes Phase 2 (Leaf Rollout / Dual-Stream Batching) and provides the RL observability metrics required by the paper (`EPIC-03`, `AD-005`).

**Acceptance Criteria**:

1. WHILE `DualStreamBatchSampler` is configured with `dual_stream_ratio` $\beta \in [0.0, 1.0]$ and negative batch size $B_{\text{neg}} \ge 1$, the system SHALL assemble each negative mini-batch from $K_{\text{mcts}} = \lfloor (1 - \beta) B_{\text{neg}} \rfloor$ instances sampled from the MCTS-selected leaf's class pool and $K_{\text{uni}} = \lceil \beta B_{\text{neg}} \rceil$ instances sampled uniformly from the global negative pool such that $K_{\text{mcts}} + K_{\text{uni}} == B_{\text{neg}}$.
2. IF a selected leaf node contains fewer instances than the requested MCTS sub-batch size ($|I_l| < K_{\text{mcts}}$) THEN `DualStreamBatchSampler` SHALL sample the $K_{\text{mcts}}$ leaf instances with replacement and emit a warning once per sampler instance.
3. WHEN `compute_visitation_entropy(tree)` is called on a `FastHierarchyTree` with leaf visit counts $n_1, \dots, n_K$ THEN the system SHALL return the Shannon entropy $H_t = -\sum_{i \in \text{Leaves}, \, n_i > 0} p_i \ln p_i$ (where $p_i = n_i / \sum_{j \in \text{Leaves}} n_j$) and normalized entropy $H_t / \ln K \in [0.0, 1.0]$, returning `(0.0, 0.0)` when total leaf visits equal `0`.
4. WHEN `compute_superclass_visit_distribution(tree)` is called on either a Semantic Tree or a Ward Confusion Tree THEN the system SHALL return a dictionary mapping all 20 CIFAR-100 coarse superclass names to their aggregated leaf visit proportions summing to `1.0` (or `0.0` when total leaf visits equal `0`).

**Independent Test**: Run `pytest tests/test_mcts_agent.py -k "sampler or diagnostics"` to verify exact $K_{\text{mcts}}$ and $K_{\text{uni}}$ splits across $\beta \in \{0.0, 0.5, 1.0\}$, replacement fallback with single-warning emission on small leaves, and exact Shannon entropy $H_t$ on uniform ($\ln K$) vs. single-leaf ($0.0$) visit distributions.

---

## Edge Cases

- IF a leaf node contains fewer instances than the requested MCTS sub-batch size $K_{\text{mcts}}$ THEN `DualStreamBatchSampler` SHALL sample with replacement and log a warning once per sampler run.
- IF all sibling nodes at an internal tree level have identical $Q$-values ($Q_{\max} == Q_{\min}$) THEN `normalize_sibling_q_values()` SHALL return an array of zeros (`0.0`) so selection relies purely on the UCB1 exploration term without division-by-zero (`NaN`).
- WHEN a node has never been visited ($n_i = 0$) THEN `MCTSHNMAgent.select_leaf()` SHALL prioritize it deterministically before exploiting already-visited siblings ($n_j \ge 1$).
- IF `compute_visitation_entropy(tree)` or `compute_superclass_visit_distribution(tree)` is called before any leaf has been visited ($\sum_{j \in \text{Leaves}} n_j == 0$) THEN the system SHALL return `0.0` entropy and `0.0` visit proportions without division by zero.
- IF `DualStreamBatchSampler` is initialized with `neg_batch_size <= 0`, `dual_stream_ratio < 0.0` or `dual_stream_ratio > 1.0`, or `pos_batch_size < 0` THEN the system SHALL raise a `ValueError`.

---

## Requirement Traceability

Each requirement maps to a unique ID for tracking across design, tasks, and verification.

| Requirement ID | Story | Phase | Status |
| -------------- | ----- | ----- | ------ |
| MCTS-01 | P1: UCB1 Selection & Min-Max Q-Normalization (`ISS-07`) | Execute | Verified |
| MCTS-02 | P1: Momentum Backpropagation & Hierarchical Aggregation (`ISS-08`) | Execute | Verified |
| MCTS-03 | P1: Dual-Stream Batch Sampler & RL Diagnostics (`ISS-09`) | Execute | Verified |

**Coverage:** 3 total requirements mapped to `EPIC-03`.

---

## Differences from the source

Every departure from the original *GiantHunter* paper (`btaf239.pdf`) and official repository (`FuchuanQu/GiantHunter`) is linked to an active architectural decision in `.specs/STATE.md`.

| Source shows | Product shows | Confirmed |
| ------------ | ------------- | --------- |
| Official `FuchuanQu/GiantHunter` repository omits the MCTS training loop (`agent.py`, `sampler.py`, `diagnostics.py`) | Full modular Python implementation of the 4-phase MCTS agent, Dual-Stream sampler, and visitation entropy diagnostics under `src/mcts/` | y - AD-001 |
| GiantHunter Eq. (1) uses raw unnormalized $Q_i$ in UCB1 ($U_i = Q_i + c \sqrt{\ln N_i / n_i}$) | Supports both Strict GiantHunter (`q_norm=False`) and Enhanced MCTS-HNM (`q_norm=True`, normalizing sibling $Q$-values via Min-Max into $[0, 1]$) | y - AD-005 |
| GiantHunter samples 100% of negative mini-batch instances from the single MCTS-selected leaf | Supports configurable `dual_stream_ratio` $\beta \in [0.0, 1.0]$ combining $\lfloor (1-\beta) B_{\text{neg}} \rfloor$ MCTS leaf negatives with $\lceil \beta B_{\text{neg}} \rceil$ uniform anchor negatives | y - AD-005 |
| GiantHunter computes reward only for binary classification (`BCEWithLogitsLoss`) | Supports reward computation for both Scenario 1 (binary One-vs-Rest) and Scenario 2 (100-way multi-class curriculum) | y - AD-004 |

## Sources

- `.specs/features/00-epic-roadmap/spec.md` (`EPIC-03`) - parent epic specification and acceptance criteria for Feature `F3`.
- `ROADMAP.md` (`Feature F3`, `ISS-07`–`ISS-09`, `T3.1`–`T3.4`) - planned issue and task decomposition.
- `ARCHITECTURE.md` (`Section 2: Reinforcement Learning Formulation` & `Section 4: Dual-Stream Mini-Batch Training`) - mathematical specification of UCB1, Min-Max $Q$-normalization, momentum update, weighted parent aggregation, and Dual-Stream batching.
- `../btaf239.pdf` (*Qu et al., Bioinformatics 2025, 41, i30–i39*) - binding reference for GiantHunter Equations (1)–(4) ($c=2.0$, $\alpha=0.3$, $\epsilon=10^{-5}$, instance-weighted parent update $M_c / M_p$).

---

## Success Criteria

- [x] `.venv/bin/pytest -v tests/test_mcts_agent.py` passes 100% of unit and integration checks covering `MCTS-01`, `MCTS-02`, and `MCTS-03`.
- [x] `MCTSHNMAgent` deterministically explores all unvisited children before revisiting any visited sibling and accurately switches between Strict GiantHunter (`q_norm=False`) and Enhanced MCTS-HNM (`q_norm=True`).
- [x] `MCTSHNMAgent.update()` matches exact analytical values for multi-step leaf momentum ($Q_l^{(t)} = \alpha r + (1-\alpha) Q_l^{(t-1)}$) and bottom-up instance-weighted parent aggregation ($Q_p = \frac{1}{M_p} \sum M_c Q_c$) on both Semantic and Ward Confusion trees.
- [x] `DualStreamBatchSampler` and `src/mcts/diagnostics.py` enforce exact batch size conservation ($K_{\text{mcts}} + K_{\text{uni}} == B_{\text{neg}}$), single-warning replacement fallback for small leaves, and non-NaN Tree Visitation Entropy $H_t$.
