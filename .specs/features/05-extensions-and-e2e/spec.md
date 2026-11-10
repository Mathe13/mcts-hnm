# F5: Dynamic Entropy Leaf Expansion & Tier-2 End-to-End Vision Pipeline Specification

**Approved**: 2026-11-09 - user: "prossiga"

## Problem Statement

The P1 MVP (`F1`–`F4`) keeps MCTS hierarchy leaves fixed at the 100 fine classes of CIFAR-100 and trains a 2-layer MLP head on frozen ResNet-18 embeddings (`AD-001`, `AD-006`). However, within a single hard negative class, individual instances vary widely in visual ambiguity relative to the decision boundary, and frozen features cannot adapt their convolutional representations during hard negative mining. Feature `F5` extends the MCTS-HNM framework with two P2 capabilities (`EPIC-05`): (1) dynamic leaf sub-clustering (`src/mcts/expansion.py`), which splits frequently visited class leaves ($n_l \ge N_{\text{split}}$) into $K_{\text{sub}}$ child leaves partitioned by model prediction entropy $H(p)$ (`AD-006`), and (2) a Tier-2 end-to-end ResNet-18 training pipeline (`src/training/e2e_trainer.py`) on raw $32 \times 32$ RGB images with optional periodic confusion-tree refreshes, orchestrated via versioned notebook `notebooks/04_tier2_e2e_and_expansion.ipynb` (`AD-001`, `AD-008`).

## Goals

- [x] Implement dynamic prediction-entropy leaf expansion (`src/mcts/expansion.py`) that partitions a leaf node's instance pool into $K_{\text{sub}}$ entropy quantiles when its visit count reaches $n_l \ge N_{\text{split}}$, mutating `FastHierarchyTree` in-place while preserving all structural and instance-count invariants (`ISS-13`, `AD-006`).
- [x] Integrate optional dynamic leaf expansion (`enable_leaf_expansion`, `split_threshold`, `k_sub`) into MCTS training loops and `MCTSDiagnosticsTracker` so sub-clustered leaves map cleanly back to their canonical CIFAR-100 fine class (`0..99`) and coarse superclass (`0..19`) (`ISS-13`).
- [x] Implement the Tier-2 end-to-end trainable ResNet-18 vision pipeline (`src/training/e2e_trainer.py`) operating on $32 \times 32$ RGB image tensors across `uniform`, `ohem`, `mcts_strict`, and `mcts_enhanced` strategies, with support for periodic Ward confusion-tree rebuilds (`refresh_tree_every`) and atomic JSON/CSV result export (`ISS-14`, `AD-001`).
- [x] Deliver versioned notebook `notebooks/04_tier2_e2e_and_expansion.ipynb` executing the P2 dynamic leaf expansion ablation and the Tier-2 end-to-end ResNet-18 validation experiment (`ISS-14`, `AD-008`).

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
| ------- | ------ |
| Multi-level recursive leaf splitting beyond 1 sub-cluster depth per fine class | Keeps tree depth bounded and prevents degenerate single-instance leaves (`AD-006`). |
| Distributed multi-GPU DDP training | Single-process CPU/GPU execution is sufficient for CIFAR-100 ResNet-18 experiments on JupyterHub (`AD-001`, `AD-008`). |
| Final LaTeX manuscript compilation (`paper/main.tex`) and publication-wide figure suite (`notebooks/05_paper_artifacts.ipynb`) | Scoped in Feature `F6` (`06-paper-and-artifacts`, `AD-007`). |

---

## Assumptions & Open Questions

Every ambiguity is resolved or recorded here.

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| Prediction entropy formula $H(p)$ for leaf splitting (`ISS-13`, `AD-006`) | For binary classification (`is_binary=True`), compute $p = \sigma(z)$ and binary Shannon entropy $H(p) = -\big(p \ln(p + \epsilon) + (1 - p) \ln(1 - p + \epsilon)\big)$; for multi-class (`is_binary=False`), compute $p = \text{softmax}(z)$ and $H(p) = -\sum_{c} p_c \ln(p_c + \epsilon)$ with $\epsilon = 10^{-12}$ | Standard Shannon prediction entropy measuring classifier uncertainty on each instance in the leaf pool (`EPIC-05` AC 1) | y |
| Sub-cluster partitioning & virtual class ID encoding (`ISS-13`) | Sort the leaf's instance indices by $H(p)$ ascending (ties broken deterministically by dataset index) and split via `np.array_split(sorted_indices, k_sub)` into $K_{\text{sub}}$ non-empty quantile buckets; assign each child leaf a unique virtual class ID `sub_cls_id = (sub_idx + 1) * 1000 + base_class_id` (where `base_class_id = sub_cls_id % 1000`) | Preserves `FastHierarchyTree.validate_tree()` uniqueness invariants and `DualStreamBatchSampler` compatibility while allowing `base_class_id` recovery (`sub_cls_id % 1000`) for superclass diagnostics | y |
| Newly created child leaf $Q$-value & visit initialization (`ISS-13`) | Initialize each new sub-cluster child leaf with `visits = 0` and `q_values = 0.0` (so UCB1 unvisited-child priority explores all $K_{\text{sub}}$ entropy quantiles at least once before exploiting the highest-loss quantile) | Matches MCTS unvisited-node exploration semantics established in Feature `F3` (`MCTS-01`) | y |
| Maximum split depth per class leaf (`ISS-13`) | Only original class leaves (`0 <= leaf_to_class_id < 1000`) with `instance_counts[l] >= k_sub` are eligible for expansion; already-expanded sub-cluster leaves (`leaf_to_class_id >= 1000`) are not split further | Prevents recursive fragmentation into tiny sub-clusters while fulfilling the P2 class-to-subcluster ablation (`AD-006`) | y |
| Tier-2 CIFAR-100 ResNet-18 architecture (`ISS-14`, `AD-001`) | Use a CIFAR-adapted ResNet-18 (`3x3` `conv1` with `stride=1, padding=1` and `Identity` `maxpool` for $32 \times 32$ inputs, plus configurable `width_multiplier` for fast unit-test fixtures) trained end-to-end with `Adam(lr=1e-3)` | Avoids aggressive $7 \times 7$ stride-2 downsampling on $32 \times 32$ CIFAR-100 images while remaining 100% compatible with standard ResNet-18 residual blocks | y |
| Periodic confusion tree refresh (`ISS-14`) | When `refresh_tree_every` is set and `tree_type == "confusion_ward"`, evaluate the current ResNet-18 model on the validation image split, compute a smoothed confusion matrix, and rebuild the Ward Confusion Tree in-place | Tracks evolving feature representations during end-to-end CNN training (`ROADMAP.md` `T5.2`) | y |
| Remaining implicit-requirement dimensions (auth, rate limits, TTL/expiry, external network calls, concurrency) | N/A for single-process PyTorch/NumPy MCTS tree expansion and end-to-end CNN training | Local/JupyterHub compute pipeline with atomic file writes for experiment artifacts | y |

**Open questions:** none - all resolved and recorded above.

---

## User Stories

### P1: Dynamic Leaf Sub-Clustering by Prediction Entropy (`ISS-13`) ⭐ MVP

**User Story**: As a Reinforcement Learning researcher testing whether instance-level difficulty refinement improves hard negative mining (`AD-006`), I want a dynamic leaf expansion module (`src/mcts/expansion.py`) that splits high-visit class leaves ($n_l \ge N_{\text{split}}$) into $K_{\text{sub}}$ child leaves partitioned by model prediction entropy $H(p)$ so that the MCTS agent can focus specifically on the most ambiguous instances within a hard class.

**Why P1**: Implements the core P2 algorithmic extension specified in `EPIC-05` Acceptance Criterion 1 and `ROADMAP.md` `ISS-13` (`T5.1`).

**Acceptance Criteria**:

1. WHEN `compute_prediction_entropy(logits, is_binary)` is called on finite model logits of shape `[N]` / `[N, 1]` (`is_binary=True`) or `[N, C]` (`is_binary=False`) THEN the system SHALL return a 1D `float64` NumPy array of non-negative Shannon entropies $H(p)$ of shape `(N,)`.
2. WHERE dynamic leaf expansion (`enable_leaf_expansion=True`) is enabled, WHEN an eligible class leaf node $l$ (`0 <= leaf_to_class_id[l] < 1000` and `instance_counts[l] >= k_sub`) reaches `tree.visits[l] >= split_threshold` ($N_{\text{split}}$) THEN `maybe_expand_leaf()` SHALL convert node $l$ into an internal node (`tree.is_leaf[l] == False`, `tree.leaf_to_class_id[l] == -1`) and attach $K_{\text{sub}}$ new child leaf nodes partitioned by ascending prediction entropy $H(p)$ such that $\sum_{c \in \text{Children}(l)} M_c == M_l$ and `tree.validate_tree()` passes.
3. WHEN `MCTSHNMAgent`, `DualStreamBatchSampler`, and `MCTSDiagnosticsTracker` operate on an expanded `FastHierarchyTree` THEN `select_leaf()`, `sample_batch()`, `update()`, and `compute_superclass_visit_distribution()` SHALL traverse, sample from, backpropagate through, and map virtual sub-cluster IDs (`sub_cls_id % 1000`) to their canonical CIFAR-100 fine class and superclass without errors.
4. IF `expand_leaf_by_entropy()` or `maybe_expand_leaf()` is called with `k_sub < 2`, `split_threshold < 1`, an internal node ID, an out-of-range node ID, or mismatched entropy/instance counts THEN the system SHALL raise a `ValueError`.

**Independent Test**: Run `.venv/bin/pytest -v tests/test_dynamic_expansion.py -k "entropy or expand"` to verify exact Shannon entropy values, leaf splitting into $K_{\text{sub}}$ quantile children when $n_l \ge N_{\text{split}}$, instance count conservation, and `FastHierarchyTree.validate_tree()` integrity.

---

### P1: Tier-2 End-to-End ResNet-18 Trainer & Confusion Tree Refresh (`ISS-14`) ⭐ MVP

**User Story**: As a researcher validating whether MCTS-HNM gains transfer from frozen embeddings to full representation learning (`AD-001`), I want a Tier-2 end-to-end CNN trainer (`src/training/e2e_trainer.py`) that fine-tunes a ResNet-18 backbone on raw $32 \times 32$ RGB images across `uniform`, `ohem`, `mcts_strict`, and `mcts_enhanced` strategies (with optional dynamic leaf expansion and periodic confusion-tree refresh) so that I can compare end-to-end test metrics and export structured JSON/CSV artifacts.

**Why P1**: Fulfills `EPIC-05` Acceptance Criterion 2 and `ROADMAP.md` `ISS-14` (`T5.2`).

**Acceptance Criteria**:

1. WHEN `Tier2ResNet18(num_outputs=K)` is called on a 4D RGB image batch of shape `[B, 3, 32, 32]` THEN the model SHALL return logits of shape `[B, K]` (or `[B]` when `squeeze_binary=True` and `num_outputs=1`) and `[B, D]` feature embeddings via `forward_features()`.
2. WHEN `train_e2e_scenario1()` or `train_e2e_scenario2()` is executed for a strategy in `{"uniform", "ohem", "mcts_strict", "mcts_enhanced"}` THEN the system SHALL train `Tier2ResNet18` end-to-end with `Adam(lr=lr)`, update the MCTS tree on every step for MCTS strategies (including dynamic leaf expansion when `enable_leaf_expansion=True`), optionally rebuild the Ward Confusion Tree every `refresh_tree_every` steps, and record evaluation snapshots at every `eval_every` steps.
3. WHEN `run_e2e_experiment()` is invoked across strategies and seeds THEN the system SHALL execute each end-to-end run deterministically, aggregate `mean ± std` test metrics, compute paired statistical tests against `mcts_enhanced`, and atomically write JSON and CSV summaries to `output_dir`.
4. IF `train_e2e_scenario1()`, `train_e2e_scenario2()`, or `run_e2e_experiment()` receives an unknown strategy, invalid image tensor shape (not 4D `[N, 3, H, W]`), `num_steps <= 0`, or `eval_every <= 0` THEN the system SHALL raise a `ValueError`.

**Independent Test**: Run `.venv/bin/pytest -v tests/test_dynamic_expansion.py -k "e2e"` on a synthetic $32 \times 32$ RGB dataset fixture to verify end-to-end ResNet-18 gradient updates, dynamic expansion during training, periodic confusion-tree refresh, and atomic JSON/CSV output.

---

### P2: Versioned Notebook 04 for Dynamic Expansion & Tier-2 E2E Validation (`ISS-14`)

**User Story**: As an experimenter on a remote JupyterHub (`AD-008`), I want a versioned Jupyter notebook (`notebooks/04_tier2_e2e_and_expansion.ipynb`) that orchestrates both the P2 dynamic entropy leaf expansion ablation and the Tier-2 end-to-end ResNet-18 benchmark so that the complete `F5` experimental suite runs reproducibly from top to bottom.

**Why P2**: Fulfills `AD-008` and `EPIC-05` Acceptance Criterion 2 by providing the interactive notebook orchestration layer for remote JupyterHub execution.

**Acceptance Criteria**:

1. WHEN `notebooks/04_tier2_e2e_and_expansion.ipynb` is executed end-to-end THEN the notebook SHALL run without errors, execute the dynamic entropy leaf expansion ablation (`enable_leaf_expansion=False` vs. `True`) and the Tier-2 end-to-end ResNet-18 comparison (`uniform`, `ohem`, `mcts_enhanced`), render diagnostic plots and summary tables, and save JSON/CSV results under `results/`.

**Independent Test**: Run `.venv/bin/pytest -v tests/test_dynamic_expansion.py -k "notebook_04"` to execute `notebooks/04_tier2_e2e_and_expansion.ipynb` via `nbclient` and verify all cells complete with zero errors and produce output artifacts.

---

## Edge Cases

- IF a leaf node reaches `visits[l] >= split_threshold` (`N_split`) but already represents an expanded sub-cluster (`leaf_to_class_id[l] >= 1000`) or contains fewer than `k_sub` instances (`instance_counts[l] < k_sub`) THEN `maybe_expand_leaf()` SHALL return `False` without modifying the tree.
- IF `k_sub` exceeds the current column width `B_max` of `tree.children` (`shape: [V, B_max]`) during `expand_leaf_by_entropy()` THEN the system SHALL widen `tree.children` to `max(B_max, k_sub)` padded with `-1` so all existing and new child slices remain valid.
- IF all instances in a leaf node have identical prediction entropy $H(p)$ THEN `expand_leaf_by_entropy()` SHALL break ties deterministically by ascending sample index and partition into $K_{\text{sub}}$ non-empty buckets via `np.array_split`.
- WHEN `DualStreamBatchSampler` is already instantiated before a leaf splits, its `sample_batch(new_child_leaf_id)` call SHALL automatically resolve the new child leaf's `leaf_to_class_id` and `class_to_indices` entry from the mutated `FastHierarchyTree` without requiring sampler reconstruction.

---

## Requirement Traceability

Each requirement maps to a unique ID for tracking across design, tasks, and verification.

| Requirement ID | Story | Phase | Status |
| -------------- | ----- | ----- | ------ |
| EXP-01 | P1: Dynamic Leaf Sub-Clustering by Prediction Entropy (`ISS-13`) | Execute | Verified |
| EXP-02 | P1: Tier-2 End-to-End ResNet-18 Trainer & Confusion Tree Refresh (`ISS-14`) | Execute | Verified |
| EXP-03 | P2: Versioned Notebook 04 for Dynamic Expansion & Tier-2 E2E Validation (`ISS-14`) | Execute | Verified |

**Coverage:** 3 total requirements mapped to `EPIC-05`.

---

## Differences from the source

Every departure from the original *GiantHunter* paper (`btaf239.pdf`) and official repository (`FuchuanQu/GiantHunter`) is linked to an active architectural decision in `.specs/STATE.md`.

| Source shows | Product shows | Confirmed |
| ------------ | ------------- | --------- |
| GiantHunter uses a static biological taxonomy tree whose leaves remain permanently fixed at the species level | Adds dynamic leaf sub-clustering (`src/mcts/expansion.py`) that splits high-visit class leaves ($n_l \ge N_{\text{split}}$) into $K_{\text{sub}}$ prediction-entropy quantiles | y - AD-006 |
| GiantHunter trains a 1-layer Transformer on genomic Protein-Cluster tokens | Implements a Tier-2 end-to-end trainable ResNet-18 (`src/training/e2e_trainer.py`) on raw $32 \times 32$ CIFAR-100 RGB images with periodic confusion-tree refresh | y - AD-001 |
| GiantHunter executes via a monolithic CLI script (`gianthunter.py`) | Orchestrates dynamic leaf expansion and Tier-2 end-to-end experiments via `notebooks/04_tier2_e2e_and_expansion.ipynb` calling modular APIs in `src/` | y - AD-008 |

## Sources

- `.specs/features/00-epic-roadmap/spec.md` (`EPIC-05`) - parent epic specification and acceptance criteria for Feature `F5`.
- `ROADMAP.md` (`Feature F5`, `ISS-13`–`ISS-14`, `T5.1`–`T5.3`) - planned issue and task breakdown.
- `ARCHITECTURE.md` (`Section 2: Reinforcement Learning Formulation` & `Section 5: Two-Tier Execution & Notebook Orchestration`) - architectural design of Tier-2 end-to-end execution and tree state transitions.
- `.specs/STATE.md` (`AD-001`, `AD-002`, `AD-004`, `AD-005`, `AD-006`, `AD-008`) - active architectural decisions governing dynamic leaf expansion and Tier-2 ResNet-18 validation.

---

## Success Criteria

- [x] `.venv/bin/pytest -v tests/test_dynamic_expansion.py` passes 100% of unit and integration checks covering `EXP-01`, `EXP-02`, and `EXP-03`.
- [x] `expand_leaf_by_entropy()` and `maybe_expand_leaf()` deterministically split eligible leaves ($n_l \ge N_{\text{split}}$) into $K_{\text{sub}}$ entropy quantile child leaves while preserving total instance counts and `FastHierarchyTree.validate_tree()` invariants.
- [x] `train_e2e_scenario1()`, `train_e2e_scenario2()`, and `run_e2e_experiment()` fine-tune `Tier2ResNet18` end-to-end on $32 \times 32$ RGB images across baselines and MCTS-HNM, supporting both dynamic leaf expansion and periodic Ward confusion-tree refreshes.
- [x] `notebooks/04_tier2_e2e_and_expansion.ipynb` executes cleanly from top to bottom via `nbclient` and saves structured JSON/CSV artifacts under `results/`.
