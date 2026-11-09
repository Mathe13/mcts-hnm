# F2: Dual Hierarchy Construction Engine Specification

**Approved**: 2026-11-08 - user: "prossiga"

## Problem Statement

Monte Carlo Tree Search for Hard Negative Mining (MCTS-HNM) requires navigating a hierarchical tree of candidate classes at every training step (`60,000+` iterations), making pointer-based Python tree objects a computational bottleneck. Furthermore, *GiantHunter* relies exclusively on a static biological taxonomy tree, leaving untested the central scientific hypothesis of this work (`AD-002`): whether an empirical **Confusion-Driven Hierarchy** (built via Ward's minimum variance clustering over model validation errors) guides MCTS toward decision-boundary hard negatives more effectively than a static **Semantic Hierarchy** (CIFAR-100's 20 coarse superclasses $\rightarrow$ 100 fine classes). Feature `F2` implements the vectorized `FastHierarchyTree` data structure (`src/trees/base_tree.py`), the native CIFAR-100 Semantic Tree builder (`src/trees/semantic_tree.py`), the data-driven Ward Confusion Tree builder (`src/trees/confusion_tree.py`), and the versioned notebook orchestrator (`notebooks/02_tree_construction.ipynb`).

## Goals

- [x] Implement a vectorized, contiguous-array tree representation (`FastHierarchyTree` in `src/trees/base_tree.py`) storing `parents`, `children`, `child_counts`, `is_leaf`, `leaf_to_class_id`, `instance_counts`, `visits`, and `q_values` in NumPy arrays with deterministic `.npz` serialization and state reset (`ISS-04`).
- [x] Implement `build_semantic_tree()` (`src/trees/semantic_tree.py`) constructing the 3-level CIFAR-100 Semantic Hierarchy (Virtual Root $\rightarrow$ 20 Superclasses $\rightarrow$ 99 Negative Leaves in Scenario 1 or 100 Fine Class Leaves in Scenario 2) with exact bottom-up instance count aggregation (`ISS-05`, `AD-002`, `AD-004`).
- [x] Implement `compute_confusion_distance_matrix()` and `build_confusion_tree()` (`src/trees/confusion_tree.py`) converting a validation confusion matrix $C \in \mathbb{R}^{K \times K}$ into symmetric normalized distances $D_{i,j} = 1 - \frac{C_{i,j} + C_{j,i}}{2 \cdot \max(C)}$ and a binary Ward linkage hierarchy (`scipy.cluster.hierarchy.linkage(method="ward")`), plus a helper `estimate_validation_confusion_matrix()` to compute $C$ from cached Tier-1 embeddings (`ISS-06`, `AD-002`).
- [x] Deliver a versioned Jupyter notebook (`notebooks/02_tree_construction.ipynb`) that constructs both Semantic and Confusion trees for Scenario 1 and Scenario 2, renders confusion matrix heatmaps and Ward dendrograms, verifies structural invariants, and exports serialized tree artifacts (`ISS-06`, `AD-008`).

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
| ------- | ------ |
| UCB1 selection policy, Min-Max $Q$-normalization, and momentum backpropagation | Covered in Feature `F3` (`03-mcts-hnm-agent`). |
| Dual-Stream mini-batch sampling and Tree Visitation Entropy tracking | Covered in Feature `F3` (`03-mcts-hnm-agent`). |
| Full multi-seed MLP/CNN benchmark training loops and OHEM baselines | Covered in Feature `F4` (`04-tier1-benchmark-suite`). |
| Dynamic leaf sub-clustering by prediction entropy ($n_l \ge N_{\text{split}}$) | Scoped as P2 extension in Feature `F5` (`05-extensions-and-e2e`, `AD-006`). |

---

## Assumptions & Open Questions

Every ambiguity is resolved or recorded here.

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| Vectorized `children` representation for variable branching factors (`FastHierarchyTree`) | 2D contiguous `int64` array `children` of shape `[num_nodes, max_branching]` padded with `-1` alongside 1D `child_counts: int64 [num_nodes]` and helper `get_children(node_id)` | Supports both 20-way/5-way branching in the Semantic Tree and 2-way binary branching in the Ward Confusion Tree without ragged Python lists | y |
| Confusion distance diagonal and off-diagonal self-confusion convention | Compute off-diagonal symmetric similarity from misclassifications ($C^{\text{off}}_{i,j}$ with zeroed diagonal $C^{\text{off}}_{i,i} = 0$) via $D_{i,j} = 1 - \frac{C^{\text{off}}_{i,j} + C^{\text{off}}_{j,i}}{2 \cdot \max(C^{\text{off}})}$ for $i \neq j$ and enforce $D_{i,i} = 0.0$ (if $\max(C^{\text{off}}) == 0$, off-diagonal $D_{i,j} = 1.0$) | Diagonal entries $C_{i,i}$ represent correct classifications; zeroing the diagonal before scaling by $\max(C^{\text{off}})$ prevents high accuracy on easy classes from compressing off-diagonal confusion distances toward `1.0` while preserving exact symmetry and $D_{i,i} = 0$ for `scipy.spatial.distance.squareform` | y |
| Scenario 1 (99 leaves) vs. Scenario 2 (100 leaves) support in both tree builders | Accept `class_to_indices: dict[int, np.ndarray]` (99 negative classes from `Scenario1Split.negative_class_to_indices` or 100 classes from `Scenario2Split.class_to_indices`); in Scenario 1 Confusion Tree, slice the $99 \times 99$ submatrix corresponding to the sorted active class IDs | Directly consumes the `Scenario1Split` and `Scenario2Split` contracts produced in Feature `F1` (`AD-004`) | y |
| Baseline validation confusion matrix estimation for Notebook 02 | Provide `estimate_validation_confusion_matrix(train_embeddings, train_labels, val_embeddings, val_labels, seed=42)` using a deterministic linear probe (`LogisticRegression` on `[N, 512]` embeddings) with Laplace smoothing ($\epsilon = 10^{-3}$) so every class row has positive sum | Produces a fast, deterministic $100 \times 100$ validation confusion matrix from `F1` cached embeddings in $< 3$ seconds on CPU | y |
| Tree serialization format | Compressed NumPy archive (`.npz`) via `FastHierarchyTree.save(path)` and `FastHierarchyTree.load(path)` with atomic `.tmp.npz` rename | Zero pickle code-execution risk, fast I/O, and native preservation of all contiguous NumPy arrays and tree metadata | y |
| Remaining implicit-requirement dimensions (auth, rate limits, TTL/expiry, external services, concurrency) | N/A for deterministic in-memory NumPy tree construction and local `.npz` artifact serialization | Single-process scientific library with no network calls or multi-user state | y |

**Open questions:** none - all resolved and recorded above.

---

## User Stories

### P1: Vectorized Hierarchy Tree Data Structure (`ISS-04`) ⭐ MVP

**User Story**: As an RL researcher running 60,000+ MCTS iterations per experiment, I want a vectorized `FastHierarchyTree` class (`src/trees/base_tree.py`) backed by contiguous NumPy arrays so that tree traversal, backpropagation, state resets, and serialization execute without Python object pointer overhead.

**Why P1**: Both the Semantic Tree (`ISS-05`) and Ward Confusion Tree (`ISS-06`), as well as the MCTS agent in `F3`, depend on this shared container and its structural invariants.

**Acceptance Criteria**:

1. The `FastHierarchyTree` class SHALL store node attributes in contiguous 1D/2D NumPy arrays: `parents` (`int64 [V]`), `children` (`int64 [V, B_max]`, padded with `-1`), `child_counts` (`int64 [V]`), `is_leaf` (`bool [V]`), `depths` (`int64 [V]`), `leaf_to_class_id` (`int64 [V]`, `-1` for internal nodes), `instance_counts` (`int64 [V]`), `visits` (`int64 [V]`), and `q_values` (`float64 [V]`).
2. WHEN `validate_tree()` is called on a `FastHierarchyTree` instance THEN the system SHALL verify that `root_id` has `parents[root_id] == -1`, every non-root node is reachable from `root_id` without cycles, every internal node has `child_counts[p] >= 1` and `instance_counts[p] == sum(instance_counts[c] for c in get_children(p))`, and every leaf node has `is_leaf[l] == True`, `child_counts[l] == 0`, `leaf_to_class_id[l] >= 0`, and `instance_counts[l] > 0`.
3. IF a `FastHierarchyTree` is constructed or validated with a disconnected node, a cycle, a negative or zero leaf `instance_count`, or a parent `instance_count` that does not equal the sum of its children's `instance_counts` THEN the system SHALL raise a `ValueError` describing the violated tree invariant.
4. WHEN `reset_mcts_state()` is called on a `FastHierarchyTree` THEN the system SHALL zero out `visits` (`0`) and `q_values` (`0.0`) across all nodes in-place while preserving tree topology and `instance_counts`.
5. WHEN `tree.save(path)` and `FastHierarchyTree.load(path)` are called THEN the system SHALL persist and reconstruct an identical `FastHierarchyTree` with bit-identical NumPy arrays and class index pools.

**Independent Test**: Run `pytest tests/test_trees.py -k "base_tree"` to verify array contiguity, bottom-up instance count invariants, `ValueError` on corrupted parent/child counts, MCTS state reset, and `.npz` round-trip serialization.

---

### P1: Mode A — Native CIFAR-100 Semantic Hierarchy Builder (`ISS-05`) ⭐ MVP

**User Story**: As a researcher benchmarking taxonomic priors (`AD-002`), I want `build_semantic_tree()` (`src/trees/semantic_tree.py`) to build the 3-level CIFAR-100 Semantic Hierarchy (Virtual Root $\rightarrow$ 20 Superclasses $\rightarrow$ Fine Classes) for both Scenario 1 (99 negative classes) and Scenario 2 (100 classes) so that MCTS can navigate the ground-truth human taxonomy.

**Why P1**: Provides the baseline taxonomic hierarchy analogous to GiantHunter's NCBI taxonomy tree (`AD-002`, `EPIC-02`).

**Acceptance Criteria**:

1. WHEN `build_semantic_tree(class_to_indices)` is called with a Scenario 2 mapping of all 100 fine classes (`0..99`) THEN the system SHALL return a valid 3-level `FastHierarchyTree` containing `1` virtual root (`depth=0`), `20` superclass internal nodes (`depth=1`), and `100` fine-class leaf nodes (`depth=2`, `121` total nodes) where each superclass node's children match `CIFAR100Taxonomy.coarse_to_fine_ids`.
2. WHEN `build_semantic_tree(class_to_indices)` is called with a Scenario 1 mapping of 99 negative fine classes (excluding `target_class`) THEN the system SHALL construct a valid 3-level `FastHierarchyTree` with `1` virtual root, `20` superclass nodes (the target class's superclass having `4` children and the remaining `19` superclasses having `5` children), and `99` leaf nodes (`120` total nodes).
3. IF `build_semantic_tree()` receives an empty `class_to_indices` mapping, a class ID outside `0..99`, or any class with an empty sample index array THEN the system SHALL raise a `ValueError`.

**Independent Test**: Run `pytest tests/test_trees.py -k "semantic_tree"` to verify exact node counts (`121` for Scenario 2, `120` for Scenario 1), superclass-to-leaf mappings, and root instance count conservation (`45,000` in Scenario 2; `44,550` in Scenario 1).

---

### P1: Mode B — Data-Driven Ward Confusion Tree & Notebook 02 (`ISS-06`) ⭐ MVP

**User Story**: As a researcher testing whether empirical model confusion outperforms static taxonomy (`AD-002`), I want `build_confusion_tree()` (`src/trees/confusion_tree.py`) and `notebooks/02_tree_construction.ipynb` to compute symmetric confusion distances $D_{i,j}$, build a binary Ward dendrogram tree, and visualize and export both hierarchies on JupyterHub.

**Why P1**: Implements the core novelty of `EPIC-02` (`AD-002`) and its end-to-end JupyterHub orchestration notebook (`AD-008`).

**Acceptance Criteria**:

1. WHEN a confusion matrix $C \in \mathbb{R}^{K \times K}$ is passed to `compute_confusion_distance_matrix(C)` THEN the system SHALL return a symmetric distance matrix $D \in \mathbb{R}^{K \times K}$ in `[0.0, 1.0]` with $D_{i,i} = 0.0$ and off-diagonal entries $D_{i,j} = 1 - \frac{C^{\text{off}}_{i,j} + C^{\text{off}}_{j,i}}{2 \cdot \max(C^{\text{off}})}$ (where $C^{\text{off}}$ has zeroed diagonal).
2. WHEN `build_confusion_tree(confusion_matrix, class_to_indices)` is called for $K$ active classes ($K=99$ in Scenario 1 or $K=100$ in Scenario 2) THEN the system SHALL compute Ward's minimum variance linkage (`scipy.cluster.hierarchy.linkage(condensed_D, method="ward")`) and convert the $(K - 1) \times 4$ linkage matrix into a binary `FastHierarchyTree` with `2K - 1` total nodes (`K` leaves and `K - 1` internal nodes, each internal node having exactly `2` children) where highly confused class pairs are merged at lower linkage heights than unconfused pairs.
3. IF `confusion_matrix` is not a 2D square matrix, contains `NaN`, `Inf`, or negative entries, or contains any row whose sum across all columns is `<= 0` THEN the system SHALL raise a `ValueError` with a descriptive diagnostic message before tree construction.
4. WHEN `notebooks/02_tree_construction.ipynb` is validated or executed THEN the notebook SHALL contain valid `nbformat` v4 structure importing from `src.utils.seed`, `src.data.*`, and `src.trees.*` (adhering to `AD-008`) to construct Semantic and Ward Confusion trees for Scenario 1 and Scenario 2, plot the confusion matrix and Ward dendrogram, and save serialized `.npz` tree artifacts.

**Independent Test**: Run `pytest tests/test_trees.py` to verify distance symmetry, Ward clustering of synthetic confused class pairs into sibling leaves, `ValueError` on zero-sum/NaN confusion matrices, and notebook structure/execution for `notebooks/02_tree_construction.ipynb`.

---

## Edge Cases

- IF a confusion matrix has zero off-diagonal entries ($\max(C^{\text{off}}) == 0$, i.e., a perfect classifier with positive diagonal entries) THEN `compute_confusion_distance_matrix()` SHALL set all off-diagonal distances to `1.0` and diagonal distances to `0.0` without dividing by zero (`NaN`).
- IF a 100$\times$100 confusion matrix is passed to `build_confusion_tree()` alongside a 99-class `class_to_indices` dictionary (Scenario 1) THEN the system SHALL automatically slice the $99 \times 99$ submatrix corresponding to the sorted active class IDs before computing distances and Ward linkage.
- IF `build_semantic_tree()` receives a subset of classes where an entire CIFAR-100 superclass has zero active fine classes THEN the system SHALL omit the empty superclass internal node rather than creating a childless internal node.
- IF `FastHierarchyTree.save(path)` is interrupted THEN the system SHALL write to a temporary `.tmp.npz` file and atomically replace `path` so a corrupted tree archive is never left on disk.

---

## Requirement Traceability

Each requirement maps to a unique ID for tracking across design, tasks, and verification.

| Requirement ID | Story | Phase | Status |
| -------------- | ----- | ----- | ------ |
| TREE-01 | P1: Vectorized Hierarchy Tree Data Structure (`ISS-04`) | Execute | Verified |
| TREE-02 | P1: Mode A — Native CIFAR-100 Semantic Hierarchy Builder (`ISS-05`) | Execute | Verified |
| TREE-03 | P1: Mode B — Data-Driven Ward Confusion Tree & Notebook 02 (`ISS-06`) | Execute | Verified |

**Coverage:** 3 total requirements mapped to `EPIC-02`.

---

## Differences from the source

Every departure from the original *GiantHunter* paper (`btaf239.pdf`) and official repository (`FuchuanQu/GiantHunter`) is linked to an active architectural decision in `.specs/STATE.md`.

| Source shows | Product shows | Confirmed |
| ------------ | ------------- | --------- |
| GiantHunter builds a single static 4-level biological taxonomy tree from NCBI virus taxonomy (`Family -> Genus -> Species`) | Implements both the static 3-level CIFAR-100 Semantic Tree (`Root -> 20 Superclasses -> Fine Classes`) and a data-driven binary Ward Confusion Tree built from validation confusion matrices | y - AD-002 |
| GiantHunter constructs only a binary-classification negative tree (excluding the positive NCLDV phylum) | Supports both Scenario 1 (99 negative fine classes excluding `target_class`) and Scenario 2 (all 100 fine classes for multi-class curriculum) | y - AD-004 |
| GiantHunter omits the training tree data structure from its public repository | Implements a contiguous NumPy-array `FastHierarchyTree` with `.npz` serialization and orchestrates tree construction/visualization via `notebooks/02_tree_construction.ipynb` | y - AD-008 |

## Sources

- `.specs/features/00-epic-roadmap/spec.md` (`EPIC-02`) - parent epic specification and acceptance criteria for Feature `F2`.
- `ROADMAP.md` (`Feature F2`, `ISS-04`–`ISS-06`, `T2.1`–`T2.4`) - planned issue and task decomposition.
- `ARCHITECTURE.md` (`Section 3: Dual Tree Construction Engine`) - mathematical specification of $D_{i,j}$ and Ward linkage conversion.
- `../btaf239.pdf` (*Qu et al., Bioinformatics 2025*) - reference for taxonomic tree construction and bottom-up instance count weighting ($M_c / M_p$).

---

## Success Criteria

- [x] `.venv/bin/pytest -v tests/test_trees.py` passes 100% of unit and integration checks covering `TREE-01`, `TREE-02`, and `TREE-03`.
- [x] Both `build_semantic_tree()` and `build_confusion_tree()` satisfy all `FastHierarchyTree` structural invariants (single root, cycle-free reachability, exact bottom-up instance count sums) across Scenario 1 (`99` leaves) and Scenario 2 (`100` leaves).
- [x] `build_confusion_tree()` provably groups synthetically confused class pairs as sibling leaves at the lowest Ward linkage merge height and rejects zero-sum or `NaN` confusion matrices with `ValueError`.
- [x] `notebooks/02_tree_construction.ipynb` is versioned, valid (`nbformat` v4), and invokes `src.trees.*` cleanly for remote JupyterHub execution.
