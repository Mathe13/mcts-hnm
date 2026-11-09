# F2: Dual Hierarchy Construction Engine Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, the walk, check_proofs, the Verifier and the screens sub-agent).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Design**: `.specs/features/02-dual-hierarchy-trees/design.md`
**Status**: Done
**Approved**: 2026-11-08 - user: "prossiga"
**Build**: `.venv/bin/python -m compileall -q src tests`
**Run unit**: `.venv/bin/pytest -v`

---

## Test Coverage Matrix

> Generated from codebase, project guidelines, and spec - confirm before Execute. Guidelines found: `AGENTS.md`, `ROADMAP.md`, `ARCHITECTURE.md`.

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Core Data Structure (`src/trees/base_tree.py`) | unit | All branches; 1:1 to `TREE-01` ACs; invariant violations, cycle/disconnect detection, MCTS state reset, and `.npz` round-trip serialization | `tests/test_trees.py` | `.venv/bin/pytest -v tests/test_trees.py` |
| Domain / Semantic Tree (`src/trees/semantic_tree.py`) | unit | All branches; 1:1 to `TREE-02` ACs; Scenario 1 (99 leaves), Scenario 2 (100 leaves), empty superclass pruning, and invalid input edge cases | `tests/test_trees.py` | `.venv/bin/pytest -v tests/test_trees.py` |
| Domain / Ward Confusion Tree (`src/trees/confusion_tree.py`) | unit | All branches; 1:1 to `TREE-03` ACs 1–3; distance symmetry, zero off-diagonal handling, 100->99 slicing, Ward sibling grouping, and zero-row/NaN `ValueError` checks | `tests/test_trees.py` | `.venv/bin/pytest -v tests/test_trees.py` |
| Notebook Orchestration (`notebooks/02_tree_construction.ipynb`) | integration | Valid `nbformat` v4 structure, modular `src.*` imports, and end-to-end execution smoke check (`TREE-03` AC 4, `AD-008`) | `tests/test_trees.py` | `.venv/bin/pytest -v tests/test_trees.py` |

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

### Phase 1: Vectorized Tree Core & Semantic Hierarchy (`ISS-04`, `ISS-05`)

Tasks that establish the contiguous-array `FastHierarchyTree` container and the static CIFAR-100 Semantic Hierarchy builder.

```
T1 → T2
```

### Phase 2: Ward Confusion Hierarchy & Notebook 02 Orchestrator (`ISS-06`)

Implements empirical confusion distance computation, Ward hierarchical linkage tree construction, and the versioned JupyterHub notebook `notebooks/02_tree_construction.ipynb`.

```
T3 → T4
```

---

## Task Breakdown

### T1: Implement `FastHierarchyTree` Data Structure and `.npz` Serialization

**What**: Implement `FastHierarchyTree` in `src/trees/base_tree.py` (and package exports in `src/trees/__init__.py`) with contiguous NumPy arrays, `validate_tree()`, `reset_mcts_state()`, and atomic `.npz` `save()`/`load()` methods, plus unit tests in `tests/test_trees.py`.
**Where**: `src/trees/base_tree.py`
**Depends on**: None
**Reuses**: Atomic `.tmp` persistence pattern from `src/data/extract_embeddings.py`
**Requirement**: TREE-01

**Done when**:

- [x] `FastHierarchyTree` stores contiguous NumPy arrays (`parents`, `children`, `child_counts`, `is_leaf`, `depths`, `leaf_to_class_id`, `instance_counts`, `visits`, `q_values`) and exposes `get_children()`, `leaf_node_ids()`, and `class_id_to_leaf_node_id()`
- [x] `validate_tree()` verifies single root (`parents[root_id] == -1`), cycle-free reachability of all nodes, leaf/internal invariants, and exact parent instance count conservation ($M_p = \sum M_c$), raising `ValueError` on any violation
- [x] `reset_mcts_state()` resets `visits` to `0` and `q_values` to `0.0` in-place, and `save()`/`load()` persist and restore bit-identical trees via atomic `.tmp.npz` rename
- [x] Unit tests `test_base_tree_contiguous_arrays_and_invariants`, `test_base_tree_validate_tree_reachability_and_counts`, `test_base_tree_invalid_invariants_raise`, `test_base_tree_reset_mcts_state`, and `test_base_tree_save_and_load_roundtrip` pass in `tests/test_trees.py`
- [x] Gate check passes: `.venv/bin/pytest -v`
- [x] Test count: 22 tests pass (17 from F1 + 5 new in `tests/test_trees.py`)

**Tests**: unit
**Gate**: quick
**Commit**: `feat(trees): implement vectorized FastHierarchyTree and npz serialization`

---

### T2: Implement Mode A — Native CIFAR-100 Semantic Hierarchy Builder

**What**: Implement `build_semantic_tree()` in `src/trees/semantic_tree.py` mapping Virtual Root $\rightarrow$ 20 Superclasses $\rightarrow$ Fine Classes for Scenario 1 (99 negative classes) and Scenario 2 (100 classes), with unit tests in `tests/test_trees.py`.
**Where**: `src/trees/semantic_tree.py`
**Depends on**: T1
**Reuses**: `src/data/cifar100_taxonomy.py` (`get_cifar100_taxonomy`), `src/trees/base_tree.py` (`FastHierarchyTree`)
**Requirement**: TREE-02

**Done when**:

- [x] `build_semantic_tree(class_to_indices)` constructs a 3-level `FastHierarchyTree` with `121` nodes (`1` root, `20` superclasses, `100` leaves) for Scenario 2 and `120` nodes (`1` root, `20` superclasses, `99` leaves) for Scenario 1
- [x] Empty superclasses are omitted when a subset of classes is provided, and invalid class IDs or empty index arrays raise `ValueError`
- [x] Unit tests `test_semantic_tree_scenario2_100_classes`, `test_semantic_tree_scenario1_99_negative_classes`, and `test_semantic_tree_invalid_inputs_and_empty_superclass` pass in `tests/test_trees.py`
- [x] Gate check passes: `.venv/bin/pytest -v`
- [x] Test count: 25 tests pass (no silent deletions)

**Tests**: unit
**Gate**: quick
**Commit**: `feat(trees): add native CIFAR-100 semantic hierarchy builder`

---

### T3: Implement Mode B — Data-Driven Ward Confusion Tree Builder

**What**: Implement `validate_confusion_matrix()`, `compute_confusion_distance_matrix()`, `build_confusion_tree()`, and `estimate_validation_confusion_matrix()` in `src/trees/confusion_tree.py` with unit tests in `tests/test_trees.py`.
**Where**: `src/trees/confusion_tree.py`
**Depends on**: T2
**Reuses**: `src/trees/base_tree.py` (`FastHierarchyTree`), `src/data/cifar100_taxonomy.py`, `src/utils/seed.py`
**Requirement**: TREE-03

**Done when**:

- [x] `compute_confusion_distance_matrix(C)` validates $C$ (rejecting non-square, negative, `NaN`/`Inf`, or zero-sum rows with `ValueError`) and returns symmetric $D \in [0, 1]^{K \times K}$ with $D_{i,i} = 0.0$ and zero-off-diagonal fallback $D_{i,j} = 1.0$
- [x] `build_confusion_tree()` slices $100 \times 100$ matrices to active classes when $K < 100$, runs `scipy.cluster.hierarchy.linkage(..., method="ward")`, and builds a binary `FastHierarchyTree` (`2K - 1` nodes, `root_id = 0`) that merges confused class pairs at lowest linkage height
- [x] `estimate_validation_confusion_matrix()` deterministically estimates a valid confusion matrix from train/val feature embeddings
- [x] Unit tests `test_confusion_distance_matrix_symmetry_and_zero_off_diag`, `test_build_confusion_tree_ward_hierarchy_and_slicing`, and `test_confusion_matrix_invalid_inputs_raise` pass in `tests/test_trees.py`
- [x] Gate check passes: `.venv/bin/pytest -v`
- [x] Test count: 28 tests pass (no silent deletions)

**Tests**: unit
**Gate**: quick
**Commit**: `feat(trees): implement Ward confusion hierarchy builder and distance metrics`

---

### T4: Create Versioned `notebooks/02_tree_construction.ipynb` Orchestrator

**What**: Create `notebooks/02_tree_construction.ipynb` calling `src.utils.seed`, `src.data.*`, and `src.trees.*` to build Semantic and Ward Confusion trees for Scenario 1 and Scenario 2, visualize the confusion matrix and Ward dendrogram, and export `.npz` tree artifacts, with integration validation in `tests/test_trees.py`.
**Where**: `notebooks/02_tree_construction.ipynb`
**Depends on**: T3
**Reuses**: `src/utils/seed.py`, `src/data/splits.py`, `src/trees/base_tree.py`, `src/trees/semantic_tree.py`, `src/trees/confusion_tree.py`
**Requirement**: TREE-03

**Done when**:

- [x] `notebooks/02_tree_construction.ipynb` is a valid `nbformat` v4 notebook with markdown documentation and thin code cells calling `src.trees.*`
- [x] Integration test `test_notebook_02_structure_and_execution` in `tests/test_trees.py` verifies `nbformat` validity, required `src.*` imports, and end-to-end execution of the tree construction pipeline
- [x] Gate check passes: `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v`
- [x] Test count: 29 tests pass (no silent deletions)

**Tests**: integration
**Gate**: full
**Commit**: `feat(notebooks): add 02_tree_construction.ipynb orchestrator`

---

## Checks

**C1** - `FastHierarchyTree` stores `parents`, `children`, `child_counts`, `is_leaf`, `depths`, `leaf_to_class_id`, `instance_counts`, `visits`, and `q_values` in contiguous NumPy arrays with exact dtypes (TREE-01, AC 1)
Proof: run unit | tests/test_trees.py::test_base_tree_contiguous_arrays_and_invariants PASSED

**C2** - `validate_tree()` verifies single root (`parents[root_id] == -1`), cycle-free reachability of all nodes, and exact parent instance count conservation $M_p = \sum M_c$ (TREE-01, AC 2)
Proof: run unit | tests/test_trees.py::test_base_tree_validate_tree_reachability_and_counts PASSED

**C3** - `FastHierarchyTree` raises `ValueError` when constructed or validated with a disconnected node, cycle, non-positive leaf `instance_count`, or mismatched parent `instance_count` (TREE-01, AC 3)
Proof: run unit | tests/test_trees.py::test_base_tree_invalid_invariants_raise PASSED

**C4** - `reset_mcts_state()` zeros out `visits` (`0`) and `q_values` (`0.0`) in-place while preserving tree topology and `instance_counts` (TREE-01, AC 4)
Proof: run unit | tests/test_trees.py::test_base_tree_reset_mcts_state PASSED

**C5** - `tree.save(path)` and `FastHierarchyTree.load(path)` atomically persist and reconstruct a bit-identical `FastHierarchyTree` and its `class_to_indices` pools (TREE-01, AC 5)
Proof: run unit | tests/test_trees.py::test_base_tree_save_and_load_roundtrip PASSED

**C6** - `build_semantic_tree()` on a Scenario 2 100-class mapping builds a 3-level `FastHierarchyTree` with `1` root, `20` superclasses, and `100` leaves (`121` total nodes) matching `CIFAR100Taxonomy` (TREE-02, AC 1)
Proof: run unit | tests/test_trees.py::test_semantic_tree_scenario2_100_classes PASSED

**C7** - `build_semantic_tree()` on a Scenario 1 99-negative-class mapping builds a 3-level `FastHierarchyTree` with `1` root, `20` superclasses (`4` leaves in target superclass, `5` in remaining `19`), and `99` leaves (`120` total nodes) (TREE-02, AC 2)
Proof: run unit | tests/test_trees.py::test_semantic_tree_scenario1_99_negative_classes PASSED

**C8** - `build_semantic_tree()` raises `ValueError` on empty `class_to_indices`, out-of-range class IDs, or empty class index arrays, and omits empty superclasses when given a partial class subset (TREE-02, AC 3)
Proof: run unit | tests/test_trees.py::test_semantic_tree_invalid_inputs_and_empty_superclass PASSED

**C9** - `compute_confusion_distance_matrix(C)` returns a symmetric matrix $D \in [0, 1]^{K \times K}$ with $D_{i,i} = 0.0$ and off-diagonal $D_{i,j} = 1 - \frac{C^{\text{off}}_{i,j} + C^{\text{off}}_{j,i}}{2 \cdot \max(C^{\text{off}})}$, handling zero off-diagonal matrices without `NaN` (TREE-03, AC 1)
Proof: run unit | tests/test_trees.py::test_confusion_distance_matrix_symmetry_and_zero_off_diag PASSED

**C10** - `build_confusion_tree()` builds a binary Ward linkage `FastHierarchyTree` with `2K - 1` nodes (`K` leaves, `K - 1` internal nodes with `2` children each), slices $100 \times 100$ matrices for Scenario 1 (`K=99`), and merges highly confused class pairs as sibling leaves (TREE-03, AC 2)
Proof: run unit | tests/test_trees.py::test_build_confusion_tree_ward_hierarchy_and_slicing PASSED

**C11** - `compute_confusion_distance_matrix()` and `build_confusion_tree()` raise `ValueError` when `confusion_matrix` is non-square, contains `NaN`, `Inf`, or negative values, or has any zero-sum row (TREE-03, AC 3)
Proof: run unit | tests/test_trees.py::test_confusion_matrix_invalid_inputs_raise PASSED

**C12** - `notebooks/02_tree_construction.ipynb` is a valid `nbformat` v4 notebook that imports from `src.utils.seed`, `src.data.*`, and `src.trees.*` to construct both trees, plot heatmaps/dendrograms, and export `.npz` artifacts (TREE-03, AC 4)
Proof: run unit | tests/test_trees.py::test_notebook_02_structure_and_execution PASSED

---

## Handoff

The feature's live state, written as it happens - a compaction gives no warning.

**Settled mid-build:** none
**Dispatched:** Independent Verifier (`6be1ce0e-a49c-4a65-9002-4b6c20ed04da`) at `efac279` - returned `PASS`
**Walked:** Executed all 12 checks in `tests/test_trees.py` (29 total tests passing across repository) and ran `notebooks/02_tree_construction.ipynb` end-to-end to verify Semantic (`121`/`120` nodes) and Ward Confusion (`199`/`197` nodes) tree construction and `.npz` export.

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
| T1: Implement `FastHierarchyTree` Data Structure and `.npz` Serialization | 1 core module (`src/trees/base_tree.py`) | ✅ Granular |
| T2: Implement Mode A — Native CIFAR-100 Semantic Hierarchy Builder | 1 domain module (`src/trees/semantic_tree.py`) | ✅ Granular |
| T3: Implement Mode B — Data-Driven Ward Confusion Tree Builder | 1 domain module (`src/trees/confusion_tree.py`) | ✅ Granular |
| T4: Create Versioned `notebooks/02_tree_construction.ipynb` Orchestrator | 1 Jupyter notebook (`notebooks/02_tree_construction.ipynb`) | ✅ Granular |

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
| T1: Implement `FastHierarchyTree` Data Structure and `.npz` Serialization | Core Data Structure (`src/trees/base_tree.py`) | unit | unit | ✅ OK |
| T2: Implement Mode A — Native CIFAR-100 Semantic Hierarchy Builder | Domain / Semantic Tree (`src/trees/semantic_tree.py`) | unit | unit | ✅ OK |
| T3: Implement Mode B — Data-Driven Ward Confusion Tree Builder | Domain / Ward Confusion Tree (`src/trees/confusion_tree.py`) | unit | unit | ✅ OK |
| T4: Create Versioned `notebooks/02_tree_construction.ipynb` Orchestrator | Notebook Orchestration (`notebooks/02_tree_construction.ipynb`) | integration | integration | ✅ OK |
