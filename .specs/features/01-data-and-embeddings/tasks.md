# F1: Project Scaffold, CIFAR-100 Loader & Embedding Cache Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, the walk, check_proofs, the Verifier and the screens sub-agent).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Design**: `.specs/features/01-data-and-embeddings/design.md`
**Status**: In Progress
**Approved**: 2026-11-08 - user: "prossiga"
**Build**: `.venv/bin/python -m compileall -q src tests`
**Run unit**: `.venv/bin/pytest -v`

---

## Test Coverage Matrix

> Generated from codebase, project guidelines, and spec - confirm before Execute. Guidelines found: `AGENTS.md`, `ROADMAP.md`, `ARCHITECTURE.md`.

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Config / Packaging (`pyproject.toml`) | unit | Package metadata, Python version constraint, and required dependencies verified | `tests/test_seed_and_taxonomy.py` | `.venv/bin/pytest -v tests/test_seed_and_taxonomy.py` |
| Utility (`src/utils/seed.py`) | unit | All branches; 1:1 to `DATA-01` ACs; negative/invalid seed edge cases | `tests/test_seed_and_taxonomy.py` | `.venv/bin/pytest -v tests/test_seed_and_taxonomy.py` |
| Domain / Taxonomy (`src/data/cifar100_taxonomy.py`) | unit | All branches; 1:1 to `DATA-01` ACs; out-of-range fine label edge cases | `tests/test_seed_and_taxonomy.py` | `.venv/bin/pytest -v tests/test_seed_and_taxonomy.py` |
| Data Pipeline / Extraction (`src/data/extract_embeddings.py`) | unit | All branches; 1:1 to `DATA-02` ACs; invalid `val_fraction` and corrupt cache edge cases | `tests/test_extract_embeddings.py` | `.venv/bin/pytest -v tests/test_extract_embeddings.py` |
| Domain / Splitters (`src/data/splits.py`) | unit | All branches; 1:1 to `DATA-03` ACs; out-of-bounds target/ratio and minimum-1 positive clamping | `tests/test_splits_and_notebook.py` | `.venv/bin/pytest -v tests/test_splits_and_notebook.py` |
| Notebook Orchestration (`notebooks/01_data_and_embeddings.ipynb`) | integration | Valid `nbformat` v4 structure, modular `src.*` imports, and end-to-end execution smoke check | `tests/test_splits_and_notebook.py` | `.venv/bin/pytest -v tests/test_splits_and_notebook.py` |

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

### Phase 1: Package Foundation, Seeding & Taxonomy (`ISS-01`, `ISS-02a`)

Tasks that establish the `uv` environment, deterministic RNG seeding, and the CIFAR-100 20-superclass $\leftrightarrow$ 100-fine-class taxonomy.

```
T1 → T2 → T3
```

### Phase 2: Frozen ResNet-18 Embedding Extraction & Cache (`ISS-02b`)

Builds stratified train/val splitting and deterministic `[N, 512]` feature extraction with atomic disk caching.

```
T4
```

### Phase 3: Imbalanced Scenario Splitters & Notebook 01 (`ISS-03`)

Implements Scenario 1 (One-vs-Rest) and Scenario 2 (100-way curriculum) splitters and creates the versioned JupyterHub notebook `notebooks/01_data_and_embeddings.ipynb`.

```
T5 → T6
```

---

## Task Breakdown

### T1: Configure `pyproject.toml` and Package Layout

**What**: Create `pyproject.toml` with project dependencies (`torch`, `torchvision`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `ipykernel`, `nbformat`, `pytest`), initialize `src/` package markers, and add `pyproject.toml` verification tests.
**Where**: `pyproject.toml`
**Depends on**: None
**Reuses**: `AGENTS.md` Section 2 technical stack specification
**Requirement**: DATA-01

**Done when**:

- [x] `pyproject.toml` defines Python $\ge 3.10`, dependencies, and `pytest` configuration with `pythonpath = ["."]`
- [x] `src/__init__.py`, `src/utils/__init__.py`, and `src/data/__init__.py` exist
- [x] Unit tests in `tests/test_seed_and_taxonomy.py` verify `pyproject.toml` metadata and dependencies (`test_pyproject_configuration`)
- [x] Gate check passes: `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v`
- [x] Test count: 1 test passes (no silent deletions)

**Tests**: unit
**Gate**: build
**Commit**: `feat(env): configure pyproject.toml and package layout`

---

### T2: Implement Deterministic Seed Manager

**What**: Implement `set_global_seed()` and `get_torch_generator()` in `src/utils/seed.py` with input validation and unit tests in `tests/test_seed_and_taxonomy.py`.
**Where**: `src/utils/seed.py`
**Depends on**: T1
**Reuses**: `pyproject.toml` test runner setup from T1
**Requirement**: DATA-01

**Done when**:

- [x] `set_global_seed(seed)` seeds `random`, `PYTHONHASHSEED`, `numpy.random`, and `torch` (CPU/CUDA/cuDNN deterministic mode)
- [x] Negative or non-integer seeds (including `bool`) raise `ValueError`
- [x] Unit tests `test_set_global_seed_reproducibility` and `test_set_global_seed_invalid_raises` pass in `tests/test_seed_and_taxonomy.py`
- [x] Gate check passes: `.venv/bin/pytest -v`
- [x] Test count: 3 tests pass (no silent deletions)

**Tests**: unit
**Gate**: quick
**Commit**: `feat(utils): implement deterministic global seed manager`

---

### T3: Implement CIFAR-100 Coarse-to-Fine Taxonomy Module

**What**: Implement `CIFAR100Taxonomy`, `get_cifar100_taxonomy()`, and `fine_to_coarse_tensor()` in `src/data/cifar100_taxonomy.py` with full unit test coverage in `tests/test_seed_and_taxonomy.py`.
**Where**: `src/data/cifar100_taxonomy.py`
**Depends on**: T2
**Reuses**: `src/utils/seed.py`
**Requirement**: DATA-01

**Done when**:

- [x] All 100 fine classes (`0..99`) map bijectively to 20 coarse superclasses (`0..19`, 5 fine classes per superclass)
- [x] `fine_to_coarse_tensor()` converts both `torch.Tensor` and `np.ndarray` inputs while raising `ValueError` on out-of-range labels (`< 0` or `> 99`)
- [x] Unit tests `test_cifar100_taxonomy_bijection`, `test_fine_to_coarse_tensor_conversion`, and `test_fine_to_coarse_out_of_range_raises` pass
- [x] Gate check passes: `.venv/bin/pytest -v`
- [x] Test count: 6 tests pass (no silent deletions)

**Tests**: unit
**Gate**: quick
**Commit**: `feat(data): add CIFAR-100 coarse-to-fine taxonomy lookup`

---

### T4: Implement Frozen ResNet-18 Embedding Extractor and Atomic Cache Manager

**What**: Implement `create_stratified_train_val_indices()`, `build_frozen_resnet18_extractor()`, `extract_cifar100_embeddings()`, and `load_embedding_cache()` in `src/data/extract_embeddings.py` with unit tests in `tests/test_extract_embeddings.py`.
**Where**: `src/data/extract_embeddings.py`
**Depends on**: T3
**Reuses**: `src/utils/seed.py`, `src/data/cifar100_taxonomy.py`
**Requirement**: DATA-02

**Done when**:

- [x] `create_stratified_train_val_indices()` produces disjoint 90/10 stratified train/val index arrays (`45,000` / `5,000` on 50k labels) and raises `ValueError` for invalid `val_fraction`
- [x] `extract_cifar100_embeddings()` extracts `float32` `[N, 512]` embeddings, saves atomically via `.tmp` rename, and reuses existing valid caches when `overwrite=False`
- [x] `load_embedding_cache()` validates required splits (`train`, `val`, `test`), feature dimension (`512`), and absence of `NaN`/`Inf` values, raising `ValueError` on corruption
- [x] Gate check passes: `.venv/bin/pytest -v`
- [x] Test count: 11 tests pass (no silent deletions)

**Tests**: unit
**Gate**: quick
**Commit**: `feat(data): implement frozen ResNet-18 embedding extractor and cache`

---

### T5: Implement Imbalanced Scenario 1 and Scenario 2 Splitters

**What**: Implement `Scenario1Split`, `Scenario2Split`, `build_scenario1_split()`, and `build_scenario2_split()` in `src/data/splits.py` with unit tests in `tests/test_splits_and_notebook.py`.
**Where**: `src/data/splits.py`
**Depends on**: T4
**Reuses**: `src/data/cifar100_taxonomy.py`, `src/utils/seed.py`
**Requirement**: DATA-03

**Done when**:

- [ ] `build_scenario1_split()` deterministically subsamples target class positives ($\max(1, \lfloor \text{pos\_fraction} \times N_{\text{target}} \rfloor)$), retains all 99 negative classes in `negative_class_to_indices`, and raises `ValueError` on invalid `target_class` or `pos_fraction`
- [ ] `build_scenario2_split()` maps all 100 fine classes (`0..99`) to their sample index arrays
- [ ] Unit tests in `tests/test_splits_and_notebook.py` cover `DATA-03` ACs 1–3 and clamping edge cases
- [ ] Gate check passes: `uv run pytest -v`
- [ ] Test count: 16 tests pass (no silent deletions)

**Tests**: unit
**Gate**: quick
**Commit**: `feat(data): add Scenario 1 imbalanced and Scenario 2 multi-class splitters`

---

### T6: Create Versioned `notebooks/01_data_and_embeddings.ipynb` Orchestrator

**What**: Create `notebooks/01_data_and_embeddings.ipynb` calling `src.utils.seed` and `src.data.*` to orchestrate CIFAR-100 download, embedding extraction, cache verification, and Scenario 1/2 split visualizations, plus integration validation in `tests/test_splits_and_notebook.py`.
**Where**: `notebooks/01_data_and_embeddings.ipynb`
**Depends on**: T5
**Reuses**: `src/utils/seed.py`, `src/data/cifar100_taxonomy.py`, `src/data/extract_embeddings.py`, `src/data/splits.py`
**Requirement**: DATA-03

**Done when**:

- [ ] `notebooks/01_data_and_embeddings.ipynb` is a valid `nbformat` v4 notebook with markdown documentation and modular `src.*` code cells
- [ ] Integration test `test_notebook_01_structure_and_imports` in `tests/test_splits_and_notebook.py` verifies `nbformat` validity, required `src.*` imports, and thin-orchestration compliance (`AD-008`)
- [ ] Gate check passes: `uv run python -m compileall -q src tests && uv run pytest -v`
- [ ] Test count: 17 tests pass (no silent deletions)

**Tests**: integration
**Gate**: full
**Commit**: `feat(notebooks): add 01_data_and_embeddings.ipynb orchestrator`

---

## Checks

**C1** - `pyproject.toml` declares Python `>=3.10`, `src` package discovery, and all required scientific, Jupyter, and test dependencies (DATA-01, AC 1)
Proof: run unit | tests/test_seed_and_taxonomy.py::test_pyproject_configuration PASSED

**C2** - `set_global_seed(42)` seeds Python `random`, `PYTHONHASHSEED`, `numpy`, and `torch` with deterministic cuDNN flags so repeated draws are bit-identical (DATA-01, AC 2)
Proof: run unit | tests/test_seed_and_taxonomy.py::test_set_global_seed_reproducibility PASSED

**C3** - `set_global_seed()` raises `ValueError` when called with a negative integer or non-integer value (DATA-01, AC 3)
Proof: run unit | tests/test_seed_and_taxonomy.py::test_set_global_seed_invalid_raises PASSED

**C4** - `get_cifar100_taxonomy()` maps all 100 fine classes (`0..99`) into 20 disjoint superclasses (`0..19`) of exactly 5 fine classes each (DATA-01, AC 4)
Proof: run unit | tests/test_seed_and_taxonomy.py::test_cifar100_taxonomy_bijection PASSED

**C5** - `fine_to_coarse_tensor()` maps 1D fine label tensors/arrays (`0..99`) to coarse labels (`0..19`) with preserved shape/device and raises `ValueError` on out-of-range labels (DATA-01, AC 5)
Proof: run unit | tests/test_seed_and_taxonomy.py::test_fine_to_coarse_tensor_conversion PASSED

**C6** - `fine_to_coarse_tensor()` raises `ValueError` when given fine labels outside `0..99` (DATA-01, Edge Case 1)
Proof: run unit | tests/test_seed_and_taxonomy.py::test_fine_to_coarse_out_of_range_raises PASSED

**C7** - `create_stratified_train_val_indices()` partitions 50,000 labels into 45,000 train (`450`/class) and 5,000 val (`50`/class) disjoint indices and rejects invalid `val_fraction` (DATA-02, AC 1)
Proof: run unit | tests/test_extract_embeddings.py::test_stratified_train_val_indices PASSED

**C8** - `extract_cifar100_embeddings()` extracts `float32` `[N, 512]` embeddings along with `int64` `fine_labels` and `coarse_labels` for `train`, `val`, and `test` and saves atomically to `cache_path` (DATA-02, AC 2)
Proof: run unit | tests/test_extract_embeddings.py::test_extract_cifar100_embeddings_creates_cache PASSED

**C9** - `extract_cifar100_embeddings()` with `overwrite=False` validates and reuses an existing cache bundle without re-running backbone extraction (DATA-02, AC 3)
Proof: run unit | tests/test_extract_embeddings.py::test_extract_cifar100_embeddings_reuses_valid_cache PASSED

**C10** - `load_embedding_cache()` raises `ValueError` when a cache file is missing split keys, has wrong embedding dimension, or contains `NaN`/`Inf` values (DATA-02, AC 4)
Proof: run unit | tests/test_extract_embeddings.py::test_load_embedding_cache_corrupted_raises PASSED

**C11** - `build_scenario1_split()` deterministically subsamples $\max(1, \lfloor \text{pos\_fraction} \times N_{\text{target}} \rfloor)$ positive indices while preserving all 99 negative classes in `negative_class_to_indices` (DATA-03, AC 1)
Proof: run unit | tests/test_splits_and_notebook.py::test_build_scenario1_split_ratios_and_clamping PASSED

**C12** - `build_scenario1_split()` raises `ValueError` when `target_class` is outside `0..99` or `pos_fraction` is outside `(0, 1]` (DATA-03, AC 2)
Proof: run unit | tests/test_splits_and_notebook.py::test_build_scenario1_split_invalid_args_raise PASSED

**C13** - `build_scenario2_split()` maps all 100 fine classes (`0..99`) to their exact sample index arrays with zero dropped classes (DATA-03, AC 3)
Proof: run unit | tests/test_splits_and_notebook.py::test_build_scenario2_split_complete PASSED

**C14** - `notebooks/01_data_and_embeddings.ipynb` is a valid `nbformat` v4 notebook that imports from `src.utils.seed` and `src.data.*` to orchestrate download, extraction, and Scenario 1/2 split visualizations (DATA-03, AC 4)
Proof: run unit | tests/test_splits_and_notebook.py::test_notebook_01_structure_and_imports PASSED

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
| T1: Configure `pyproject.toml` and Package Layout | 1 config file + empty `__init__.py` markers | ✅ Granular |
| T2: Implement Deterministic Seed Manager | 1 utility module (`src/utils/seed.py`) | ✅ Granular |
| T3: Implement CIFAR-100 Coarse-to-Fine Taxonomy Module | 1 domain module (`src/data/cifar100_taxonomy.py`) | ✅ Granular |
| T4: Implement Frozen ResNet-18 Embedding Extractor and Atomic Cache Manager | 1 pipeline module (`src/data/extract_embeddings.py`) | ✅ Granular |
| T5: Implement Imbalanced Scenario 1 and Scenario 2 Splitters | 1 splitter module (`src/data/splits.py`) | ✅ Granular |
| T6: Create Versioned `notebooks/01_data_and_embeddings.ipynb` Orchestrator | 1 Jupyter notebook (`notebooks/01_data_and_embeddings.ipynb`) | ✅ Granular |

---

## Diagram-Definition Cross-Check

| Task | Depends On (task body) | Diagram Shows | Status |
| ---- | ---------------------- | ------------- | ------ |
| T1 | None | Phase 1 start (`T1`) | ✅ Match |
| T2 | T1 | `T1 → T2` | ✅ Match |
| T3 | T2 | `T2 → T3` | ✅ Match |
| T4 | T3 | Phase 1 (`T3`) → Phase 2 (`T4`) | ✅ Match |
| T5 | T4 | Phase 2 (`T4`) → Phase 3 (`T5`) | ✅ Match |
| T6 | T5 | `T5 → T6` | ✅ Match |

---

## Test Co-location Validation

| Task | Code Layer Created/Modified | Matrix Requires | Task Says | Status |
| ---- | --------------------------- | --------------- | --------- | ------ |
| T1: Configure `pyproject.toml` and Package Layout | Config / Packaging | unit | unit | ✅ OK |
| T2: Implement Deterministic Seed Manager | Utility (`src/utils/seed.py`) | unit | unit | ✅ OK |
| T3: Implement CIFAR-100 Coarse-to-Fine Taxonomy Module | Domain / Taxonomy (`src/data/cifar100_taxonomy.py`) | unit | unit | ✅ OK |
| T4: Implement Frozen ResNet-18 Embedding Extractor and Atomic Cache Manager | Data Pipeline (`src/data/extract_embeddings.py`) | unit | unit | ✅ OK |
| T5: Implement Imbalanced Scenario 1 and Scenario 2 Splitters | Domain / Splitters (`src/data/splits.py`) | unit | unit | ✅ OK |
| T6: Create Versioned `notebooks/01_data_and_embeddings.ipynb` Orchestrator | Notebook Orchestration (`notebooks/01_data_and_embeddings.ipynb`) | integration | integration | ✅ OK |
