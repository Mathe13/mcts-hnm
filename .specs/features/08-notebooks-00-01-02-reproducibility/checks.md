# Notebooks 00, 01, and 02 Reproducibility - Checks

Profile: light
Plan: `.specs/features/08-notebooks-00-01-02-reproducibility/plan.md`

## Intent

12 checks in 3 slices · 4 one-way doors · 0 open

## Checks

### S1 - Notebook 01 Data Pipeline Isolation · 3 files · 24 KB · ~8k

**C1** - `extract_or_generate_embeddings` in REAL mode raises `FileNotFoundError` when CIFAR-100 is absent (S1, AC 1)
Proof: `pytest tests/test_notebooks_00_01_02.py::test_extract_embeddings_real_mode_missing_cifar_raises`

**C2** - `extract_or_generate_embeddings` in SMOKE mode generates isolated synthetic bundle with metadata (S1, AC 2, AC 3)
Proof: `pytest tests/test_notebooks_00_01_02.py::test_extract_embeddings_smoke_mode_synthetic_generation`

**C3** - Notebook 01 initial cell declares `EXECUTION_MODE = ExecutionMode.REAL` and outputs banner (S1, AC 4)
Proof: `pytest tests/test_notebooks_00_01_02.py::test_notebook_01_structure_and_banner`

### S2 - Notebook 02 Tree Construction and Methodological Linkage · 3 files · 32 KB · ~11k

**C4** - Notebook 02 in REAL mode rejects SMOKE/synthetic embeddings with status 422 (S2, AC 1)
Proof: `pytest tests/test_notebooks_00_01_02.py::test_notebook_02_rejects_synthetic_in_real`

**C5** - `build_confusion_tree` with `linkage_method="ward"` computes Ward clustering matching baseline (S2, AC 2)
Proof: `pytest tests/test_notebooks_00_01_02.py::test_build_confusion_tree_ward_linkage`

**C6** - `build_confusion_tree` with `linkage_method="average"` computes valid Average linkage hierarchy (S2, AC 3)
Proof: `pytest tests/test_notebooks_00_01_02.py::test_build_confusion_tree_average_linkage`

**C7** - Serialized trees into `embeddings/{mode}/trees/` preserve node counts, leaf indices, and non-empty instances (S2, AC 4, AC 5)
Proof: `pytest tests/test_notebooks_00_01_02.py::test_tree_serialization_and_integrity_in_isolated_dirs`

**C8** - Notebook 02 initial cell declares `EXECUTION_MODE = ExecutionMode.REAL` and outputs banner (S2, AC 4)
Proof: `pytest tests/test_notebooks_00_01_02.py::test_notebook_02_structure_and_banner`

### S3 - Notebook 00 Master Pipeline Orchestration · 3 files · 28 KB · ~10k

**C9** - Notebook 00 initial cell declares `EXECUTION_MODE = ExecutionMode.REAL` with banner and eliminates FAST_MODE (S3, AC 1)
Proof: `pytest tests/test_notebooks_00_01_02.py::test_notebook_00_structure_and_contract`

**C10** - Notebook 00 executes all stages in SMOKE mode without writing to REAL directories (S3, AC 1)
Proof: `pytest tests/test_notebooks_00_01_02.py::test_notebook_00_smoke_execution`

**C11** - Notebook 00 in REAL mode halts with explicit error when real dependencies are missing (S3, AC 2)
Proof: `pytest tests/test_notebooks_00_01_02.py::test_notebook_00_real_mode_missing_dependencies_raises`

**C12** - Mid-pipeline attempt to mutate `ExecutionMode` raises `IllegalStateError` (S3, AC 3)
Proof: `pytest tests/test_notebooks_00_01_02.py::test_mid_pipeline_mode_mutation_raises`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| `extract_or_generate_embeddings()` statuses (3) | 200 C2 · 404 C1 · 422 C4 | - |
| `build_confusion_tree()` statuses (2) | 200 C5 · 422 C6 | - |
| `execute_notebook()` statuses (2) | 200 C10 · 400 C11 | - |
| `linkage_method` options (2) | `ward` C5 · `average` C6 | - |
| notebook mode selectors (3) | notebook 00 C9 · notebook 01 C3 · notebook 02 C8 | - |

- Claims naming a status code, route or response shape: C1, C2, C4, C5, C6, C10, C11 - each has a proof that crosses the boundary
- No other check claims more than the single case its proof exercises

## Swept

- validation: C1, C4
- failure modes: C1, C11
- idempotency: C7
- authorization: n/a - local research experimentation without user auth roles
- concurrency: C12
- data lifecycle: C7
- dependency failure: C1, C11
- state transitions: C12
- observability: C3, C8, C9

## Handoff

- S1-S3 = 84 KB total across data, trees, notebooks, and tests; estimate ~29k tokens, well under the 150k budget - one builder
