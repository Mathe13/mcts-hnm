# 08-notebooks-00-01-02-reproducibility verification

**Verdict**: PASS
**Profile**: light
**Diff range**: origin/main..7296939
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | `extract_or_generate_embeddings` in REAL mode raises `FileNotFoundError` when CIFAR-100 is absent (S1, AC 1) | `pytest tests/test_notebooks_00_01_02.py::test_extract_embeddings_real_mode_missing_cifar_raises` exit 0 | `tests/test_notebooks_00_01_02.py:49` - `with pytest.raises(FileNotFoundError, match="CIFAR-100 raw files not found"):` | PASS |
| C2 | `extract_or_generate_embeddings` in SMOKE mode generates isolated synthetic bundle with metadata (S1, AC 2, AC 3) | `pytest tests/test_notebooks_00_01_02.py::test_extract_embeddings_smoke_mode_synthetic_generation` exit 0 | `tests/test_notebooks_00_01_02.py:70` - `assert cache_path.is_file()` and `tests/test_notebooks_00_01_02.py:85` - `assert meta["synthetic_data"] is True` | PASS |
| C3 | Notebook 01 initial cell declares `EXECUTION_MODE = ExecutionMode.REAL` and outputs banner (S1, AC 4) | `pytest tests/test_notebooks_00_01_02.py::test_notebook_01_structure_and_banner` exit 0 | `tests/test_notebooks_00_01_02.py:99` - `assert "EXECUTION_MODE = ExecutionMode.REAL" in first_code_src` | PASS |
| C4 | Notebook 02 in REAL mode rejects SMOKE/synthetic embeddings with status 422 (S2, AC 1) | `pytest tests/test_notebooks_00_01_02.py::test_notebook_02_rejects_synthetic_in_real` exit 0 | `tests/test_notebooks_00_01_02.py:115` - `with pytest.raises(ValueError, match="REAL mode rejected synthetic/smoke artifact"):` | PASS |
| C5 | `build_confusion_tree` with `linkage_method="ward"` computes Ward clustering matching baseline (S2, AC 2) | `pytest tests/test_notebooks_00_01_02.py::test_build_confusion_tree_ward_linkage` exit 0 | `tests/test_notebooks_00_01_02.py:134` - `assert tree_ward.num_nodes == 19` | PASS |
| C6 | `build_confusion_tree` with `linkage_method="average"` computes valid Average linkage hierarchy (S2, AC 3) | `pytest tests/test_notebooks_00_01_02.py::test_build_confusion_tree_average_linkage` exit 0 | `tests/test_notebooks_00_01_02.py:153` - `assert tree_avg.num_nodes == 19` and `tests/test_notebooks_00_01_02.py:158` - `with pytest.raises(ValueError, match="Unsupported linkage_method"):` | PASS |
| C7 | Serialized trees into `embeddings/{mode}/trees/` preserve node counts, leaf indices, and non-empty instances (S2, AC 4, AC 5) | `pytest tests/test_notebooks_00_01_02.py::test_tree_serialization_and_integrity_in_isolated_dirs` exit 0 | `tests/test_notebooks_00_01_02.py:178` - `assert reloaded.num_nodes == tree.num_nodes` | PASS |
| C8 | Notebook 02 initial cell declares `EXECUTION_MODE = ExecutionMode.REAL` and outputs banner (S2, AC 4) | `pytest tests/test_notebooks_00_01_02.py::test_notebook_02_structure_and_banner` exit 0 | `tests/test_notebooks_00_01_02.py:194` - `assert "EXECUTION_MODE = ExecutionMode.REAL" in first_code_src` | PASS |
| C9 | Notebook 00 initial cell declares `EXECUTION_MODE = ExecutionMode.REAL` with banner and eliminates FAST_MODE (S3, AC 1) | `pytest tests/test_notebooks_00_01_02.py::test_notebook_00_structure_and_contract` exit 0 | `tests/test_notebooks_00_01_02.py:208` - `assert "EXECUTION_MODE = ExecutionMode.REAL" in first_code_src` | PASS |
| C10 | Notebook 00 executes all stages in SMOKE mode without writing to REAL directories (S3, AC 1) | `pytest tests/test_notebooks_00_01_02.py::test_notebook_00_smoke_execution` exit 0 | `tests/test_notebooks_00_01_02.py:219` - `assert "smoke" in str(paths.results_dir)` | PASS |
| C11 | Notebook 00 in REAL mode halts with explicit error when real dependencies are missing (S3, AC 2) | `pytest tests/test_notebooks_00_01_02.py::test_notebook_00_real_mode_missing_dependencies_raises` exit 0 | `tests/test_notebooks_00_01_02.py:231` - `with pytest.raises(FileNotFoundError, match="Real CIFAR-100 embeddings missing"):` | PASS |
| C12 | Mid-pipeline attempt to mutate `ExecutionMode` raises `IllegalStateError` (S3, AC 3) | `pytest tests/test_notebooks_00_01_02.py::test_mid_pipeline_mode_mutation_raises` exit 0 | `tests/test_notebooks_00_01_02.py:239` - `with pytest.raises(IllegalStateError, match="Cannot change active session mode"):` | PASS |

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| `extract_or_generate_embeddings()` statuses (3) | 200 C2 · 404 C1 · 422 C4 | - |
| `build_confusion_tree()` statuses (2) | 200 C5 · 422 C6 | - |
| `execute_notebook()` statuses (2) | 200 C10 · 400 C11 | - |
| `linkage_method` options (2) | `ward` C5 · `average` C6 | - |
| notebook mode selectors (3) | notebook 00 C9 · notebook 01 C3 · notebook 02 C8 | - |

## Gate

`.venv/bin/pytest` - 99 passed, 0 failed
