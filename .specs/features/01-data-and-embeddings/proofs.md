# 01-data-and-embeddings proofs

**HEAD**: 7dd4cdc
**Tree**: clean
**Shell**: /bin/sh
**Result**: 15 pass, 0 fail

| Check | Proof | Result |
| --- | --- | --- |
| C1 | `run unit / tests/test_seed_and_taxonomy.py::test_pyproject_configuration PASSED` | pass |
| C2 | `run unit / tests/test_seed_and_taxonomy.py::test_set_global_seed_reproducibility PASSED` | pass |
| C3 | `run unit / tests/test_seed_and_taxonomy.py::test_set_global_seed_invalid_raises PASSED` | pass |
| C4 | `run unit / tests/test_seed_and_taxonomy.py::test_cifar100_taxonomy_bijection PASSED` | pass |
| C5 | `run unit / tests/test_seed_and_taxonomy.py::test_fine_to_coarse_tensor_conversion PASSED` | pass |
| C6 | `run unit / tests/test_seed_and_taxonomy.py::test_fine_to_coarse_out_of_range_raises PASSED` | pass |
| C7 | `run unit / tests/test_extract_embeddings.py::test_stratified_train_val_indices PASSED` | pass |
| C8 | `run unit / tests/test_extract_embeddings.py::test_extract_cifar100_embeddings_creates_cache PASSED` | pass |
| C9 | `run unit / tests/test_extract_embeddings.py::test_extract_cifar100_embeddings_reuses_valid_cache PASSED` | pass |
| C10 | `run unit / tests/test_extract_embeddings.py::test_load_embedding_cache_corrupted_raises PASSED` | pass |
| C11 | `run unit / tests/test_splits_and_notebook.py::test_build_scenario1_split_ratios_and_clamping PASSED` | pass |
| C12 | `run unit / tests/test_splits_and_notebook.py::test_build_scenario1_split_invalid_args_raise PASSED` | pass |
| C13 | `run unit / tests/test_splits_and_notebook.py::test_build_scenario2_split_complete PASSED` | pass |
| C14 | `run unit / tests/test_splits_and_notebook.py::test_notebook_01_structure_and_imports PASSED` | pass |
| Build | `.venv/bin/python -m compileall -q src tests` | pass |

## Output

`Run unit` - exit 0

```
tests/test_extract_embeddings.py::test_load_embedding_cache_corrupted_raises PASSED [ 29%]
tests/test_seed_and_taxonomy.py::test_pyproject_configuration PASSED     [ 35%]
tests/test_seed_and_taxonomy.py::test_set_global_seed_reproducibility PASSED [ 41%]
tests/test_seed_and_taxonomy.py::test_set_global_seed_invalid_raises PASSED [ 47%]
tests/test_seed_and_taxonomy.py::test_cifar100_taxonomy_bijection PASSED [ 52%]
tests/test_seed_and_taxonomy.py::test_fine_to_coarse_tensor_conversion PASSED [ 58%]
tests/test_seed_and_taxonomy.py::test_fine_to_coarse_out_of_range_raises PASSED [ 64%]
tests/test_splits_and_notebook.py::test_build_scenario1_split_ratios_and_clamping PASSED [ 70%]
tests/test_splits_and_notebook.py::test_build_scenario1_split_by_name_and_seed_determinism PASSED [ 76%]
tests/test_splits_and_notebook.py::test_build_scenario1_split_invalid_args_raise PASSED [ 82%]
tests/test_splits_and_notebook.py::test_build_scenario2_split_complete PASSED [ 88%]
tests/test_splits_and_notebook.py::test_build_scenario2_split_missing_class_raises PASSED [ 94%]
tests/test_splits_and_notebook.py::test_notebook_01_structure_and_imports PASSED [100%]

============================== 17 passed in 3.27s ==============================
```
