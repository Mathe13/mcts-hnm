# 01-data-and-embeddings verification

**Verdict**: PASS
**Proofs**: `proofs.md` at 7dd4cdc
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Test | Holds |
| --- | --- | --- |
| C1 | `tests/test_seed_and_taxonomy.py::test_pyproject_configuration` | yes |
| C2 | `tests/test_seed_and_taxonomy.py::test_set_global_seed_reproducibility` | yes |
| C3 | `tests/test_seed_and_taxonomy.py::test_set_global_seed_invalid_raises` | yes |
| C4 | `tests/test_seed_and_taxonomy.py::test_cifar100_taxonomy_bijection` | yes |
| C5 | `tests/test_seed_and_taxonomy.py::test_fine_to_coarse_tensor_conversion` | yes |
| C6 | `tests/test_seed_and_taxonomy.py::test_fine_to_coarse_out_of_range_raises` | yes |
| C7 | `tests/test_extract_embeddings.py::test_stratified_train_val_indices` | yes |
| C8 | `tests/test_extract_embeddings.py::test_extract_cifar100_embeddings_creates_cache` | yes |
| C9 | `tests/test_extract_embeddings.py::test_extract_cifar100_embeddings_reuses_valid_cache` | yes |
| C10 | `tests/test_extract_embeddings.py::test_load_embedding_cache_corrupted_raises` | yes |
| C11 | `tests/test_splits_and_notebook.py::test_build_scenario1_split_ratios_and_clamping` | yes |
| C12 | `tests/test_splits_and_notebook.py::test_build_scenario1_split_invalid_args_raise` | yes |
| C13 | `tests/test_splits_and_notebook.py::test_build_scenario2_split_complete` | yes |
| C14 | `tests/test_splits_and_notebook.py::test_notebook_01_structure_and_imports` | yes |

## Findings

None - probed boundary seed `0`, empty/3D tensor shapes in `fine_to_coarse_tensor()`, `pos_fraction=1.0` full-pool boundary in `build_scenario1_split()`, and `-inf` / mismatched `coarse_labels` integrity checks in `validate_embedding_bundle()` at `c1e031a`.
