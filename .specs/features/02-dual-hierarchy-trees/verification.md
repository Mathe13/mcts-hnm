# 02-dual-hierarchy-trees verification

**Verdict**: PASS
**Proofs**: `proofs.md` at 19bc837
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Test | Holds |
| --- | --- | --- |
| C1 | `tests/test_trees.py::test_base_tree_contiguous_arrays_and_invariants` | yes |
| C2 | `tests/test_trees.py::test_base_tree_validate_tree_reachability_and_counts` | yes |
| C3 | `tests/test_trees.py::test_base_tree_invalid_invariants_raise` | yes |
| C4 | `tests/test_trees.py::test_base_tree_reset_mcts_state` | yes |
| C5 | `tests/test_trees.py::test_base_tree_save_and_load_roundtrip` | yes |
| C6 | `tests/test_trees.py::test_semantic_tree_scenario2_100_classes` | yes |
| C7 | `tests/test_trees.py::test_semantic_tree_scenario1_99_negative_classes` | yes |
| C8 | `tests/test_trees.py::test_semantic_tree_invalid_inputs_and_empty_superclass` | yes |
| C9 | `tests/test_trees.py::test_confusion_distance_matrix_symmetry_and_zero_off_diag` | yes |
| C10 | `tests/test_trees.py::test_build_confusion_tree_ward_hierarchy_and_slicing` | yes |
| C11 | `tests/test_trees.py::test_confusion_matrix_invalid_inputs_raise` | yes |
| C12 | `tests/test_trees.py::test_notebook_02_structure_and_execution` | yes |

## Findings

None - probed single-class (`K=1`) and invalid boolean/negative/2D inputs in `build_semantic_tree()`, minimum-size (`K=2`) vs. `K=1` rejection and maximally asymmetric off-diagonal confusion distances in `build_confusion_tree()`, and extensionless `.npz` round-trip persistence in `FastHierarchyTree.save()` at `efac279`.
