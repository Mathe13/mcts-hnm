# 02-dual-hierarchy-trees proofs

**HEAD**: 19bc837
**Tree**: clean
**Shell**: /bin/sh
**Result**: 13 pass, 0 fail

| Check | Proof | Result |
| --- | --- | --- |
| C1 | `run unit / tests/test_trees.py::test_base_tree_contiguous_arrays_and_invariants PASSED` | pass |
| C2 | `run unit / tests/test_trees.py::test_base_tree_validate_tree_reachability_and_counts PASSED` | pass |
| C3 | `run unit / tests/test_trees.py::test_base_tree_invalid_invariants_raise PASSED` | pass |
| C4 | `run unit / tests/test_trees.py::test_base_tree_reset_mcts_state PASSED` | pass |
| C5 | `run unit / tests/test_trees.py::test_base_tree_save_and_load_roundtrip PASSED` | pass |
| C6 | `run unit / tests/test_trees.py::test_semantic_tree_scenario2_100_classes PASSED` | pass |
| C7 | `run unit / tests/test_trees.py::test_semantic_tree_scenario1_99_negative_classes PASSED` | pass |
| C8 | `run unit / tests/test_trees.py::test_semantic_tree_invalid_inputs_and_empty_superclass PASSED` | pass |
| C9 | `run unit / tests/test_trees.py::test_confusion_distance_matrix_symmetry_and_zero_off_diag PASSED` | pass |
| C10 | `run unit / tests/test_trees.py::test_build_confusion_tree_ward_hierarchy_and_slicing PASSED` | pass |
| C11 | `run unit / tests/test_trees.py::test_confusion_matrix_invalid_inputs_raise PASSED` | pass |
| C12 | `run unit / tests/test_trees.py::test_notebook_02_structure_and_execution PASSED` | pass |
| Build | `.venv/bin/python -m compileall -q src tests` | pass |

## Output

`Run unit` - exit 0

```
tests/test_splits_and_notebook.py::test_notebook_01_structure_and_imports PASSED [ 58%]
tests/test_trees.py::test_base_tree_contiguous_arrays_and_invariants PASSED [ 62%]
tests/test_trees.py::test_base_tree_validate_tree_reachability_and_counts PASSED [ 65%]
tests/test_trees.py::test_base_tree_invalid_invariants_raise PASSED      [ 68%]
tests/test_trees.py::test_base_tree_reset_mcts_state PASSED              [ 72%]
tests/test_trees.py::test_base_tree_save_and_load_roundtrip PASSED       [ 75%]
tests/test_trees.py::test_semantic_tree_scenario2_100_classes PASSED     [ 79%]
tests/test_trees.py::test_semantic_tree_scenario1_99_negative_classes PASSED [ 82%]
tests/test_trees.py::test_semantic_tree_invalid_inputs_and_empty_superclass PASSED [ 86%]
tests/test_trees.py::test_confusion_distance_matrix_symmetry_and_zero_off_diag PASSED [ 89%]
tests/test_trees.py::test_build_confusion_tree_ward_hierarchy_and_slicing PASSED [ 93%]
tests/test_trees.py::test_confusion_matrix_invalid_inputs_raise PASSED   [ 96%]
tests/test_trees.py::test_notebook_02_structure_and_execution PASSED     [100%]

============================== 29 passed in 8.65s ==============================
```
