# 03-mcts-hnm-agent proofs

**HEAD**: 4bf5bd5
**Tree**: clean
**Shell**: /bin/sh
**Result**: 13 pass, 0 fail

| Check | Proof | Result |
| --- | --- | --- |
| C1 | `run unit / tests/test_mcts_agent.py::test_ucb1_selection_strict_q_norm_false PASSED` | pass |
| C2 | `run unit / tests/test_mcts_agent.py::test_ucb1_selection_enhanced_q_norm_true_and_equal_q_zero PASSED` | pass |
| C3 | `run unit / tests/test_mcts_agent.py::test_select_leaf_prioritizes_unvisited_children PASSED` | pass |
| C4 | `run unit / tests/test_mcts_agent.py::test_agent_invalid_hyperparameters_raise PASSED` | pass |
| C5 | `run unit / tests/test_mcts_agent.py::test_backpropagation_leaf_momentum_update PASSED` | pass |
| C6 | `run unit / tests/test_mcts_agent.py::test_backpropagation_weighted_ancestor_aggregation PASSED` | pass |
| C7 | `run unit / tests/test_mcts_agent.py::test_compute_mcts_reward_binary_and_multiclass PASSED` | pass |
| C8 | `run unit / tests/test_mcts_agent.py::test_backpropagation_invalid_inputs_raise PASSED` | pass |
| C9 | `run unit / tests/test_mcts_agent.py::test_dual_stream_batch_sampler_ratios_and_positives PASSED` | pass |
| C10 | `run unit / tests/test_mcts_agent.py::test_dual_stream_sampler_small_leaf_replacement_and_validation PASSED` | pass |
| C11 | `run unit / tests/test_mcts_agent.py::test_visitation_entropy_zero_single_and_uniform PASSED` | pass |
| C12 | `run unit / tests/test_mcts_agent.py::test_superclass_visit_distribution_and_tracker PASSED` | pass |
| Build | `.venv/bin/python -m compileall -q src tests` | pass |

## Output

`Run unit` - exit 0

```
tests/test_splits_and_notebook.py::test_notebook_01_structure_and_imports PASSED [ 70%]
tests/test_trees.py::test_base_tree_contiguous_arrays_and_invariants PASSED [ 73%]
tests/test_trees.py::test_base_tree_validate_tree_reachability_and_counts PASSED [ 75%]
tests/test_trees.py::test_base_tree_invalid_invariants_raise PASSED      [ 78%]
tests/test_trees.py::test_base_tree_reset_mcts_state PASSED              [ 80%]
tests/test_trees.py::test_base_tree_save_and_load_roundtrip PASSED       [ 82%]
tests/test_trees.py::test_semantic_tree_scenario2_100_classes PASSED     [ 85%]
tests/test_trees.py::test_semantic_tree_scenario1_99_negative_classes PASSED [ 87%]
tests/test_trees.py::test_semantic_tree_invalid_inputs_and_empty_superclass PASSED [ 90%]
tests/test_trees.py::test_confusion_distance_matrix_symmetry_and_zero_off_diag PASSED [ 92%]
tests/test_trees.py::test_build_confusion_tree_ward_hierarchy_and_slicing PASSED [ 95%]
tests/test_trees.py::test_confusion_matrix_invalid_inputs_raise PASSED   [ 97%]
tests/test_trees.py::test_notebook_02_structure_and_execution PASSED     [100%]

============================== 41 passed in 9.93s ==============================
```
