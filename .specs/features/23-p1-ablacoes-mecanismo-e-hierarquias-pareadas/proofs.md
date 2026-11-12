# 23-p1-ablacoes-mecanismo-e-hierarquias-pareadas proofs

**HEAD**: 2fdf64f
**Tree**: clean
**Shell**: /bin/sh
**Result**: 7 pass, 0 fail

| Check | Proof | Result |
| --- | --- | --- |
| C1 | `run unit / tests/test_p1_ablations_and_hierarchies.py::test_tree_topology_diagnostics PASSED` | pass |
| C2 | `run unit / tests/test_p1_ablations_and_hierarchies.py::test_factorial_2x2_effects_calculation PASSED` | pass |
| C3 | `run unit / tests/test_p1_ablations_and_hierarchies.py::test_benchmark_records_computational_costs PASSED` | pass |
| C4 | `run unit / tests/test_p1_ablations_and_hierarchies.py::test_axis2_paired_hierarchy_inferential_tests PASSED` | pass |
| C5 | `run unit / tests/test_p1_ablations_and_hierarchies.py::test_axis2_persists_tree_topology_diagnostics PASSED` | pass |
| C6 | `run unit / tests/test_p1_ablations_and_hierarchies.py::test_factorial_2x2_execution_and_persistence PASSED` | pass |
| Build | `.venv/bin/python -m compileall -q src tests` | pass |

## Output

`Run unit` - exit 0

```
tests/test_p1_ablations_and_hierarchies.py::test_axis2_paired_hierarchy_inferential_tests
tests/test_p1_ablations_and_hierarchies.py::test_axis2_paired_hierarchy_inferential_tests
tests/test_p1_ablations_and_hierarchies.py::test_axis2_persists_tree_topology_diagnostics
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/trainer.py:396: UserWarning: Leaf node 26 (class 9) has only 10 instances (< requested K_mcts=24); sampling with replacement.
    batch = mcts_sampler.sample_batch(mcts_leaf_id)

tests/test_p1_ablations_and_hierarchies.py::test_factorial_2x2_execution_and_persistence
tests/test_p1_ablations_and_hierarchies.py::test_factorial_2x2_execution_and_persistence
tests/test_p1_ablations_and_hierarchies.py::test_factorial_2x2_execution_and_persistence
tests/test_p1_ablations_and_hierarchies.py::test_factorial_2x2_execution_and_persistence
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/trainer.py:396: UserWarning: Leaf node 11 (class 4) has only 10 instances (< requested K_mcts=48); sampling with replacement.
    batch = mcts_sampler.sample_batch(mcts_leaf_id)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================== 7 passed, 15 warnings in 8.70s ========================
```
