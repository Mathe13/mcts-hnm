# 05-extensions-and-e2e proofs

**HEAD**: 58f3471
**Tree**: clean
**Shell**: /bin/sh
**Result**: 12 pass, 0 fail

| Check | Proof | Result |
| --- | --- | --- |
| C1 | `run unit / tests/test_dynamic_expansion.py::test_compute_prediction_entropy_binary_and_multiclass PASSED` | pass |
| C2 | `run unit / tests/test_dynamic_expansion.py::test_expand_leaf_by_entropy_splits_and_preserves_invariants PASSED` | pass |
| C3 | `run unit / tests/test_dynamic_expansion.py::test_mcts_agent_sampler_and_diagnostics_on_expanded_tree PASSED` | pass |
| C4 | `run unit / tests/test_dynamic_expansion.py::test_expansion_invalid_arguments_raise PASSED` | pass |
| C5 | `run unit / tests/test_dynamic_expansion.py::test_expansion_edge_cases_noop_bmax_and_ties PASSED` | pass |
| C6 | `run unit / tests/test_dynamic_expansion.py::test_tier1_trainers_with_dynamic_leaf_expansion PASSED` | pass |
| C7 | `run unit / tests/test_dynamic_expansion.py::test_tier2_resnet18_forward_and_features PASSED` | pass |
| C8 | `run unit / tests/test_dynamic_expansion.py::test_train_e2e_scenario1_and_scenario2_with_refresh_and_expansion ` | pass |
| C9 | `run unit / tests/test_dynamic_expansion.py::test_run_e2e_experiment_multi_seed_and_export PASSED` | pass |
| C10 | `run unit / tests/test_dynamic_expansion.py::test_e2e_trainer_invalid_arguments_raise PASSED` | pass |
| C11 | `run unit / tests/test_dynamic_expansion.py::test_notebook_04_execution PASSED` | pass |
| Build | `.venv/bin/python -m compileall -q src tests` | pass |

## Output

`Run unit` - exit 0

```
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/trainer.py:300: UserWarning: Leaf node 21 (class 4) has only 30 instances (< requested K_mcts=48); sampling with replacement.
    batch = mcts_sampler.sample_batch(mcts_leaf_id)

tests/test_training_loop.py::test_notebook_03_execution
tests/test_training_loop.py::test_notebook_03_execution
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/trainer.py:473: UserWarning: Leaf node 21 (class 4) has only 30 instances (< requested K_mcts=32); sampling with replacement.
    batch = mcts_sampler.sample_batch(mcts_leaf_id)

tests/test_training_loop.py::test_run_benchmark_multi_seed_and_export
tests/test_training_loop.py::test_run_benchmark_multi_seed_and_export
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/trainer.py:300: UserWarning: Leaf node 3 (class 1) has only 20 instances (< requested K_mcts=24); sampling with replacement.
    batch = mcts_sampler.sample_batch(mcts_leaf_id)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 65 passed, 6 warnings in 27.88s ========================
```
