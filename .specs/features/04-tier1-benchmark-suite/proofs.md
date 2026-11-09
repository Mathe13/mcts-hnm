# 04-tier1-benchmark-suite proofs

**HEAD**: e0ae07d
**Tree**: clean
**Shell**: /bin/sh
**Result**: 13 pass, 0 fail

| Check | Proof | Result |
| --- | --- | --- |
| C1 | `run unit / tests/test_training_loop.py::test_tier1_mlp_head_forward_and_optimizer PASSED` | pass |
| C2 | `run unit / tests/test_training_loop.py::test_uniform_sampler_batch_generation PASSED` | pass |
| C3 | `run unit / tests/test_training_loop.py::test_ohem_sampler_selects_highest_loss PASSED` | pass |
| C4 | `run unit / tests/test_training_loop.py::test_model_and_sampler_invalid_args_raise PASSED` | pass |
| C5 | `run unit / tests/test_training_loop.py::test_train_scenario1_metrics_and_entropy PASSED` | pass |
| C6 | `run unit / tests/test_training_loop.py::test_train_scenario2_metrics_and_entropy PASSED` | pass |
| C7 | `run unit / tests/test_training_loop.py::test_mcts_updates_and_shifts_leaf_visits PASSED` | pass |
| C8 | `run unit / tests/test_training_loop.py::test_trainer_invalid_arguments_raise PASSED` | pass |
| C9 | `run unit / tests/test_training_loop.py::test_compute_paired_statistical_tests PASSED` | pass |
| C10 | `run unit / tests/test_training_loop.py::test_compute_paired_statistical_tests_zero_diff_fallback PASSED` | pass |
| C11 | `run unit / tests/test_training_loop.py::test_run_benchmark_multi_seed_and_export PASSED` | pass |
| C12 | `run unit / tests/test_training_loop.py::test_notebook_03_execution PASSED` | pass |
| Build | `.venv/bin/python -m compileall -q src tests` | pass |

## Output

`Run unit` - exit 0

```
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/trainer.py:296: UserWarning: Leaf node 21 (class 4) has only 30 instances (< requested K_mcts=48); sampling with replacement.
    batch = mcts_sampler.sample_batch(mcts_leaf_id)

tests/test_training_loop.py::test_notebook_03_execution
tests/test_training_loop.py::test_notebook_03_execution
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/trainer.py:453: UserWarning: Leaf node 21 (class 4) has only 30 instances (< requested K_mcts=32); sampling with replacement.
    batch = mcts_sampler.sample_batch(mcts_leaf_id)

tests/test_training_loop.py::test_run_benchmark_multi_seed_and_export
tests/test_training_loop.py::test_run_benchmark_multi_seed_and_export
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/trainer.py:296: UserWarning: Leaf node 3 (class 1) has only 20 instances (< requested K_mcts=24); sampling with replacement.
    batch = mcts_sampler.sample_batch(mcts_leaf_id)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 54 passed, 6 warnings in 19.67s ========================
```
