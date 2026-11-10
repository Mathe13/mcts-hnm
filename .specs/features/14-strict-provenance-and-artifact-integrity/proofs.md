# 14-strict-provenance-and-artifact-integrity proofs

**HEAD**: c5c081e
**Tree**: clean
**Shell**: /bin/sh
**Result**: 9 pass, 0 fail

| Check | Proof | Result |
| --- | --- | --- |
| C1 | `run unit / tests/test_experiment_config.py::test_validate_provenance_rejects_synthetic_in_real PASSED` | pass |
| C2 | `run unit / tests/test_paper_artifacts_real.py::test_load_real_benchmark_results_provenance_validation PASSED` | pass |
| C3 | `run unit / tests/test_paper_artifacts.py::test_latex_tables_raise_on_missing_metrics PASSED` | pass |
| C4 | `run unit / tests/test_paper_artifacts_real.py::test_validate_artifact_completeness PASSED` | pass |
| C5 | `run unit / tests/test_tier2_e2e_real.py::test_load_cifar100_raw_images_smoke_synthetic PASSED` | pass |
| C6 | `run unit / tests/test_experiment_config.py::test_validate_mode_isolated_path PASSED` | pass |
| C7 | `run unit / tests/test_automated_suite_real.py::test_notebook_00_smoke_e2e PASSED` | pass |
| C8 | `run unit / tests/test_automated_suite_real.py::test_reduced_real_integration_path PASSED` | pass |
| Build | `.venv/bin/python -m compileall -q src tests` | pass |

## Output

`Run unit` - exit 0

```
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/trainer.py:300: UserWarning: Leaf node 9 (class 4) has only 10 instances (< requested K_mcts=24); sampling with replacement.
    batch = mcts_sampler.sample_batch(mcts_leaf_id)

tests/test_tier1_benchmarks_real.py::test_scenario2_mcts_diagnostics_tracking
tests/test_tier1_benchmarks_real.py::test_scenario2_results_contain_class_coverage_and_recall
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/trainer.py:473: UserWarning: Leaf node 10 (class 4) has only 10 instances (< requested K_mcts=32); sampling with replacement.
    batch = mcts_sampler.sample_batch(mcts_leaf_id)

tests/test_training_loop.py::test_run_benchmark_multi_seed_and_export
tests/test_training_loop.py::test_run_benchmark_multi_seed_and_export
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/trainer.py:300: UserWarning: Leaf node 3 (class 1) has only 20 instances (< requested K_mcts=24); sampling with replacement.
    batch = mcts_sampler.sample_batch(mcts_leaf_id)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================= 155 passed, 31 warnings in 76.71s (0:01:16) ==================
```
