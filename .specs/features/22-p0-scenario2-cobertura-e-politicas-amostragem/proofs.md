# 22-p0-scenario2-cobertura-e-politicas-amostragem proofs

**HEAD**: 71f07ae
**Tree**: clean
**Shell**: /bin/sh
**Result**: 7 pass, 0 fail

| Check | Proof | Result |
| --- | --- | --- |
| C1 | `run unit / tests/test_scenario2_coverage_and_variants.py::test_mcts_coverage_balanced_strategy_registration PA` | pass |
| C2 | `run unit / tests/test_scenario2_coverage_and_variants.py::test_mcts_coverage_balanced_batch_composition PASSED` | pass |
| C3 | `run unit / tests/test_scenario2_coverage_and_variants.py::test_scenario2_benchmark_runs_all_variants PASSED` | pass |
| C4 | `run unit / tests/test_scenario2_coverage_and_variants.py::test_scenario2_records_quantitative_diagnostics PASS` | pass |
| C5 | `run unit / tests/test_scenario2_coverage_and_variants.py::test_scenario2_paired_differences_against_uniform PA` | pass |
| C6 | `run unit / tests/test_scenario2_coverage_and_variants.py::test_scenario2_rejects_unsupported_strategy PASSED` | pass |
| Build | `.venv/bin/python -m compileall -q src tests` | pass |

## Output

`Run unit` - exit 0

```
tests/test_scenario2_coverage_and_variants.py::test_scenario2_paired_differences_against_uniform
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/trainer.py:739: UserWarning: Leaf node 21 (class 4) has only 6 instances (< requested K_mcts=32); sampling with replacement.
    batch = mcts_sampler.sample_batch(mcts_leaf_id)

tests/test_scenario2_coverage_and_variants.py::test_scenario2_paired_differences_against_uniform
tests/test_scenario2_coverage_and_variants.py::test_scenario2_paired_differences_against_uniform
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/trainer.py:739: UserWarning: Leaf node 21 (class 4) has only 6 instances (< requested K_mcts=16); sampling with replacement.
    batch = mcts_sampler.sample_batch(mcts_leaf_id)

tests/test_scenario2_coverage_and_variants.py::test_scenario2_paired_differences_against_uniform
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/.venv/lib/python3.10/site-packages/scipy/stats/_axis_nan_policy.py:430: RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation. This occurs when the data are nearly identical. Results may be unreliable.
    return hypotest_fun_in(*args, **kwds)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================== 6 passed, 8 warnings in 9.30s =========================
```
