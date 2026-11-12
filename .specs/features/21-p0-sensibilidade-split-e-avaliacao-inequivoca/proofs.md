# 21-p0-sensibilidade-split-e-avaliacao-inequivoca proofs

**HEAD**: 3af5e81
**Tree**: clean
**Shell**: /bin/sh
**Result**: 7 pass, 0 fail

| Check | Proof | Result |
| --- | --- | --- |
| C1 | `run unit / tests/test_sensitivity_runner_equivalence.py::test_sensitivity_runner_uses_scenario1_split PASSED` | pass |
| C2 | `run unit / tests/test_sensitivity_runner_equivalence.py::test_sensitivity_indices_match_scenario1_split PASSED` | pass |
| C3 | `run unit / tests/test_sensitivity_runner_equivalence.py::test_sensitivity_output_records_pool_metadata PASSED` | pass |
| C4 | `run unit / tests/test_sensitivity_runner_equivalence.py::test_sensitivity_exports_4_quadrant_metrics PASSED` | pass |
| C5 | `run unit / tests/test_sensitivity_runner_equivalence.py::test_sensitivity_requires_validation_data PASSED` | pass |
| C6 | `run unit / tests/test_sensitivity_runner_equivalence.py::test_benchmark_and_sensitivity_index_equivalence PASS` | pass |
| Build | `.venv/bin/python -m compileall -q src tests` | pass |

## Output

`Run unit` - exit 0

```
tests/test_sensitivity_runner_equivalence.py::test_sensitivity_exports_4_quadrant_metrics PASSED [ 57%]
tests/test_sensitivity_runner_equivalence.py::test_sensitivity_requires_validation_data PASSED [ 71%]
tests/test_sensitivity_runner_equivalence.py::test_benchmark_and_sensitivity_index_equivalence PASSED [ 85%]
tests/test_tier1_benchmarks_real.py::test_run_sensitivity_ablation PASSED [100%]

=============================== warnings summary ===============================
tests/test_tier1_benchmarks_real.py::test_run_sensitivity_ablation
tests/test_tier1_benchmarks_real.py::test_run_sensitivity_ablation
tests/test_tier1_benchmarks_real.py::test_run_sensitivity_ablation
tests/test_tier1_benchmarks_real.py::test_run_sensitivity_ablation
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/trainer.py:391: UserWarning: Leaf node 9 (class 4) has only 10 instances (< requested K_mcts=24); sampling with replacement.
    batch = mcts_sampler.sample_batch(mcts_leaf_id)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================= 7 passed, 12 deselected, 4 warnings in 8.12s =================
```
