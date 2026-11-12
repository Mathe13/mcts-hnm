# 24-p2-expansao-estudo-razoes-e-classes-pareadas proofs

**HEAD**: 2eae17b
**Tree**: clean
**Shell**: /bin/sh
**Result**: 7 pass, 0 fail

| Check | Proof | Result |
| --- | --- | --- |
| C1 | `run unit / tests/test_p2_study_expansion.py::test_multi_ratio_execution_and_positive_fractions PASSED` | pass |
| C2 | `run unit / tests/test_p2_study_expansion.py::test_method_by_ratio_interaction_calculation PASSED` | pass |
| C3 | `run unit / tests/test_p2_study_expansion.py::test_multi_ratio_persistence_and_interactions_payload PASSED` | pass |
| C4 | `run unit / tests/test_p2_study_expansion.py::test_multi_target_class_execution PASSED` | pass |
| C5 | `run unit / tests/test_p2_study_expansion.py::test_target_class_persistence_and_aggregation PASSED` | pass |
| C6 | `run unit / tests/test_p2_study_expansion.py::test_notebook_03_multi_ratio_and_multi_class_integration PASSED` | pass |
| Build | `.venv/bin/python -m compileall -q src tests` | pass |

## Output

`Run unit` - exit 0

```
tests/test_p2_study_expansion.py::test_method_by_ratio_interaction_calculation
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/.venv/lib/python3.10/site-packages/scipy/stats/_axis_nan_policy.py:586: RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation. This occurs when the data are nearly identical. Results may be unreliable.
    res = hypotest_fun_out(*samples, **kwds)

tests/test_p2_study_expansion.py::test_multi_ratio_persistence_and_interactions_payload
tests/test_p2_study_expansion.py::test_multi_ratio_persistence_and_interactions_payload
tests/test_p2_study_expansion.py::test_multi_ratio_persistence_and_interactions_payload
tests/test_p2_study_expansion.py::test_multi_ratio_persistence_and_interactions_payload
tests/test_p2_study_expansion.py::test_target_class_persistence_and_aggregation
tests/test_p2_study_expansion.py::test_target_class_persistence_and_aggregation
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/trainer.py:396: UserWarning: Leaf node 11 (class 4) has only 15 instances (< requested K_mcts=24); sampling with replacement.
    batch = mcts_sampler.sample_batch(mcts_leaf_id)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================== 6 passed, 7 warnings in 10.82s ========================
```
