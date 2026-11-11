# 17-fase-a-auditoria-e-integridade proofs

**HEAD**: 453e9ba
**Tree**: clean
**Shell**: /bin/sh
**Result**: 17 pass, 0 fail

| Check | Proof | Result |
| --- | --- | --- |
| C1 | `run unit / tests/test_phase_a_audit.py::test_audit_documentation_present_and_complete PASSED` | pass |
| C2 | `run unit / tests/test_phase_a_audit.py::test_generate_run_manifest PASSED` | pass |
| C3 | `run unit / tests/test_phase_a_audit.py::test_legacy_immutability PASSED` | pass |
| C4 | `run unit / tests/test_phase_a_splits.py::test_build_stratified_imbalanced_split_exact_counts PASSED` | pass |
| C5 | `run unit / tests/test_phase_a_splits.py::test_nested_negative_subset_containment PASSED` | pass |
| C6 | `run unit / tests/test_phase_a_splits.py::test_infeasible_counts_raise_value_error PASSED` | pass |
| C7 | `run unit / tests/test_phase_a_splits.py::test_official_test_split_unmodified PASSED` | pass |
| C8 | `run unit / tests/test_phase_a_data_segregation.py::test_periodic_eval_uses_validation_strictly PASSED` | pass |
| C9 | `run unit / tests/test_phase_a_data_segregation.py::test_missing_validation_raises_value_error PASSED` | pass |
| C10 | `run unit / tests/test_phase_a_data_segregation.py::test_tree_calibration_rejects_test_data PASSED` | pass |
| C11 | `run unit / tests/test_phase_a_runner_equivalence.py::test_resolved_training_config_immutability PASSED` | pass |
| C12 | `run unit / tests/test_phase_a_runner_equivalence.py::test_runners_deterministic_equivalence PASSED` | pass |
| C13 | `run unit / tests/test_phase_a_persistence.py::test_atomic_run_id_persistence PASSED` | pass |
| C14 | `run unit / tests/test_phase_a_persistence.py::test_paired_statistics_key_validation PASSED` | pass |
| C15 | `run unit / tests/test_phase_a_tree_validation.py::test_ward_and_average_linkage_properties PASSED` | pass |
| C16 | `run unit / tests/test_phase_a_gate.py::test_verify_gate_a_passes PASSED` | pass |
| Build | `.venv/bin/python -m compileall -q src tests` | pass |

## Output

`Run unit` - exit 0

```
tests/test_phase_a_persistence.py::test_atomic_run_id_persistence PASSED [ 68%]
tests/test_phase_a_persistence.py::test_paired_statistics_key_validation PASSED [ 73%]
tests/test_phase_a_tree_validation.py::test_is_euclidean_dissimilarity_schoenberg PASSED [ 78%]
tests/test_phase_a_tree_validation.py::test_ward_and_average_linkage_properties PASSED [ 84%]
tests/test_phase_a_tree_validation.py::test_build_confusion_tree_rejects_unknown_method PASSED [ 89%]
tests/test_phase_a_gate.py::test_verify_gate_a_passes PASSED             [ 94%]
tests/test_phase_a_gate.py::test_verify_gate_a PASSED                    [100%]

=============================== warnings summary ===============================
tests/test_phase_a_persistence.py::test_atomic_run_id_persistence
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/trainer.py:312: UserWarning: Leaf node 9 (class 4) has only 10 instances (< requested K_mcts=24); sampling with replacement.
    batch = mcts_sampler.sample_batch(mcts_leaf_id)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================== 19 passed, 1 warning in 4.55s =========================
```
