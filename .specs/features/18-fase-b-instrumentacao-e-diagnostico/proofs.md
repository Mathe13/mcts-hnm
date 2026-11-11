# 18-fase-b-instrumentacao-e-diagnostico proofs

**HEAD**: 5ac79e2
**Tree**: clean
**Shell**: /bin/sh
**Result**: 16 pass, 0 fail

| Check | Proof | Result |
| --- | --- | --- |
| C1 | `run unit / tests/test_phase_b_observability.py::test_exposure_entropy_zero_selections PASSED` | pass |
| C2 | `run unit / tests/test_phase_b_observability.py::test_exposure_entropy_and_worst10_calculation PASSED` | pass |
| C3 | `run unit / tests/test_phase_b_observability.py::test_scenario1_step_history_logging PASSED` | pass |
| C4 | `run unit / tests/test_phase_b_observability.py::test_scenario2_step_history_logging PASSED` | pass |
| C5 | `run unit / tests/test_phase_b_mcts_diagnostics.py::test_mcts_policy_diagnostics_structure PASSED` | pass |
| C6 | `run unit / tests/test_phase_b_mcts_diagnostics.py::test_mcts_exploration_positivity PASSED` | pass |
| C7 | `run unit / tests/test_phase_b_mcts_diagnostics.py::test_micro_tree_no_starvation PASSED` | pass |
| C8 | `run unit / tests/test_phase_b_mcts_diagnostics.py::test_micro_tree_backpropagation_to_root PASSED` | pass |
| C9 | `run unit / tests/test_phase_b_checkpoints.py::test_checkpoint_save_and_restore_roundtrip PASSED` | pass |
| C10 | `run unit / tests/test_phase_b_checkpoints.py::test_binary_threshold_calibration_optimizes_f1 PASSED` | pass |
| C11 | `run unit / tests/test_phase_b_checkpoints.py::test_scenario1_reports_calibrated_f1_and_auprc PASSED` | pass |
| C12 | `run unit / tests/test_phase_b_gate.py::test_pilot_runner_smoke_execution PASSED` | pass |
| C13 | `run unit / tests/test_phase_b_gate.py::test_pilot_output_isolation PASSED` | pass |
| C14 | `run unit / tests/test_phase_b_gate.py::test_verify_gate_b_passes PASSED` | pass |
| C15 | `run unit / tests/test_phase_b_gate.py::test_verify_gate_b_rejects_nan_and_invariants PASSED` | pass |
| Build | `.venv/bin/python -m compileall -q src tests` | pass |

## Output

`Run unit` - exit 0

```
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/trainer.py:318: UserWarning: Leaf node 21 (class 4) has only 5 instances (< requested K_mcts=24); sampling with replacement.
    batch = mcts_sampler.sample_batch(mcts_leaf_id)

tests/test_phase_b_gate.py::test_pilot_runner_smoke_execution
tests/test_phase_b_gate.py::test_pilot_runner_smoke_execution
tests/test_phase_b_gate.py::test_verify_gate_b_passes
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/trainer.py:318: UserWarning: Leaf node 21 (class 4) has only 10 instances (< requested K_mcts=24); sampling with replacement.
    batch = mcts_sampler.sample_batch(mcts_leaf_id)

tests/test_phase_b_gate.py::test_pilot_output_isolation
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/trainer.py:318: UserWarning: Leaf node 21 (class 4) has only 2 instances (< requested K_mcts=24); sampling with replacement.
    batch = mcts_sampler.sample_batch(mcts_leaf_id)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================== 15 passed, 7 warnings in 4.98s ========================
```
