# 20-fase-d-campanha-cientifica-e-artefatos proofs

**HEAD**: fcf6dd1
**Tree**: clean
**Shell**: /bin/sh
**Result**: 18 pass, 0 fail

| Check | Proof | Result |
| --- | --- | --- |
| C1 | `run unit / tests/test_phase_d_campaign.py::test_campaign_runner_scenario1_execution PASSED` | pass |
| C2 | `run unit / tests/test_phase_d_campaign.py::test_campaign_runner_scenario2_execution PASSED` | pass |
| C3 | `run unit / tests/test_phase_d_campaign.py::test_campaign_runner_provenance_and_manifest PASSED` | pass |
| C4 | `run unit / tests/test_phase_d_campaign.py::test_campaign_runner_handles_failure_gracefully PASSED` | pass |
| C5 | `run unit / tests/test_phase_d_metrics.py::test_scenario1_computes_auprc_and_calibrated_f1 PASSED` | pass |
| C6 | `run unit / tests/test_phase_d_metrics.py::test_scenario2_computes_macro_f1_and_worst10 PASSED` | pass |
| C7 | `run unit / tests/test_phase_d_metrics.py::test_computational_cost_telemetry_recording PASSED` | pass |
| C8 | `run unit / tests/test_phase_d_metrics.py::test_dual_checkpoint_metrics_distinction PASSED` | pass |
| C9 | `run unit / tests/test_phase_d_statistics.py::test_paired_method_differences_and_cohens_d PASSED` | pass |
| C10 | `run unit / tests/test_phase_d_statistics.py::test_method_by_ratio_interaction_statistics PASSED` | pass |
| C11 | `run unit / tests/test_phase_d_statistics.py::test_statistical_summary_confidence_intervals PASSED` | pass |
| C12 | `run unit / tests/test_phase_d_artifacts.py::test_figure_generation_validation_and_ablation PASSED` | pass |
| C13 | `run unit / tests/test_phase_d_artifacts.py::test_latex_table_generation_with_checkpoints_and_auprc PASSED` | pass |
| C14 | `run unit / tests/test_phase_d_artifacts.py::test_artifacts_reject_synthetic_or_missing_real_data PASSED` | pass |
| C15 | `run unit / tests/test_phase_d_gate.py::test_verify_gate_d_passes_complete_campaign PASSED` | pass |
| C16 | `run unit / tests/test_phase_d_gate.py::test_verify_gate_d_rejects_incomplete_or_synthetic_campaign PASSED` | pass |
| C17 | `run unit / tests/test_phase_d_gate.py::test_verify_gate_d_generates_signed_audit_manifest PASSED` | pass |
| Build | `.venv/bin/python -m compileall -q src tests` | pass |

## Output

`Run unit` - exit 0

```
tests/test_phase_d_gate.py::test_verify_gate_d_rejects_incomplete_or_synthetic_campaign PASSED [ 94%]
tests/test_phase_d_gate.py::test_verify_gate_d_generates_signed_audit_manifest PASSED [100%]

=============================== warnings summary ===============================
tests/test_phase_d_campaign.py::test_campaign_runner_scenario1_execution
tests/test_phase_d_campaign.py::test_campaign_runner_scenario1_execution
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/trainer.py:382: UserWarning: Leaf node 9 (class 4) has only 10 instances (< requested K_mcts=48); sampling with replacement.
    batch = mcts_sampler.sample_batch(mcts_leaf_id)

tests/test_phase_d_campaign.py::test_campaign_runner_scenario2_execution
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/trainer.py:725: UserWarning: Leaf node 10 (class 4) has only 10 instances (< requested K_mcts=32); sampling with replacement.
    batch = mcts_sampler.sample_batch(mcts_leaf_id)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================== 17 passed, 3 warnings in 9.43s ========================
```
