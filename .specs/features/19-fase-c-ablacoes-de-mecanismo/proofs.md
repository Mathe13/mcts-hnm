# 19-fase-c-ablacoes-de-mecanismo proofs

**HEAD**: ca4aeb4
**Tree**: clean
**Shell**: /bin/sh
**Result**: 20 pass, 0 fail

| Check | Proof | Result |
| --- | --- | --- |
| C1 | `run unit / tests/test_phase_c_ablations.py::test_uniform_grouped_batch_composition PASSED` | pass |
| C2 | `run unit / tests/test_phase_c_ablations.py::test_batch_diversity_calculation PASSED` | pass |
| C3 | `run unit / tests/test_phase_c_ablations.py::test_batch_diversity_integration_with_exposure PASSED` | pass |
| C4 | `run unit / tests/test_phase_c_ablations.py::test_uniform_grouped_parameter_validation PASSED` | pass |
| C5 | `run unit / tests/test_phase_c_ablations.py::test_factorial_variants_configuration PASSED` | pass |
| C6 | `run unit / tests/test_phase_c_ablations.py::test_factorial_variants_configuration PASSED` | pass |
| C7 | `run unit / tests/test_phase_c_ablations.py::test_factorial_variants_configuration PASSED` | pass |
| C8 | `run unit / tests/test_phase_c_ablations.py::test_factorial_variants_configuration PASSED` | pass |
| C9 | `run unit / tests/test_phase_c_ablations.py::test_factorial_controlled_factors_invariance PASSED` | pass |
| C10 | `run unit / tests/test_phase_c_selectors.py::test_greedy_loss_ema_selector_behavior PASSED` | pass |
| C11 | `run unit / tests/test_phase_c_selectors.py::test_flat_ucb_unvisited_priority_and_selection PASSED` | pass |
| C12 | `run unit / tests/test_phase_c_selectors.py::test_greedy_loss_ema_selector_behavior PASSED` | pass |
| C13 | `run unit / tests/test_phase_c_selectors.py::test_flat_ucb_unvisited_priority_and_selection PASSED` | pass |
| C14 | `run unit / tests/test_phase_c_hierarchies.py::test_tree_structural_metrics_calculation PASSED` | pass |
| C15 | `run unit / tests/test_phase_c_hierarchies.py::test_random_tree_construction_and_invariants PASSED` | pass |
| C16 | `run unit / tests/test_phase_c_hierarchies.py::test_upgma_average_linkage_tree_validity PASSED` | pass |
| C17 | `run unit / tests/test_phase_c_gate.py::test_verify_gate_c_passes_valid_ablations PASSED` | pass |
| C18 | `run unit / tests/test_phase_c_gate.py::test_verify_gate_c_rejects_divergent_parameters PASSED` | pass |
| C19 | `run unit / tests/test_phase_c_gate.py::test_verify_gate_c_passes_valid_ablations PASSED` | pass |
| Build | `.venv/bin/python -m compileall -q src tests` | pass |

## Output

`Run unit` - exit 0

```
tests/test_phase_c_ablations.py::test_batch_diversity_calculation PASSED [  7%]
tests/test_phase_c_ablations.py::test_batch_diversity_integration_with_exposure PASSED [ 15%]
tests/test_phase_c_ablations.py::test_uniform_grouped_batch_composition PASSED [ 23%]
tests/test_phase_c_ablations.py::test_uniform_grouped_parameter_validation PASSED [ 30%]
tests/test_phase_c_ablations.py::test_factorial_variants_configuration PASSED [ 38%]
tests/test_phase_c_ablations.py::test_factorial_controlled_factors_invariance PASSED [ 46%]
tests/test_phase_c_selectors.py::test_greedy_loss_ema_selector_behavior PASSED [ 53%]
tests/test_phase_c_selectors.py::test_flat_ucb_unvisited_priority_and_selection PASSED [ 61%]
tests/test_phase_c_hierarchies.py::test_tree_structural_metrics_calculation PASSED [ 69%]
tests/test_phase_c_hierarchies.py::test_random_tree_construction_and_invariants PASSED [ 76%]
tests/test_phase_c_hierarchies.py::test_upgma_average_linkage_tree_validity PASSED [ 84%]
tests/test_phase_c_gate.py::test_verify_gate_c_passes_valid_ablations PASSED [ 92%]
tests/test_phase_c_gate.py::test_verify_gate_c_rejects_divergent_parameters PASSED [100%]

============================== 13 passed in 4.86s ==============================
```
