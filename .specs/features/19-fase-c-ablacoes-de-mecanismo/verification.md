# 19-fase-c-ablacoes-de-mecanismo verification

**Verdict**: PASS
**Proofs**: `proofs.md` at ca4aeb4
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Test | Holds |
| --- | --- | --- |
| C1 | `tests/test_phase_c_ablations.py` `test_uniform_grouped_batch_composition` | yes |
| C2 | `tests/test_phase_c_ablations.py` `test_batch_diversity_calculation` | yes |
| C3 | `tests/test_phase_c_ablations.py` `test_batch_diversity_integration_with_exposure` | yes |
| C4 | `tests/test_phase_c_ablations.py` `test_uniform_grouped_parameter_validation` | yes |
| C5 | `tests/test_phase_c_ablations.py` `test_factorial_variants_configuration` | yes |
| C6 | `tests/test_phase_c_ablations.py` `test_factorial_variants_configuration` | yes |
| C7 | `tests/test_phase_c_ablations.py` `test_factorial_variants_configuration` | yes |
| C8 | `tests/test_phase_c_ablations.py` `test_factorial_variants_configuration` | yes |
| C9 | `tests/test_phase_c_ablations.py` `test_factorial_controlled_factors_invariance` | yes |
| C10 | `tests/test_phase_c_selectors.py` `test_greedy_loss_ema_selector_behavior` | yes |
| C11 | `tests/test_phase_c_selectors.py` `test_flat_ucb_unvisited_priority_and_selection` | yes |
| C12 | `tests/test_phase_c_selectors.py` `test_greedy_loss_ema_selector_behavior` | yes |
| C13 | `tests/test_phase_c_selectors.py` `test_flat_ucb_unvisited_priority_and_selection` | yes |
| C14 | `tests/test_phase_c_hierarchies.py` `test_tree_structural_metrics_calculation` | yes |
| C15 | `tests/test_phase_c_hierarchies.py` `test_random_tree_construction_and_invariants` | yes |
| C16 | `tests/test_phase_c_hierarchies.py` `test_upgma_average_linkage_tree_validity` | yes |
| C17 | `tests/test_phase_c_gate.py` `test_verify_gate_c_passes_valid_ablations` | yes |
| C18 | `tests/test_phase_c_gate.py` `test_verify_gate_c_rejects_divergent_parameters` | yes |
| C19 | `tests/test_phase_c_gate.py` `test_verify_gate_c_passes_valid_ablations` | yes |

## Findings

None - probed boundary cases on batch diversity empty/single/multi distributions, UniformGroupedBatchSampler ratio bounds, GreedyLossEMA single-class and ties, FlatUCB single-class and unvisited priority, FastHierarchyTree topological metrics, and Gate C verifier invariant mismatches and non-finite metric rejections.
