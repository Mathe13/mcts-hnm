# 03-mcts-hnm-agent verification

**Verdict**: PASS
**Proofs**: `proofs.md` at 4bf5bd5
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Test | Holds |
| --- | --- | --- |
| C1 | `tests/test_mcts_agent.py` `test_ucb1_selection_strict_q_norm_false` | yes |
| C2 | `tests/test_mcts_agent.py` `test_ucb1_selection_enhanced_q_norm_true_and_equal_q_zero` | yes |
| C3 | `tests/test_mcts_agent.py` `test_select_leaf_prioritizes_unvisited_children` | yes |
| C4 | `tests/test_mcts_agent.py` `test_agent_invalid_hyperparameters_raise` | yes |
| C5 | `tests/test_mcts_agent.py` `test_backpropagation_leaf_momentum_update` | yes |
| C6 | `tests/test_mcts_agent.py` `test_backpropagation_weighted_ancestor_aggregation` | yes |
| C7 | `tests/test_mcts_agent.py` `test_compute_mcts_reward_binary_and_multiclass` | yes |
| C8 | `tests/test_mcts_agent.py` `test_backpropagation_invalid_inputs_raise` | yes |
| C9 | `tests/test_mcts_agent.py` `test_dual_stream_batch_sampler_ratios_and_positives` | yes |
| C10 | `tests/test_mcts_agent.py` `test_dual_stream_sampler_small_leaf_replacement_and_validation` | yes |
| C11 | `tests/test_mcts_agent.py` `test_visitation_entropy_zero_single_and_uniform` | yes |
| C12 | `tests/test_mcts_agent.py` `test_superclass_visit_distribution_and_tracker` | yes |

## Findings

None - Probed full 100-class Scenario 2 Semantic and Ward Confusion trees (`exploration_c=0.0` pure greedy vs `exploration_c=2.0`, `momentum_alpha=1.0` memoryless updates, 250-step imbalanced instance-weighted ancestor $Q_p$ and visit conservation invariants, and 100-leaf Shannon entropy $\ln 100$) plus `DualStreamBatchSampler` boundary/irrational ratios (`neg_batch_size=1`, `beta` in `{1e-9, 1/7, 1/3, 2/3, 1-1e-9}`, positive replacement) and extreme binary/100-way logits (`[-100, 100]`) in `compute_mcts_reward`.
