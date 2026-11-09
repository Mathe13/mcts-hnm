# 04-tier1-benchmark-suite verification

**Verdict**: PASS
**Proofs**: `proofs.md` at e0ae07d
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Test | Holds |
| --- | --- | --- |
| C1 | `tests/test_training_loop.py` `test_tier1_mlp_head_forward_and_optimizer` | yes |
| C2 | `tests/test_training_loop.py` `test_uniform_sampler_batch_generation` | yes |
| C3 | `tests/test_training_loop.py` `test_ohem_sampler_selects_highest_loss` | yes |
| C4 | `tests/test_training_loop.py` `test_model_and_sampler_invalid_args_raise` | yes |
| C5 | `tests/test_training_loop.py` `test_train_scenario1_metrics_and_entropy` | yes |
| C6 | `tests/test_training_loop.py` `test_train_scenario2_metrics_and_entropy` | yes |
| C7 | `tests/test_training_loop.py` `test_mcts_updates_and_shifts_leaf_visits` | yes |
| C8 | `tests/test_training_loop.py` `test_trainer_invalid_arguments_raise` | yes |
| C9 | `tests/test_training_loop.py` `test_compute_paired_statistical_tests` | yes |
| C10 | `tests/test_training_loop.py` `test_compute_paired_statistical_tests_zero_diff_fallback` | yes |
| C11 | `tests/test_training_loop.py` `test_run_benchmark_multi_seed_and_export` | yes |
| C12 | `tests/test_training_loop.py` `test_notebook_03_execution` | yes |

## Findings

None - Probed Tier-1 MLP head boundary conditions (`dropout=0.0`, 1D/3D tensor rejections, gradient flow, squeezed vs unsqueezed binary outputs), baseline sampling pools (`UniformSampler` and multi-class `OHEMSampler` under $r_{\text{ohem}} \cdot B_{\text{neg}} > |\mathcal{D}_{\text{pool}}|$), degenerate metric boundaries (all-zero or all-one Scenario 1 labels, missing class recalls in Scenario 2), minimum seed boundaries ($N=2$) and zero-difference fallbacks in paired $t$-test/Wilcoxon tests, atomic JSON/CSV persistence, and end-to-end AST validation and execution of `notebooks/03_tier1_mcts_benchmarks.ipynb`.
