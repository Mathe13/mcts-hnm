# 09-tier1-benchmarks-real-data verification

**Verdict**: PASS
**Profile**: light
**Diff range**: origin/main..422ad7f
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)

## Summary

Commit `422ad7f` implements complete real data integration, diagnostics, computational budget equivalence, and hyperparameter sensitivity analyses for Tier-1 benchmarks conforming strictly to the global `ExecutionMode` contract.
All 11 automated proofs in `tests/test_tier1_benchmarks_real.py` pass without failures or errors. Across the full test suite, all 110 tests pass cleanly. Inspection confirms no tautological assertions, no mock of tested units, and no silent synthetic fallbacks.

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | Notebook 03 in REAL mode raises `FileNotFoundError` when embeddings cache is missing (S1, AC 1) | `pytest tests/test_tier1_benchmarks_real.py::test_notebook_03_real_mode_missing_cache_raises` exit 0 | `tests/test_tier1_benchmarks_real.py:46` - `with pytest.raises(FileNotFoundError, match="Real CIFAR-100 embeddings not found"):` | PASS |
| C2 | Notebook 03 in SMOKE mode runs using smoke embeddings and writes to `results/smoke/` (S1, AC 2) | `pytest tests/test_tier1_benchmarks_real.py::test_notebook_03_smoke_execution` exit 0 | `tests/test_tier1_benchmarks_real.py:55` - `assert "smoke" in str(paths.results_dir)` | PASS |
| C3 | `run_benchmark` writes `.meta.json` sidecar alongside exported JSON results (S1, AC 3) | `pytest tests/test_tier1_benchmarks_real.py::test_run_benchmark_provenance_metadata` exit 0 | `tests/test_tier1_benchmarks_real.py:89` - `assert meta_file.is_file()` | PASS |
| C4 | Notebook 03 initial cell declares `EXECUTION_MODE = ExecutionMode.REAL` and outputs banner (S1, AC 4) | `pytest tests/test_tier1_benchmarks_real.py::test_notebook_03_structure_and_banner` exit 0 | `tests/test_tier1_benchmarks_real.py:106` - `assert "EXECUTION_MODE = ExecutionMode.REAL" in first_code_src` | PASS |
| C5 | `audit_scenario1_imbalance` computes available positives, available negatives, batch ratio, and cumulative ratio (S2, AC 1) | `pytest tests/test_tier1_benchmarks_real.py::test_audit_scenario1_imbalance` exit 0 | `tests/test_tier1_benchmarks_real.py:127` - `assert audit["available_positives"] == len(split.positive_indices)` | PASS |
| C6 | All Scenario 1 samplers consume equal negative batch budget $K_{\text{neg}} = 48$ (S2, AC 2) | `pytest tests/test_tier1_benchmarks_real.py::test_sampler_negative_batch_budget_equivalence` exit 0 | `tests/test_tier1_benchmarks_real.py:163` - `assert budget_info["neg_batch_size"] == 48` | PASS |
| C7 | Scenario 1 exported results embed the imbalance audit metadata (S2, AC 3) | `pytest tests/test_tier1_benchmarks_real.py::test_scenario1_results_contain_imbalance_audit` exit 0 | `tests/test_tier1_benchmarks_real.py:193` - `assert "imbalance_audit" in res` | PASS |
| C8 | Scenario 2 MCTS execution tracks visit counts, unvisited classes count, and visitation entropy (S3, AC 1) | `pytest tests/test_tier1_benchmarks_real.py::test_scenario2_mcts_diagnostics_tracking` exit 0 | `tests/test_tier1_benchmarks_real.py:225` - `assert "class_visit_distribution" in diag` | PASS |
| C9 | Scenario 2 exported results record class coverage percentage and per-class recall (S3, AC 2) | `pytest tests/test_tier1_benchmarks_real.py::test_scenario2_results_contain_class_coverage_and_recall` exit 0 | `tests/test_tier1_benchmarks_real.py:257` - `assert "class_coverage_pct" in diag` | PASS |
| C10 | `run_sensitivity_ablation` evaluates grid over $c$ and $\alpha$ and exports `tier1_hyperparameter_sensitivity.json` (S4, AC 1, AC 2) | `pytest tests/test_tier1_benchmarks_real.py::test_run_sensitivity_ablation` exit 0 | `tests/test_tier1_benchmarks_real.py:287` - `assert out_file.is_file()` | PASS |
| C11 | Benchmark exports both individual per-seed results and aggregated summaries (S4, AC 3) | `pytest tests/test_tier1_benchmarks_real.py::test_individual_seeds_and_aggregated_export` exit 0 | `tests/test_tier1_benchmarks_real.py:318` - `assert "per_seed_results" in res` | PASS |

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| `run_benchmark()` statuses (3) | 200 C3 · 400 C1 · 422 C7 | - |
| `run_sensitivity_ablation()` statuses (2) | 200 C10 · 422 C10 | - |
| `audit_scenario1_imbalance()` statuses (1) | 200 C5 | - |
| sampler negative batch budget (4 strategies) | `uniform` C6 · `ohem` C6 · `mcts_strict` C6 · `mcts_enhanced` C6 | - |
| sensitivity parameters (2 dimensions) | `c` C10 · `alpha` C10 | - |

## Gate

`.venv/bin/pytest tests/test_tier1_benchmarks_real.py -v` - 11 passed, 0 failed
`.venv/bin/pytest tests/ -v` - 110 passed, 0 failed
