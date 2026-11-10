# Tier-1 Benchmarks - Checks

Profile: light
Plan: `.specs/features/09-tier1-benchmarks-real-data/plan.md`
Commit: 422ad7f

## Intent

11 checks in 3 slices · 4 one-way doors · 0 open

## Checks

### S1 - Notebook 03 Mode Enforcement and Fallback Elimination · 3 files · 28 KB · ~9k

**C1** - Notebook 03 in REAL mode raises `FileNotFoundError` when embeddings cache is missing (S1, AC 1)
Proof: `pytest tests/test_tier1_benchmarks_real.py::test_notebook_03_real_mode_missing_cache_raises`

**C2** - Notebook 03 in SMOKE mode runs using smoke embeddings and writes to `results/smoke/` (S1, AC 2)
Proof: `pytest tests/test_tier1_benchmarks_real.py::test_notebook_03_smoke_execution`

**C3** - `run_benchmark` writes `.meta.json` sidecar alongside exported JSON results (S1, AC 3)
Proof: `pytest tests/test_tier1_benchmarks_real.py::test_run_benchmark_provenance_metadata`

**C4** - Notebook 03 initial cell declares `EXECUTION_MODE = ExecutionMode.REAL` and outputs banner (S1, AC 4)
Proof: `pytest tests/test_tier1_benchmarks_real.py::test_notebook_03_structure_and_banner`

### S2 - Scenario 1 Imbalance Audit and Computational Budget · 3 files · 24 KB · ~8k

**C5** - `audit_scenario1_imbalance` computes available positives, available negatives, batch ratio, and cumulative ratio (S2, AC 1)
Proof: `pytest tests/test_tier1_benchmarks_real.py::test_audit_scenario1_imbalance`

**C6** - All Scenario 1 samplers consume equal negative batch budget $K_{\text{neg}} = 48$ (S2, AC 2)
Proof: `pytest tests/test_tier1_benchmarks_real.py::test_sampler_negative_batch_budget_equivalence`

**C7** - Scenario 1 exported results embed the imbalance audit metadata (S2, AC 3)
Proof: `pytest tests/test_tier1_benchmarks_real.py::test_scenario1_results_contain_imbalance_audit`

### S3 - Scenario 2 Diagnostics and Sensitivity Analysis · 3 files · 36 KB · ~12k

**C8** - Scenario 2 MCTS execution tracks visit counts, unvisited classes count, and visitation entropy (S3, AC 1)
Proof: `pytest tests/test_tier1_benchmarks_real.py::test_scenario2_mcts_diagnostics_tracking`

**C9** - Scenario 2 exported results record class coverage percentage and per-class recall (S3, AC 2)
Proof: `pytest tests/test_tier1_benchmarks_real.py::test_scenario2_results_contain_class_coverage_and_recall`

**C10** - `run_sensitivity_ablation` evaluates grid over $c$ and $\alpha$ and exports `tier1_hyperparameter_sensitivity.json` (S4, AC 1, AC 2)
Proof: `pytest tests/test_tier1_benchmarks_real.py::test_run_sensitivity_ablation`

**C11** - Benchmark exports both individual per-seed results and aggregated summaries (S4, AC 3)
Proof: `pytest tests/test_tier1_benchmarks_real.py::test_individual_seeds_and_aggregated_export`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| `run_benchmark()` statuses (3) | 200 C3 · 400 C1 · 422 C7 | - |
| `run_sensitivity_ablation()` statuses (2) | 200 C10 · 422 C10 | - |
| `audit_scenario1_imbalance()` statuses (1) | 200 C5 | - |
| sampler negative batch budget (4 strategies) | `uniform` C6 · `ohem` C6 · `mcts_strict` C6 · `mcts_enhanced` C6 | - |
| sensitivity parameters (2 dimensions) | `c` C10 · `alpha` C10 | - |

- Claims naming a status code, route or response shape: C1, C3, C5, C7, C10 - each has a proof that crosses the boundary
- No other check claims more than the single case its proof exercises

## Swept

- validation: C1, C4
- failure modes: C1
- idempotency: C11
- authorization: n/a - local research experimentation without user auth roles
- concurrency: C11
- data lifecycle: C11
- dependency failure: C1
- state transitions: C8
- observability: C4, C5, C8, C9

## Handoff

- S1-S3 = 88 KB total across training, diagnostics, notebooks, and tests; estimate ~30k tokens, well under the 150k budget - one builder
