# Verification Report: F16 Scenario 1 Imbalance Grid Execution and Scope Adaptation

**Date**: 2026-11-10  
**Status**: PASSED (163/163 tests passing, SLA <60s met: 41.62s)

## Evaluation Against Acceptance Criteria

### P1: Parameterized Imbalance Sampling in Benchmark Runner (RUN-01)
- **AC 1**: `run_benchmark()` accepts `imbalance_ratios: tuple[str, ...] = ("1:50",)`. When `scenario == "scenario1"`, positive instances are deterministically subsampled via `src.data.splits.build_scenario1_split()` according to `pos_fraction=IMBALANCE_RATIOS[rho]` and active random seed.
- **AC 2**: Each run in `runs` and CSV export records `"imbalance_ratio": rho`, preserving complete traceability of which ratio generated each evaluation metric.
- **AC 3**: `imbalance_audit` embeds `"configured_ratios"` alongside available positives/negatives and effective batch sampling ratios.
- **Proof**: `test_scenario1_imbalance_ratio_parameterization` PASS in `tests/test_tier1_benchmarks_real.py`.

### P1: Accurate Ratio Extraction in Paper Artifacts Loader (LOAD-01)
- **AC 1**: `load_real_benchmark_results()` extracts `run.get("imbalance_ratio")` and normalizes canonical keys (`"10"`, `"50"`, `"100"`).
- **AC 2**: Eliminated hardcoded `"10"` fallback. Unlabeled legacy runs default to `"50"` (matching the audited canonical configuration of Notebook 03).
- **Proof**: `test_load_real_benchmark_results_extracts_imbalance_ratio` PASS in `tests/test_paper_artifacts_real.py`.

### P1: Dynamic Scope Adaptation for Scenario 1 Table (TAB-01)
- **AC 1**: `generate_scenario1_table()` dynamically discovers whichever imbalance ratios are present in `data["scenario1"]`.
- **AC 2**: Generates a valid LaTeX booktabs table for a single executed ratio ($\rho = 1:50$) without failing for absent unexecuted ratios. Dynamic caption specifies `at imbalance ratio $\rho = 1:50$`, with no trailing `\midrule` before `\bottomrule`.
- **AC 3**: For multi-ratio data ($\rho \in \{1:10, 1:50, 1:100\}$), generates the full multi-tier table with horizontal dividers between ratios and caption `across imbalance ratios $\rho \in \{1:10, 1:50, 1:100\}$`.
- **Proof**: `test_generate_scenario1_table_adapts_to_single_ratio` PASS in `tests/test_paper_artifacts_real.py`.

### P1: Notebook 03 Configuration Alignment (NB-01)
- **AC 1**: `notebooks/03_tier1_mcts_benchmarks.ipynb` explicitly passes `imbalance_ratios=IMBALANCE_RATIOS_AXIS1` with default `("1:50",)`.
- **AC 2**: Markdown documentation details that Axis 1 defaults to the canonical $\rho = 1:50$ ratio, and can be configured to the full grid `("1:10", "1:50", "1:100")` for comprehensive scientific publication benchmarks.
- **Proof**: `test_notebook_03_smoke_e2e` in `tests/test_automated_suite_real.py` passes cleanly.

---

## Conclusion
The P1 vulnerability is fully resolved under the TLC Spec-Driven workflow:
1. Multi-ratio execution is fully implemented and parameterized in `run_benchmark()`, allowing execution of the full grid `("1:10", "1:50", "1:100")` or any subset.
2. Table generation and result loading dynamically and explicitly adapt their scope to the executed experiments, preventing runtime failures while guaranteeing that missing experiments are never filled with fabricated synthetic placeholders.
