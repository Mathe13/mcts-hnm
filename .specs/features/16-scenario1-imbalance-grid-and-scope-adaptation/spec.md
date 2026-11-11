# F16: Scenario 1 Imbalance Grid Execution and Scope Adaptation Specification

**Approved**: 2026-11-10 - user: "usando tlc-spec-driven: P1 — O notebook 05 exige experimentos que ainda não foram executados..."

## Problem Statement

During experimental audit of the pipeline between Notebook 03 (`03_tier1_mcts_benchmarks.ipynb`) and Notebook 05 (`05_paper_artifacts.ipynb`), a key discrepancy was identified:

1. **P1 — Notebook 05 expects experiments not yet executed in Notebook 03**:
   `generate_scenario1_table()` is designed to present Scenario 1 results across three canonical imbalance ratios ($\rho \in \{1:10, 1:50, 1:100\}$) across all four negative sampling strategies (`uniform`, `ohem`, `mcts_strict`, `mcts_enhanced`). However, Notebook 03 previously ran only a single configuration (nominally $\rho = 1:50$). Consequently, when generating final paper deliverables in `ExecutionMode.REAL`, `generate_scenario1_table()` or dependent assertions could fail due to missing data.

2. **P1 — The loader mislabeled single-ratio results as "1:10"**:
   In `src/utils/paper_artifacts.py`, `load_real_benchmark_results()` previously defaulted to hardcoding `"10"` for runs parsed from `tier1_axis1_samplers.json` (`scen1_norm.setdefault(strat, {}).setdefault("10", ...)`), misrepresenting the 1:50 experimental configuration executed in Notebook 03.

3. **P1 — Lack of explicit ratio parameterization in `run_benchmark()`**:
   `run_benchmark()` in `src/training/benchmark_runner.py` previously executed Scenario 1 without an explicit `imbalance_ratios` parameter, using all available positives (`pos_fraction=1.0`) instead of subsampling according to `build_scenario1_split()` and `IMBALANCE_RATIOS` (`"1:10": 0.10`, `"1:50": 0.02`, `"1:100": 0.01`). Furthermore, individual run results did not record the active imbalance ratio in their metadata.

As stipulated by the user: *The solution is not to fabricate missing configurations with synthetic placeholders. The solution is to execute the corresponding experiments or adapt explicitly the scope of the table to the available experiments.*

---

## Out of Scope

- Modifying the core MCTS algorithms (UCB1, Min-Max Q-norm, Dual-Stream batching).
- Running full-scale GPU multi-seed benchmarks across hours during unit test suites (unit tests continue to use SMOKE mode and fast REAL fixtures within the strict <60s SLA).
- Altering mathematical metrics definitions (F1, AUPRC, FPR@95).

---

## Assumptions & Open Questions

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| Multi-ratio support in `run_benchmark` | Add `imbalance_ratios: tuple[str, ...] = ("1:50",)` to `run_benchmark` for Scenario 1 | Enables running either a single canonical ratio (1:50) or the full grid (`"1:10"`, `"1:50"`, `"1:100"`) | y |
| Deterministic positive subsampling | Use `build_scenario1_split(pos_fraction=IMBALANCE_RATIOS[rho], seed=seed)` | Adheres to `AD-004` and `src.data.splits` contract for exact determinism | y |
| Run metadata traceability | Record `"imbalance_ratio": rho` in `BenchmarkRunResult` and JSON/CSV outputs | Eliminates ambiguity about which imbalance ratio generated a given metric | y |
| Accurate ratio parsing in loader | Parse `"imbalance_ratio"` from runs in `load_real_benchmark_results`, defaulting legacy runs to `"50"` | Accurately reflects that Notebook 03 Axis 1 ran the 1:50 regime, not 1:10 | y |
| Dynamic scope adaptation in table generator | `generate_scenario1_table` dynamically generates rows and adjusts caption for whatever imbalance ratios are present | Prevents crashes when only 1:50 is executed, while generating the full multi-tier table when all ratios are executed | y |
| Strict strategy completeness per executed ratio | `validate_artifact_completeness` requires all 4 canonical strategies for all executed ratios, rejecting incomplete runs without fabricated defaults | Enforces scientific integrity: no partial strategies permitted for any evaluated ratio | y |

---

## Requirements / User Stories

### US-1: Parameterized Imbalance Sampling in Benchmark Runner (RUN-01)
- `run_benchmark()` accepts `imbalance_ratios: tuple[str, ...] = ("1:50",)`.
- When `scenario == "scenario1"`, iterates over `imbalance_ratios`.
- Calls `build_scenario1_split()` with `IMBALANCE_RATIOS[rho]` and active seed.
- Populates `imbalance_ratio` in `BenchmarkRunResult` and exported JSON/CSV.

### US-2: Accurate Ratio Normalization in Paper Artifacts Loader (LOAD-01)
- `load_real_benchmark_results()` inspects `run.get("imbalance_ratio")` (e.g. `"1:10"`, `"1:50"`, `"1:100"`).
- Normalizes to canonical key (`"10"`, `"50"`, `"100"`).
- Defaults legacy/unlabeled runs to `"50"` (matching Notebook 03's documented regime) rather than `"10"`.

### US-3: Dynamic Table Scope Adaptation (TAB-01)
- `generate_scenario1_table()` detects all present imbalance ratios.
- Renders only the executed imbalance ratios with horizontal dividers (`\midrule`) between distinct ratios.
- Adapts the LaTeX caption dynamically to reflect whether $\rho \in \{1:10, 1:50, 1:100\}$, a single ratio $\rho = 1:50$, or a specific subset.
- Validates completeness across all 4 strategies for every executed ratio.

### US-4: Notebook 03 Alignment (NB-01)
- Update `notebooks/03_tier1_mcts_benchmarks.ipynb` to explicitly specify `imbalance_ratios=("1:50",)` (or full grid when desired in REAL mode).
- Document in the markdown cells that Axis 1 evaluates $\rho = 1:50$ by default, and can be configured for $\rho \in \{1:10, 1:50, 1:100\}$ for full-scale paper artifact generation.

---

## Success Criteria

1. `pytest tests/` passes with 100% green status across all test suites in under 60 seconds.
2. `test_generate_scenario1_table` passes cleanly when given single-ratio data (1:50), verifying the generated `.tex` contains `1:50` and does not crash due to absence of `1:10` or `1:100`.
3. `test_generate_scenario1_table` passes cleanly when given full multi-ratio data (1:10, 1:50, 1:100), generating all 3 sections.
4. `run_benchmark(scenario="scenario1", imbalance_ratios=("1:50",))` records `"imbalance_ratio": "1:50"` in output runs.
5. Zero occurrences of hardcoded `"10"` in `paper_artifacts.py` loader for unlabeled legacy runs.
