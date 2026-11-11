# Tasks: F16 Scenario 1 Imbalance Grid Execution and Scope Adaptation

Status: COMPLETED

## User Stories / Task Breakdown

### Phase 1: Benchmark Runner Imbalance Parameterization (RUN-01)
- [x] Task 1.1: Add `imbalance_ratios: tuple[str, ...] = ("1:50",)` to `run_benchmark()` in `src/training/benchmark_runner.py`.
- [x] Task 1.2: In `run_benchmark()`, loop over `imbalance_ratios` for `scenario == "scenario1"` and build split with `build_scenario1_split(pos_fraction=IMBALANCE_RATIOS[rho], seed=seed)`.
- [x] Task 1.3: Add `imbalance_ratio: str | None` field to `BenchmarkRunResult` dataclass and serialize it to JSON/CSV outputs.
- [x] Task 1.4: Update `tests/test_tier1_benchmarks_real.py` to assert `imbalance_ratio` appears in `runs` and verify multi-ratio execution.

### Phase 2: Paper Artifacts Loader & Table Scope Adaptation (LOAD-01, TAB-01)
- [x] Task 2.1: Update `load_real_benchmark_results()` in `src/utils/paper_artifacts.py` to extract `imbalance_ratio` from runs, normalizing `"1:10"` -> `"10"`, `"1:50"` -> `"50"`, `"1:100"` -> `"100"`. Default legacy unlabeled runs to `"50"`.
- [x] Task 2.2: Update `generate_scenario1_table()` in `src/utils/paper_artifacts.py` to dynamically construct table caption according to evaluated ratios and render only executed ratios cleanly.
- [x] Task 2.3: Add unit tests in `tests/test_paper_artifacts_real.py` validating both single-ratio (1:50) and multi-ratio (1:10, 1:50, 1:100) table generation without synthetic fallbacks.

### Phase 3: Notebook 03 Alignment (NB-01)
- [x] Task 3.1: Update `notebooks/03_tier1_mcts_benchmarks.ipynb` Axis 1 call to pass `imbalance_ratios` parameter explicitly, documenting default 1:50 and full grid configuration.
- [x] Task 3.2: Verify Notebook 03 executes cleanly in SMOKE mode.

### Phase 4: Full Verification & Clean-Slate Auditing
- [x] Task 4.1: Run full test suite `pytest tests/` and verify 100% green under <60s SLA (163 passed in 41.62s).
- [x] Task 4.2: Generate `verification.md` and `proofs.md`.
- [x] Task 4.3: Record Architectural Decision AD-019 and update Handoff in `.specs/STATE.md`.
