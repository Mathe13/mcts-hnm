# Automated Regression Testing Suite - Checks

Profile: light
Plan: `.specs/features/12-automated-testing-suite/plan.md`

## Intent

12 checks in 3 slices · 3 one-way doors · 0 open

## Checks

### S1 - End-to-End Notebook Execution in SMOKE Mode · 3 files · 30 KB · ~10k

**C1** - Notebook 00 executes all cells in SMOKE mode without runtime errors (S1, AC 1)
Proof: `pytest tests/test_automated_suite_real.py::test_notebook_00_smoke_e2e`

**C2** - Notebook 01 executes all cells in SMOKE mode without runtime errors (S1, AC 1)
Proof: `pytest tests/test_automated_suite_real.py::test_notebook_01_smoke_e2e`

**C3** - Notebook 02 executes all cells in SMOKE mode without runtime errors (S1, AC 1)
Proof: `pytest tests/test_automated_suite_real.py::test_notebook_02_smoke_e2e`

**C4** - Notebook 03 executes all cells in SMOKE mode without runtime errors (S1, AC 1)
Proof: `pytest tests/test_automated_suite_real.py::test_notebook_03_smoke_e2e`

**C5** - Notebook 04 executes all cells in SMOKE mode without runtime errors (S1, AC 1)
Proof: `pytest tests/test_automated_suite_real.py::test_notebook_04_smoke_e2e`

**C6** - Notebook 05 executes all cells in SMOKE mode without runtime errors (S1, AC 1)
Proof: `pytest tests/test_automated_suite_real.py::test_notebook_05_smoke_e2e`

**C7** - All notebook smoke executions confine outputs to smoke directories and print execution banner (S1, AC 2, AC 3)
Proof: `pytest tests/test_automated_suite_real.py::test_notebooks_output_confinement_and_banner`

### S2 - Cross-Mode Cache Rejection and Session Immutability · 2 files · 20 KB · ~7k

**C8** - In REAL mode, loading or validating a cache tagged as SMOKE raises `ValueError` or `FileNotFoundError` (S2, AC 1)
Proof: `pytest tests/test_automated_suite_real.py::test_cross_mode_cache_rejection`

**C9** - In REAL mode, notebook 05 artifact generation raises `FileNotFoundError` when only SMOKE results exist (S2, AC 2)
Proof: `pytest tests/test_automated_suite_real.py::test_notebook_05_rejects_smoke_in_real_mode`

**C10** - Switching session execution mode from SMOKE to REAL or REAL to SMOKE raises `IllegalStateError` (S2, AC 3)
Proof: `pytest tests/test_automated_suite_real.py::test_session_mode_transition_rejection`

### S3 - Reduced Real Integration Path and Performance SLA · 2 files · 20 KB · ~7k

**C11** - A reduced real integration path with labeled fixtures executes end-to-end without errors (S3, AC 1)
Proof: `pytest tests/test_automated_suite_real.py::test_reduced_real_integration_path`

**C12** - The complete test suite runs and completes within the 60-second execution budget (S3, AC 2)
Proof: `pytest tests/test_automated_suite_real.py::test_regression_suite_runtime_under_60s`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| `run_notebook_in_smoke()` statuses (6) | 00 C1 · 01 C2 · 02 C3 · 03 C4 · 04 C5 · 05 C6 | - |
| `test_cross_mode_cache_rejection()` statuses (3) | 200 C8 · 400 C8 · 422 C8 | - |
| `test_reduced_real_integration_path()` statuses (3) | 200 C11 · 400 C9 · 422 C10 | - |
| output confinement and performance statuses (2) | confinement C7 · SLA C12 | - |

- Claims naming a status code, route or response shape: C1, C2, C3, C4, C5, C6, C8, C9, C10, C11 - each has a proof that crosses the boundary
- No other check claims more than the single case its proof exercises

## Swept

- validation: C7, C8, C9
- failure modes: C8, C9, C10
- idempotency: C1, C2, C3, C4, C5, C6, C11
- authorization: n/a - local research experimentation without user auth roles
- concurrency: C11, C12
- data lifecycle: C7, C8, C9, C11
- dependency failure: C8, C9
- state transitions: C10
- observability: C7, C12

## Handoff

- S1-S3 = 70 KB total across end-to-end notebook execution tests, cross-mode rejection tests, and integration test fixtures; estimate ~22k tokens, well under the 150k budget - one builder
