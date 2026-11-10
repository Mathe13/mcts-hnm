# Global Execution Contract - Checks

Profile: light
Plan: `.specs/features/07-global-execution-contract/plan.md`

## Intent

13 checks in 3 slices · 4 one-way doors · 0 open

## Checks

### S1 - Centralized Configuration and Mode Enforcement · 2 files · 15 KB · ~5k

**C1** - `ExperimentConfig` enforces `ExecutionMode.SMOKE` and `ExecutionMode.REAL` (S1, AC 1)
Proof: `pytest tests/test_experiment_config.py::test_execution_mode_enum_and_config`

**C2** - Invalid execution mode string raises `ValueError` (S1, AC 2)
Proof: `pytest tests/test_experiment_config.py::test_invalid_execution_mode_raises`

**C3** - REAL mode enforces `synthetic_fallback_enabled = False` (S1, AC 3)
Proof: `pytest tests/test_experiment_config.py::test_real_mode_disables_synthetic_fallback`

**C4** - Mutating active session mode raises `IllegalStateError` (S1, AC 4)
Proof: `pytest tests/test_experiment_config.py::test_mode_mutation_raises_illegal_state`

### S2 - Warning Banners and Output Notifications · 2 files · 12 KB · ~4k

**C5** - `print_execution_banner` in SMOKE mode emits prominent warning text and not valid for reporting warning (S2, AC 1)
Proof: `pytest tests/test_experiment_config.py::test_smoke_banner_contents`

**C6** - `print_execution_banner` in REAL mode emits experiment mode REAL and fallback disabled (S2, AC 2)
Proof: `pytest tests/test_experiment_config.py::test_real_banner_contents`

**C7** - `print_execution_banner` formats notebook name, mode, and target output directory (S2, AC 3)
Proof: `pytest tests/test_experiment_config.py::test_banner_formats_notebook_and_output_dir`

### S3 - Directory Isolation and Provenance Metadata · 2 files · 22 KB · ~8k

**C8** - `get_isolated_paths` separates smoke paths from real paths (S3, AC 1, AC 2)
Proof: `pytest tests/test_experiment_config.py::test_get_isolated_paths_smoke_and_real`

**C9** - Path resolution anchors relative paths to project root containing pyproject.toml (S3, AC 3)
Proof: `pytest tests/test_experiment_config.py::test_find_repo_root_and_path_anchoring`

**C10** - Directory initialization establishes `results/legacy/` and preserves existing legacy files (S3, AC 4)
Proof: `pytest tests/test_experiment_config.py::test_legacy_directory_preservation`

**C11** - `save_provenance_metadata` writes JSON containing mode, dataset, seed, synthetic_data, commit, and timestamp (S4, AC 1)
Proof: `pytest tests/test_experiment_config.py::test_save_provenance_metadata`

**C12** - `validate_provenance_metadata` in REAL mode rejects synthetic data with `ValueError` (S4, AC 2)
Proof: `pytest tests/test_experiment_config.py::test_validate_provenance_rejects_synthetic_in_real`

**C13** - `validate_provenance_metadata` raises `FileNotFoundError` when metadata sidecar is absent in REAL mode (S4, AC 3)
Proof: `pytest tests/test_experiment_config.py::test_validate_provenance_missing_file_in_real`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| ExecutionMode options (2) | `smoke` C1 · `real` C1 | - |
| `ExperimentConfig.from_dict()` statuses (3) | 200 C1 · 400 C2 · 422 C2 | - |
| `print_execution_banner()` statuses (1) | 200 C5 | - |
| `get_isolated_paths()` statuses (1) | 200 C8 | - |
| `save_provenance_metadata()` statuses (2) | 200 C11 · 400 C11 | - |
| `validate_provenance_metadata()` statuses (3) | 200 C12 · 409 C4 · 422 C12 | - |
| startup config: active session mode (2 assemblies) | notebook session C4 · CLI runner C4 | - |

- Claims naming a status code, route or response shape: C1, C2, C4, C5, C8, C11, C12 - each has a proof that crosses the boundary
- No other check claims more than the single case its proof exercises

## Swept

- validation: C2
- failure modes: C3, C12, C13
- idempotency: C10
- authorization: n/a - local scientific experimentation without access control gates
- concurrency: C4
- data lifecycle: C10
- dependency failure: C9
- state transitions: C4
- observability: C5, C6, C7

## Handoff

- S1-S3 = 49 KB total across config, utils, and tests; estimate ~17k tokens, well under the 150k budget - one builder
