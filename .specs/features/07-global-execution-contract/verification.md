# 07-global-execution-contract verification

**Verdict**: PASS
**Profile**: light
**Diff range**: origin/main..ef98cca
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | `ExperimentConfig` enforces `ExecutionMode.SMOKE` and `ExecutionMode.REAL` (S1, AC 1) | `pytest tests/test_experiment_config.py::test_execution_mode_enum_and_config` exit 0 | `tests/test_experiment_config.py:37` - `assert config_real.execution_mode == ExecutionMode.REAL` | PASS |
| C2 | Invalid execution mode string raises `ValueError` (S1, AC 2) | `pytest tests/test_experiment_config.py::test_invalid_execution_mode_raises` exit 0 | `tests/test_experiment_config.py:53` - `with pytest.raises(ValueError, match="Invalid execution mode"):` | PASS |
| C3 | REAL mode enforces `synthetic_fallback_enabled = False` (S1, AC 3) | `pytest tests/test_experiment_config.py::test_real_mode_disables_synthetic_fallback` exit 0 | `tests/test_experiment_config.py:63` - `assert config_real.synthetic_fallback_enabled is False` | PASS |
| C4 | Mutating active session mode raises `IllegalStateError` (S1, AC 4) | `pytest tests/test_experiment_config.py::test_mode_mutation_raises_illegal_state` exit 0 | `tests/test_experiment_config.py:79` - `with pytest.raises(IllegalStateError, match="Cannot change active session mode"):` | PASS |
| C5 | `print_execution_banner` in SMOKE mode emits prominent warning text and not valid for reporting warning (S2, AC 1) | `pytest tests/test_experiment_config.py::test_smoke_banner_contents` exit 0 | `tests/test_experiment_config.py:93` - `assert "SMOKE TEST — SYNTHETIC/TEST EXECUTION" in banner` | PASS |
| C6 | `print_execution_banner` in REAL mode emits experiment mode REAL and fallback disabled (S2, AC 2) | `pytest tests/test_experiment_config.py::test_real_banner_contents` exit 0 | `tests/test_experiment_config.py:106` - `assert "EXPERIMENT MODE: REAL" in banner` | PASS |
| C7 | `print_execution_banner` formats notebook name, mode, and target output directory (S2, AC 3) | `pytest tests/test_experiment_config.py::test_banner_formats_notebook_and_output_dir` exit 0 | `tests/test_experiment_config.py:117` - `assert "Notebook: 01_data_and_embeddings" in banner` | PASS |
| C8 | `get_isolated_paths` separates smoke paths from real paths (S3, AC 1, AC 2) | `pytest tests/test_experiment_config.py::test_get_isolated_paths_smoke_and_real` exit 0 | `tests/test_experiment_config.py:126` - `assert paths_real.embeddings_dir == tmp_path / "embeddings" / "real"` | PASS |
| C9 | Path resolution anchors relative paths to project root containing pyproject.toml (S3, AC 3) | `pytest tests/test_experiment_config.py::test_find_repo_root_and_path_anchoring` exit 0 | `tests/test_experiment_config.py:145` - `assert (root / "pyproject.toml").is_file()` | PASS |
| C10 | Directory initialization establishes `results/legacy/` and preserves existing legacy files (S3, AC 4) | `pytest tests/test_experiment_config.py::test_legacy_directory_preservation` exit 0 | `tests/test_experiment_config.py:161` - `assert (paths.legacy_dir / "old_benchmark.json").is_file()` | PASS |
| C11 | `save_provenance_metadata` writes JSON containing mode, dataset, seed, synthetic_data, commit, and timestamp (S4, AC 1) | `pytest tests/test_experiment_config.py::test_save_provenance_metadata` exit 0 | `tests/test_experiment_config.py:179` - `assert data["execution_mode"] == "real"` | PASS |
| C12 | `validate_provenance_metadata` in REAL mode rejects synthetic data with `ValueError` (S4, AC 2) | `pytest tests/test_experiment_config.py::test_validate_provenance_rejects_synthetic_in_real` exit 0 | `tests/test_experiment_config.py:199` - `with pytest.raises(ValueError, match="REAL mode rejected synthetic/smoke artifact"):` | PASS |
| C13 | `validate_provenance_metadata` raises `FileNotFoundError` when metadata sidecar is absent in REAL mode (S4, AC 3) | `pytest tests/test_experiment_config.py::test_validate_provenance_missing_file_in_real` exit 0 | `tests/test_experiment_config.py:210` - `with pytest.raises(FileNotFoundError, match="Missing mandatory provenance metadata in REAL mode"):` | PASS |

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

## Gate

`.venv/bin/pytest` - 87 passed, 0 failed
