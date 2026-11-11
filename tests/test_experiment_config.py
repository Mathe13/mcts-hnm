"""Tests for centralized experiment configuration, execution mode, banners, and provenance."""

import json
import os
import shutil
import tempfile
from pathlib import Path
import pytest

from src.utils.experiment_config import (
    ExecutionMode,
    ExperimentConfig,
    IllegalStateError,
    IsolatedPaths,
    find_repo_root,
    get_active_session_mode,
    get_isolated_paths,
    migrate_legacy_results,
    print_execution_banner,
    reset_active_session_mode,
    save_provenance_metadata,
    set_active_session_mode,
    validate_mode_isolated_path,
    validate_provenance_metadata,
)


@pytest.fixture(autouse=True)
def reset_session():
    """Ensure clean session mode before and after each test."""
    reset_active_session_mode()
    yield
    reset_active_session_mode()


def test_execution_mode_enum_and_config():
    """C1: ExperimentConfig enforces ExecutionMode.SMOKE and ExecutionMode.REAL."""
    config_real = ExperimentConfig(execution_mode=ExecutionMode.REAL)
    assert config_real.execution_mode == ExecutionMode.REAL
    assert config_real.execution_mode.value == "real"

    reset_active_session_mode()
    config_smoke = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    assert config_smoke.execution_mode == ExecutionMode.SMOKE
    assert config_smoke.execution_mode.value == "smoke"

    reset_active_session_mode()
    config_from_str = ExperimentConfig.from_dict({"execution_mode": "smoke", "seed": 123})
    assert config_from_str.execution_mode == ExecutionMode.SMOKE
    assert config_from_str.seed == 123


def test_invalid_execution_mode_raises():
    """C2: Invalid execution mode string raises ValueError."""
    with pytest.raises(ValueError, match="Invalid execution mode"):
        ExperimentConfig(execution_mode="invalid_mode")

    with pytest.raises(ValueError, match="Invalid execution mode"):
        ExperimentConfig.from_dict({"execution_mode": "mock"})


def test_real_mode_disables_synthetic_fallback():
    """C3: REAL mode enforces synthetic_fallback_enabled = False."""
    config_real = ExperimentConfig(execution_mode=ExecutionMode.REAL)
    assert config_real.synthetic_fallback_enabled is False

    # Attempting to enable synthetic fallback in REAL mode is strictly forbidden
    with pytest.raises(ValueError, match="Synthetic fallback cannot be enabled in REAL mode"):
        ExperimentConfig(execution_mode=ExecutionMode.REAL, synthetic_fallback_enabled=True)


def test_mode_mutation_raises_illegal_state():
    """C4: Mutating active session mode raises IllegalStateError."""
    set_active_session_mode(ExecutionMode.REAL)
    assert get_active_session_mode() == ExecutionMode.REAL

    # Setting same mode is idempotent
    set_active_session_mode(ExecutionMode.REAL)

    # Attempting to switch to SMOKE raises IllegalStateError
    with pytest.raises(IllegalStateError, match="Cannot change active session mode"):
        set_active_session_mode(ExecutionMode.SMOKE)

    # Also initializing a config with a conflicting mode raises IllegalStateError
    with pytest.raises(IllegalStateError, match="Cannot change active session mode"):
        ExperimentConfig(execution_mode=ExecutionMode.SMOKE)


def test_smoke_banner_contents(capsys):
    """C5: print_execution_banner in SMOKE mode emits prominent warning text."""
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    banner = print_execution_banner(config, notebook_name="04_tier2_e2e")
    captured = capsys.readouterr()

    assert "SMOKE TEST — SYNTHETIC/TEST EXECUTION" in banner
    assert "NOT VALID FOR SCIENTIFIC REPORTING" in banner
    assert "Notebook: 04_tier2_e2e" in banner
    assert "Mode: SMOKE" in banner
    assert banner in captured.out


def test_real_banner_contents(capsys):
    """C6: print_execution_banner in REAL mode emits experiment mode REAL and fallback disabled."""
    config = ExperimentConfig(execution_mode=ExecutionMode.REAL, dataset_name="CIFAR-100")
    banner = print_execution_banner(config, notebook_name="03_tier1_mcts")
    captured = capsys.readouterr()

    assert "EXPERIMENT MODE: REAL" in banner
    assert "Dataset: CIFAR-100" in banner
    assert "Synthetic fallback: DISABLED" in banner
    assert "Scientific results: ENABLED" in banner
    assert banner in captured.out


def test_banner_formats_notebook_and_output_dir():
    """C7: print_execution_banner formats notebook name, mode, and target output directory."""
    config = ExperimentConfig(execution_mode=ExecutionMode.REAL, output_dir="results/custom")
    banner = print_execution_banner(config, notebook_name="01_data_and_embeddings")
    assert "Notebook: 01_data_and_embeddings" in banner
    assert "Output directory:" in banner
    assert "results/real" in banner


def test_get_isolated_paths_smoke_and_real(tmp_path):
    """C8: get_isolated_paths separates smoke paths from real paths."""
    config_real = ExperimentConfig(execution_mode=ExecutionMode.REAL)
    paths_real = get_isolated_paths(config_real, repo_root=tmp_path)
    assert paths_real.embeddings_dir == tmp_path / "embeddings" / "real"
    assert paths_real.results_dir == tmp_path / "results" / "real"
    assert paths_real.checkpoints_dir == tmp_path / "checkpoints" / "real"
    assert paths_real.paper_figures_dir == tmp_path / "paper" / "figures"
    assert paths_real.paper_tables_dir == tmp_path / "paper" / "tables"

    reset_active_session_mode()
    config_smoke = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    paths_smoke = get_isolated_paths(config_smoke, repo_root=tmp_path)
    assert paths_smoke.embeddings_dir == tmp_path / "embeddings" / "smoke"
    assert paths_smoke.results_dir == tmp_path / "results" / "smoke"
    assert paths_smoke.checkpoints_dir == tmp_path / "checkpoints" / "smoke"
    assert paths_smoke.paper_figures_dir == tmp_path / "results" / "smoke" / "paper" / "figures"
    assert paths_smoke.paper_tables_dir == tmp_path / "results" / "smoke" / "paper" / "tables"


def test_find_repo_root_and_path_anchoring():
    """C9: Path resolution anchors relative paths to project root containing pyproject.toml."""
    root = find_repo_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src").is_dir()


def test_legacy_directory_preservation(tmp_path):
    """C10: Directory initialization establishes results/legacy/ and preserves existing legacy files via migrate_legacy=True or explicit migration."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    legacy_file = results_dir / "old_benchmark.json"
    legacy_file.write_text('{"legacy": true}')

    config = ExperimentConfig(execution_mode=ExecutionMode.REAL)
    paths = get_isolated_paths(config, repo_root=tmp_path, migrate_legacy=True)

    # Legacy file was moved or preserved into results/legacy
    assert paths.legacy_dir.is_dir()
    assert (paths.legacy_dir / "old_benchmark.json").is_file()
    assert json.loads((paths.legacy_dir / "old_benchmark.json").read_text())["legacy"] is True


def test_save_provenance_metadata(tmp_path):
    """C11: save_provenance_metadata writes JSON containing mode, dataset, seed, synthetic_data, commit, timestamp."""
    config = ExperimentConfig(execution_mode=ExperimentConfig(execution_mode=ExecutionMode.REAL).execution_mode, seed=42)
    artifact_path = tmp_path / "cifar100_resnet18.pt"
    artifact_path.touch()

    meta_path = save_provenance_metadata(
        artifact_path,
        config,
        extra={"data_source": "cifar100_resnet18.pt", "num_steps": 100},
    )

    assert meta_path.is_file()
    data = json.loads(meta_path.read_text())
    assert data["execution_mode"] == "real"
    assert data["dataset"] == "CIFAR-100"
    assert data["seed"] == 42
    assert data["synthetic_data"] is False
    assert data["num_steps"] == 100
    assert "git_commit" in data
    assert "timestamp_utc" in data


def test_validate_provenance_rejects_synthetic_in_real(tmp_path):
    """C12: validate_provenance_metadata in REAL mode rejects synthetic data with ValueError."""
    meta_path = tmp_path / "fake_results.json.meta.json"
    meta_path.write_text(json.dumps({
        "execution_mode": "smoke",
        "dataset": "CIFAR-100",
        "synthetic_data": True,
        "seed": 42
    }))

    # Rejection because synthetic_data is True
    with pytest.raises(ValueError, match="REAL mode rejected synthetic/smoke artifact"):
        validate_provenance_metadata(meta_path, expected_mode=ExecutionMode.REAL)

    # Valid in SMOKE mode
    valid_smoke = validate_provenance_metadata(meta_path, expected_mode=ExecutionMode.SMOKE)
    assert valid_smoke["execution_mode"] == "smoke"


def test_validate_provenance_missing_file_in_real(tmp_path):
    """C13: validate_provenance_metadata raises FileNotFoundError when metadata sidecar is absent in REAL mode."""
    missing_meta = tmp_path / "non_existent.meta.json"
    with pytest.raises(FileNotFoundError, match="Missing mandatory provenance metadata in REAL mode"):
        validate_provenance_metadata(missing_meta, expected_mode=ExecutionMode.REAL)


def test_validate_mode_isolated_path(tmp_path):
    """C6: validate_mode_isolated_path confines I/O paths and rejects unsegregated or cross-mode paths."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "results").mkdir()
    (repo / "results" / "smoke").mkdir(parents=True)
    (repo / "results" / "real").mkdir(parents=True)
    (repo / "paper").mkdir()

    # 1. Unsegregated root results/ is rejected
    with pytest.raises(ValueError, match="Direct unsegregated I/O"):
        validate_mode_isolated_path(repo / "results", mode=ExecutionMode.SMOKE, root=repo)

    with pytest.raises(ValueError, match="Direct unsegregated I/O"):
        validate_mode_isolated_path(repo / "results", mode=ExecutionMode.REAL, root=repo)

    # 2. SMOKE mode rejects root paper/ and REAL results
    with pytest.raises(ValueError, match="SMOKE mode prohibited from reading or writing root paper"):
        validate_mode_isolated_path(repo / "paper", mode=ExecutionMode.SMOKE, root=repo)

    with pytest.raises(ValueError, match="SMOKE mode prohibited from accessing REAL results"):
        validate_mode_isolated_path(repo / "results" / "real", mode=ExecutionMode.SMOKE, root=repo)

    # 3. REAL mode rejects SMOKE results
    with pytest.raises(ValueError, match="REAL mode prohibited from accessing SMOKE results"):
        validate_mode_isolated_path(repo / "results" / "smoke", mode=ExecutionMode.REAL, root=repo)

    # 4. Unsegregated custom subdirectories under results/ are rejected
    with pytest.raises(ValueError, match="inside results/ is not properly segregated"):
        validate_mode_isolated_path(repo / "results" / "custom_unsegregated", mode=ExecutionMode.SMOKE, root=repo)

    with pytest.raises(ValueError, match="inside results/ is not properly segregated"):
        validate_mode_isolated_path(repo / "results" / "custom_unsegregated", mode=ExecutionMode.REAL, root=repo)

    # 5. Valid mode paths pass
    p_smoke = validate_mode_isolated_path(repo / "results" / "smoke", mode=ExecutionMode.SMOKE, root=repo)
    assert p_smoke == (repo / "results" / "smoke").resolve()

    p_real = validate_mode_isolated_path(repo / "results" / "real", mode=ExecutionMode.REAL, root=repo)
    assert p_real == (repo / "results" / "real").resolve()


def test_get_isolated_paths_legacy_migration_default(tmp_path):
    """MIG-01: get_isolated_paths defaults to migrate_legacy=False and does not move files."""
    repo = tmp_path / "repo"
    repo.mkdir()
    results_dir = repo / "results"
    results_dir.mkdir()
    legacy_file = results_dir / "historical_run.json"
    legacy_file.write_text("{}", encoding="utf-8")

    cfg = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)

    # Calling get_isolated_paths with default migrate_legacy=False must NOT move legacy_file
    paths = get_isolated_paths(cfg, repo_root=repo)
    assert legacy_file.exists(), "Legacy file must not be moved when migrate_legacy=False (default)"
    assert not (results_dir / "legacy" / "historical_run.json").exists()

    # Explicit call to migrate_legacy_results must move the file
    moved = migrate_legacy_results(repo_root=repo)
    assert moved == 1
    assert not legacy_file.exists(), "Legacy file must be moved after explicit migrate_legacy_results()"
    assert (results_dir / "legacy" / "historical_run.json").exists()
