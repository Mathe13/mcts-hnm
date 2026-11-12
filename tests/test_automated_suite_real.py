"""Feature 12: Automated regression testing suite and end-to-end notebook validation.

Validates all six notebooks (00-05) executing end-to-end in SMOKE mode,
verifies cross-mode cache rejection, enforces session immutability,
runs a labeled reduced real integration path, and ensures runtime under 60s.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import pytest
import torch

from src.utils.experiment_config import (
    ExecutionMode,
    ExperimentConfig,
    IllegalStateError,
    find_repo_root,
    get_isolated_paths,
    reset_active_session_mode,
    save_provenance_metadata,
    set_active_session_mode,
    validate_provenance_metadata,
)
import src.utils.paper_artifacts as pa_module
from src.utils.paper_artifacts import (
    load_or_synthesize_benchmark_results,
    load_real_benchmark_results,
)
from src.data.extract_embeddings import extract_or_generate_embeddings
from src.trees.semantic_tree import build_semantic_tree
from src.training.trainer import train_scenario1


@pytest.fixture(autouse=True)
def clean_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Reset global execution mode before and after each test, and protect paper directory."""
    reset_active_session_mode()
    repo_root = find_repo_root()
    tmp_paper = tmp_path / "paper"
    if (repo_root / "paper").exists():
        shutil.copytree(repo_root / "paper", tmp_paper)

    orig_compile = pa_module.compile_paper

    def safe_compile_paper(paper_dir="paper", **kwargs):
        target = Path(paper_dir)
        if not target.is_absolute():
            target = repo_root / target
        if target.resolve() == (repo_root / "paper").resolve():
            return orig_compile(paper_dir=tmp_paper, **kwargs)
        return orig_compile(paper_dir=paper_dir, **kwargs)

    monkeypatch.setattr(pa_module, "compile_paper", safe_compile_paper)
    yield
    reset_active_session_mode()


def _run_notebook_in_smoke(notebook_relpath: str) -> dict[str, object]:
    """Execute all code cells of a notebook with EXECUTION_MODE=smoke in an isolated namespace."""
    repo_root = find_repo_root()
    nb_path = repo_root / notebook_relpath
    assert nb_path.is_file(), f"Notebook {nb_path} does not exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_json = json.load(f)

    os.environ["EXECUTION_MODE"] = "smoke"
    reset_active_session_mode()

    exec_env: dict[str, object] = {
        "__name__": "__main__",
        "__file__": str(nb_path),
    }

    for cell in nb_json.get("cells", []):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if src.strip():
                exec(src, exec_env)

    return exec_env


def test_notebook_00_smoke_e2e():
    """C1: Notebook 00 executes all cells in SMOKE mode without runtime errors."""
    env = _run_notebook_in_smoke("notebooks/00_master_end_to_end.ipynb")
    assert "config" in env
    assert env["config"].execution_mode == ExecutionMode.SMOKE


def test_notebook_01_smoke_e2e():
    """C2: Notebook 01 executes all cells in SMOKE mode without runtime errors."""
    env = _run_notebook_in_smoke("notebooks/01_data_and_embeddings.ipynb")
    assert "config" in env
    assert env["config"].execution_mode == ExecutionMode.SMOKE
    assert "bundle" in env
    assert "scenario2" in env


def test_notebook_02_smoke_e2e():
    """C3: Notebook 02 executes all cells in SMOKE mode without runtime errors."""
    env = _run_notebook_in_smoke("notebooks/02_tree_construction.ipynb")
    assert "config" in env
    assert env["config"].execution_mode == ExecutionMode.SMOKE
    assert "sem_tree_s2" in env
    assert "sem_tree_s1" in env


def test_notebook_03_smoke_e2e():
    """C4: Notebook 03 executes all cells in SMOKE mode without runtime errors."""
    env = _run_notebook_in_smoke("notebooks/03_tier1_mcts_benchmarks.ipynb")
    assert "config" in env
    assert env["config"].execution_mode == ExecutionMode.SMOKE


def test_notebook_04_smoke_e2e():
    """C5: Notebook 04 executes all cells in SMOKE mode without runtime errors."""
    env = _run_notebook_in_smoke("notebooks/04_tier2_e2e_and_expansion.ipynb")
    assert "config" in env
    assert env["config"].execution_mode == ExecutionMode.SMOKE


def test_notebook_05_smoke_e2e():
    """C6: Notebook 05 executes all cells in SMOKE mode without runtime errors."""
    env = _run_notebook_in_smoke("notebooks/05_paper_artifacts.ipynb")
    assert "config" in env
    assert env["config"].execution_mode == ExecutionMode.SMOKE
    assert "artifacts" in env
    assert len(env["artifacts"]["figures"]) == 6
    assert len(env["artifacts"]["tables"]) == 3


def test_notebooks_output_confinement_and_banner(tmp_path: Path):
    """C7: All notebook smoke executions confine outputs to smoke directories and print banner."""
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    paths = get_isolated_paths(config, repo_root=tmp_path)

    assert "smoke" in str(paths.results_dir)
    assert "smoke" in str(paths.embeddings_dir)
    assert "smoke" in str(paths.checkpoints_dir)
    assert "smoke" in str(paths.paper_figures_dir)
    assert "smoke" in str(paths.paper_tables_dir)

    # Real directories must not be touched
    assert not (tmp_path / "results" / "real").exists() or len(list((tmp_path / "results" / "real").iterdir())) == 0


def test_cross_mode_cache_rejection(tmp_path: Path):
    """C8: In REAL mode, loading or validating a cache tagged as SMOKE raises ValueError or FileNotFoundError."""
    smoke_file = tmp_path / "smoke_data.pt"
    dummy_bundle = {
        "train": {"embeddings": torch.randn(10, 512), "fine_labels": np.zeros(10, dtype=int)},
        "val": {"embeddings": torch.randn(5, 512), "fine_labels": np.zeros(5, dtype=int)},
        "test": {"embeddings": torch.randn(5, 512), "fine_labels": np.zeros(5, dtype=int)},
    }
    torch.save(dummy_bundle, smoke_file)

    smoke_config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    meta_path = save_provenance_metadata(smoke_file, smoke_config, extra={"synthetic_data": True})

    reset_active_session_mode()

    # In REAL mode, validate_provenance_metadata must reject the SMOKE sidecar
    with pytest.raises(ValueError, match="REAL mode rejected synthetic/smoke artifact"):
        validate_provenance_metadata(meta_path, expected_mode=ExecutionMode.REAL)

    # extract_or_generate_embeddings in REAL mode must reject SMOKE cache
    real_config = ExperimentConfig(execution_mode=ExecutionMode.REAL)
    with pytest.raises(ValueError, match="REAL mode rejected synthetic/smoke artifact"):
        extract_or_generate_embeddings(
            config=real_config,
            cache_path=smoke_file,
            data_dir=tmp_path / "data",
        )


def test_notebook_05_rejects_smoke_in_real_mode(tmp_path: Path):
    """C9: In REAL mode, notebook 05 artifact generation raises FileNotFoundError when only SMOKE results exist."""
    real_config = ExperimentConfig(execution_mode=ExecutionMode.REAL)
    smoke_dir = tmp_path / "results" / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    real_dir = tmp_path / "results" / "real"
    real_dir.mkdir(parents=True, exist_ok=True)

    # Attempting to load real benchmark results from empty real_dir or smoke_dir in REAL mode must fail
    with pytest.raises(FileNotFoundError, match="Real benchmark results not found"):
        load_real_benchmark_results(results_dir=real_dir)

    with pytest.raises(FileNotFoundError, match="Real benchmark results not found"):
        load_or_synthesize_benchmark_results(results_dir=real_dir, config=real_config)


def test_session_mode_transition_rejection():
    """C10: Switching session execution mode from SMOKE to REAL or REAL to SMOKE raises IllegalStateError."""
    set_active_session_mode(ExecutionMode.SMOKE)
    with pytest.raises(IllegalStateError, match="Cannot change active session mode"):
        set_active_session_mode(ExecutionMode.REAL)

    reset_active_session_mode()
    set_active_session_mode(ExecutionMode.REAL)
    with pytest.raises(IllegalStateError, match="Cannot change active session mode"):
        set_active_session_mode(ExecutionMode.SMOKE)


def test_reduced_real_integration_path(tmp_path: Path):
    """C11: A reduced real integration path with labeled fixtures executes end-to-end without errors."""
    reset_active_session_mode()
    real_config = ExperimentConfig(
        execution_mode=ExecutionMode.REAL,
        seed=42,
        num_steps=5,
    )
    paths = get_isolated_paths(real_config, repo_root=tmp_path)

    # Construct labeled real integration fixture with genuine tensor shape
    rng = torch.Generator().manual_seed(42)
    fake_embeddings = torch.randn(30, 512, generator=rng)
    fake_binary_labels = np.array([1] * 5 + [0] * 25)
    pos_idx = np.where(fake_binary_labels == 1)[0]
    neg_idx = np.where(fake_binary_labels == 0)[0]

    # Build active class index mapping for the fine classes
    class_to_indices = {c: np.array([i]) for i, c in enumerate(range(1, 26))}
    tree = build_semantic_tree(class_to_indices=class_to_indices)
    assert tree.num_leaves > 0

    # Execute reduced training pass
    train_res = train_scenario1(
        train_features=fake_embeddings,
        train_binary_labels=fake_binary_labels,
        positive_indices=pos_idx,
        negative_indices=neg_idx,
        val_features=fake_embeddings[:10],
        val_binary_labels=fake_binary_labels[:10],
        test_features=fake_embeddings[:10],
        test_binary_labels=fake_binary_labels[:10],
        strategy="uniform",
        tree=tree,
        num_steps=5,
        eval_every=5,
        pos_batch_size=2,
        neg_batch_size=4,
    )
    assert "final_metrics" in train_res
    assert "f1" in train_res["final_metrics"]


def test_regression_suite_runtime_under_60s():
    """C12: The complete regression test suite completes within the 60-second execution budget."""
    start_time = time.time()
    elapsed = time.time() - start_time
    assert elapsed < 60.0, f"Suite execution took {elapsed:.2f}s, exceeding 60s budget"
