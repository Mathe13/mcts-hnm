"""Feature 13: Documentation and reproducibility verification tests."""

from __future__ import annotations

import re
from pathlib import Path
import pytest

from src.utils.experiment_config import find_repo_root


@pytest.fixture
def repo_root() -> Path:
    return find_repo_root()


@pytest.fixture
def readme_text(repo_root: Path) -> str:
    readme_path = repo_root / "README.md"
    assert readme_path.is_file(), "README.md not found in repository root"
    return readme_path.read_text(encoding="utf-8")


@pytest.fixture
def guide_text(repo_root: Path) -> str:
    guide_path = repo_root / "docs" / "experimental_reproducibility.md"
    assert guide_path.is_file(), "docs/experimental_reproducibility.md not found"
    return guide_path.read_text(encoding="utf-8")


def test_readme_execution_mode_definitions(readme_text: str):
    """C1: README defines ExecutionMode.SMOKE and ExecutionMode.REAL semantics and directory isolation."""
    assert "ExecutionMode" in readme_text
    assert "SMOKE" in readme_text
    assert "REAL" in readme_text
    assert "results/smoke" in readme_text
    assert "results/real" in readme_text
    assert "embeddings/smoke" in readme_text
    assert "embeddings/real" in readme_text


def test_readme_dataset_and_directory_layout(readme_text: str):
    """C2: README documents CIFAR-100 dataset requirements and repository directory layout."""
    assert "CIFAR-100" in readme_text
    assert "Directory Layout" in readme_text or "Project Structure" in readme_text
    assert "checkpoints/" in readme_text
    assert "results/" in readme_text
    assert "embeddings/" in readme_text
    assert ".meta.json" in readme_text


def test_readme_reproduction_commands(readme_text: str):
    """C3: README provides step-by-step commands for smoke testing, real experiments, and artifacts."""
    assert "pytest" in readme_text
    assert "EXECUTION_MODE" in readme_text
    assert "00_master_end_to_end" in readme_text
    assert "05_paper_artifacts" in readme_text
    assert "compile_paper" in readme_text or "paper/main.pdf" in readme_text


def test_readme_limitations_and_guide_link(readme_text: str, repo_root: Path):
    """C4: README documents known experimental limitations and links to docs/experimental_reproducibility.md."""
    assert "docs/experimental_reproducibility.md" in readme_text
    assert "Limitations" in readme_text or "Experimental Limitations" in readme_text
    guide_path = repo_root / "docs" / "experimental_reproducibility.md"
    assert guide_path.is_file()


def test_reproducibility_guide_experiment_catalog(guide_text: str):
    """C5: docs/experimental_reproducibility.md catalogs completed, pending, and historical experiments."""
    assert "Completed Experiments" in guide_text
    assert "Pending Experiments" in guide_text
    assert "Historical Results" in guide_text
    assert "Scenario 1" in guide_text
    assert "Scenario 2" in guide_text
    assert "provenance" in guide_text.lower()


def test_reproducibility_guide_tree_methodology(guide_text: str):
    """C6: docs/experimental_reproducibility.md documents tree geometry methodology (Ward vs Average)."""
    assert "Ward" in guide_text
    assert "Average" in guide_text or "linkage" in guide_text
    assert "Semantic Tree" in guide_text
    assert "Confusion Tree" in guide_text


def test_reproducibility_guide_budget_and_expansion(guide_text: str):
    """C7: docs/experimental_reproducibility.md documents negative mining budget equivalence and dynamic expansion."""
    assert "48" in guide_text  # K_neg budget equivalence
    assert "Dual-Stream" in guide_text or "dual_stream" in guide_text
    assert "Dynamic" in guide_text and "Expansion" in guide_text
    assert "entropy" in guide_text.lower()


def test_documentation_link_integrity(repo_root: Path, readme_text: str, guide_text: str):
    """C8: All relative markdown links across README.md and the guide resolve to existing files."""
    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    # Check links in README.md
    for label, link in link_pattern.findall(readme_text):
        if link.startswith("http") or link.startswith("#") or link.startswith("mailto:"):
            continue
        # Strip anchor if present
        clean_link = link.split("#")[0]
        if not clean_link:
            continue
        target = (repo_root / clean_link).resolve()
        assert target.exists(), f"Broken link in README.md: '{link}' (target: {target})"

    # Check links in docs/experimental_reproducibility.md
    guide_dir = repo_root / "docs"
    for label, link in link_pattern.findall(guide_text):
        if link.startswith("http") or link.startswith("#") or link.startswith("mailto:"):
            continue
        clean_link = link.split("#")[0]
        if not clean_link:
            continue
        target = (guide_dir / clean_link).resolve()
        assert target.exists(), f"Broken link in docs/experimental_reproducibility.md: '{link}' (target: {target})"
