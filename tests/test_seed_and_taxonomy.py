"""Unit tests for DATA-01: pyproject configuration, deterministic seeding, and CIFAR-100 taxonomy."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


def test_pyproject_configuration() -> None:
    """Verify DATA-01 AC 1: pyproject.toml metadata, Python >=3.10, src root, and dependencies."""
    repo_root = Path(__file__).resolve().parent.parent
    pyproject_path = repo_root / "pyproject.toml"
    assert pyproject_path.is_file(), "pyproject.toml must exist at repository root"

    with pyproject_path.open("rb") as fh:
        config = tomllib.load(fh)

    project = config["project"]
    assert project["name"] == "mcts-hnm"
    assert project["requires-python"] == ">=3.10"

    declared_deps = " ".join(project["dependencies"]).lower()
    required_packages = (
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "ipykernel",
        "nbformat",
        "pytest",
    )
    for pkg in required_packages:
        assert pkg in declared_deps, f"Missing required dependency '{pkg}' in pyproject.toml"

    find_cfg = config["tool"]["setuptools"]["packages"]["find"]
    assert "src*" in find_cfg["include"]
    assert "." in config["tool"]["pytest"]["ini_options"]["pythonpath"]
