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


def test_set_global_seed_reproducibility() -> None:
    """Verify DATA-01 AC 2: set_global_seed(42) produces bit-identical random draws and sets cuDNN flags."""
    import os
    import random

    import numpy as np
    import torch

    from src.utils.seed import get_torch_generator, set_global_seed

    returned_seed = set_global_seed(42)
    assert returned_seed == 42
    assert os.environ.get("PYTHONHASHSEED") == "42"
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    py_draw_1 = [random.random() for _ in range(5)]
    np_draw_1 = np.random.randn(8)
    torch_draw_1 = torch.randn(8)
    gen_draw_1 = torch.randn(8, generator=get_torch_generator(42))

    # Advance RNG states with arbitrary noise before reseeding
    _ = [random.random() for _ in range(10)]
    _ = np.random.randn(10)
    _ = torch.randn(10)

    set_global_seed(42)
    py_draw_2 = [random.random() for _ in range(5)]
    np_draw_2 = np.random.randn(8)
    torch_draw_2 = torch.randn(8)
    gen_draw_2 = torch.randn(8, generator=get_torch_generator(42))

    assert py_draw_1 == py_draw_2
    assert np.array_equal(np_draw_1, np_draw_2)
    assert torch.equal(torch_draw_1, torch_draw_2)
    assert torch.equal(gen_draw_1, gen_draw_2)


def test_set_global_seed_invalid_raises() -> None:
    """Verify DATA-01 AC 3: negative or non-integer seeds raise ValueError."""
    import pytest

    from src.utils.seed import get_torch_generator, set_global_seed

    for invalid_seed in (-1, -999, 3.14, "42", None, True, False):
        with pytest.raises(ValueError, match="non-negative integer"):
            set_global_seed(invalid_seed)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="non-negative integer"):
            get_torch_generator(invalid_seed)  # type: ignore[arg-type]

