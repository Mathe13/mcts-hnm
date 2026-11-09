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


def test_cifar100_taxonomy_bijection() -> None:
    """Verify DATA-01 AC 4: 100 fine classes map bijectively into 20 superclasses of 5 classes each."""
    import numpy as np

    from src.data.cifar100_taxonomy import (
        CIFAR100_COARSE_CLASSES,
        CIFAR100_FINE_CLASSES,
        get_cifar100_taxonomy,
    )

    taxonomy = get_cifar100_taxonomy()
    assert len(taxonomy.fine_names) == 100
    assert len(taxonomy.coarse_names) == 20
    assert list(taxonomy.fine_names) == sorted(CIFAR100_FINE_CLASSES)
    assert len(set(taxonomy.fine_names)) == 100
    assert len(set(CIFAR100_COARSE_CLASSES)) == 20

    assert taxonomy.fine_to_coarse_idx.shape == (100,)
    assert taxonomy.fine_to_coarse_idx.min() == 0
    assert taxonomy.fine_to_coarse_idx.max() == 19

    seen_fine_ids: set[int] = set()
    for coarse_id in range(20):
        fine_ids = taxonomy.coarse_to_fine_ids[coarse_id]
        assert len(fine_ids) == 5, f"Superclass {coarse_id} must have 5 fine classes"
        for fid in fine_ids:
            assert taxonomy.fine_to_coarse_idx[fid] == coarse_id
            seen_fine_ids.add(fid)

    assert seen_fine_ids == set(range(100))

    # Spot-check canonical CIFAR-100 superclass memberships
    beaver_id = taxonomy.fine_name_to_id["beaver"]
    shark_id = taxonomy.fine_name_to_id["shark"]
    rocket_id = taxonomy.fine_name_to_id["rocket"]
    assert taxonomy.coarse_names[int(taxonomy.fine_to_coarse_idx[beaver_id])] == "aquatic_mammals"
    assert taxonomy.coarse_names[int(taxonomy.fine_to_coarse_idx[shark_id])] == "fish"
    assert taxonomy.coarse_names[int(taxonomy.fine_to_coarse_idx[rocket_id])] == "vehicles_2"
    assert np.bincount(taxonomy.fine_to_coarse_idx).tolist() == [5] * 20


def test_fine_to_coarse_tensor_conversion() -> None:
    """Verify DATA-01 AC 5: fine_to_coarse_tensor maps 1D/2D tensors and arrays preserving shape and device."""
    import numpy as np
    import torch

    from src.data.cifar100_taxonomy import fine_to_coarse_tensor, get_cifar100_taxonomy

    tax = get_cifar100_taxonomy()
    fine_np = np.array([4, 52, 58, 69, 73], dtype=np.int64)  # beaver, oak_tree, pickup_truck, rocket, shark
    coarse_np = fine_to_coarse_tensor(fine_np)
    assert isinstance(coarse_np, np.ndarray)
    assert coarse_np.shape == fine_np.shape
    expected = [
        tax.coarse_name_to_id["aquatic_mammals"],
        tax.coarse_name_to_id["trees"],
        tax.coarse_name_to_id["vehicles_1"],
        tax.coarse_name_to_id["vehicles_2"],
        tax.coarse_name_to_id["fish"],
    ]
    assert coarse_np.tolist() == expected

    fine_t = torch.tensor([[4, 52], [58, 73]], dtype=torch.int32)
    coarse_t = fine_to_coarse_tensor(fine_t)
    assert isinstance(coarse_t, torch.Tensor)
    assert coarse_t.shape == (2, 2)
    assert coarse_t.device == fine_t.device
    assert coarse_t.dtype == torch.long
    assert coarse_t.tolist() == [[expected[0], expected[1]], [expected[2], expected[4]]]


def test_fine_to_coarse_out_of_range_raises() -> None:
    """Verify DATA-01 Edge Case 1: out-of-range or non-integer labels raise ValueError."""
    import numpy as np
    import pytest
    import torch

    from src.data.cifar100_taxonomy import fine_to_coarse_tensor

    for bad_arr in (np.array([-1, 0]), np.array([0, 100]), np.array([150])):
        with pytest.raises(ValueError, match=r"\[0, 99\]"):
            fine_to_coarse_tensor(bad_arr)

    for bad_tensor in (torch.tensor([-5, 10]), torch.tensor([0, 100])):
        with pytest.raises(ValueError, match=r"\[0, 99\]"):
            fine_to_coarse_tensor(bad_tensor)

    with pytest.raises(ValueError, match="integer"):
        fine_to_coarse_tensor(torch.tensor([1.5, 2.0]))
    with pytest.raises(ValueError, match="integer"):
        fine_to_coarse_tensor(np.array([True, False]))


