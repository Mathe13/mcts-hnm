"""Deterministic random seed management across Python, NumPy, and PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def _validate_seed(seed: int) -> int:
    """Validate that `seed` is a non-negative integer (excluding booleans)."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"Seed must be a non-negative integer, got {seed!r} ({type(seed).__name__}).")
    return seed


def set_global_seed(seed: int, deterministic_cudnn: bool = True) -> int:
    """Seed Python `random`, `PYTHONHASHSEED`, `numpy`, and `torch` RNGs deterministically.

    Args:
        seed: Non-negative integer seed value.
        deterministic_cudnn: If True, forces deterministic cuDNN algorithms and disables benchmark.

    Returns:
        The validated integer seed.

    Raises:
        ValueError: If `seed` is negative, a boolean, or not an integer.
    """
    validated = _validate_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(validated)
    random.seed(validated)
    np.random.seed(validated)
    torch.manual_seed(validated)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(validated)
        torch.cuda.manual_seed_all(validated)

    if deterministic_cudnn and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return validated


def get_torch_generator(seed: int, device: str = "cpu") -> torch.Generator:
    """Return a deterministically seeded `torch.Generator` instance."""
    validated = _validate_seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(validated)
    return generator
