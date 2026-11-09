"""Imbalanced Scenario 1 (One-vs-Rest) and Scenario 2 (100-way Multi-class) splitters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.data.cifar100_taxonomy import CIFAR100_FINE_CLASSES, get_cifar100_taxonomy
from src.utils.seed import set_global_seed

DEFAULT_SCENARIO1_TARGETS: tuple[str, ...] = (
    "beaver",
    "oak_tree",
    "pickup_truck",
    "rocket",
    "shark",
)

IMBALANCE_RATIOS: dict[str, float] = {
    "1:10": 0.10,
    "1:50": 0.02,
    "1:100": 0.01,
}


@dataclass(frozen=True)
class Scenario1Split:
    """Imbalanced Binary One-vs-Rest split configuration and index pools."""

    target_class_id: int
    target_class_name: str
    pos_fraction: float
    positive_indices: np.ndarray
    negative_indices: np.ndarray
    negative_class_to_indices: dict[int, np.ndarray]
    binary_labels: np.ndarray


@dataclass(frozen=True)
class Scenario2Split:
    """100-way Multi-class curriculum index pools across all CIFAR-100 fine classes."""

    class_to_indices: dict[int, np.ndarray]
    active_indices: np.ndarray


def _resolve_target_class(target_class: int | str) -> tuple[int, str]:
    """Resolve `target_class` as either an integer index in `0..99` or a canonical fine class name."""
    taxonomy = get_cifar100_taxonomy()
    if isinstance(target_class, str):
        if target_class not in taxonomy.fine_name_to_id:
            raise ValueError(
                f"Unknown CIFAR-100 target class name '{target_class}'. Must be one of CIFAR100_FINE_CLASSES."
            )
        cls_id = taxonomy.fine_name_to_id[target_class]
        return cls_id, target_class

    if isinstance(target_class, bool) or not isinstance(target_class, (int, np.integer)):
        raise ValueError(f"target_class must be an int in [0, 99] or str, got {target_class!r}.")

    cls_id = int(target_class)
    if cls_id < 0 or cls_id >= len(CIFAR100_FINE_CLASSES):
        raise ValueError(f"target_class index must be in [0, 99], got {cls_id}.")
    return cls_id, CIFAR100_FINE_CLASSES[cls_id]


def _to_1d_label_array(fine_labels: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a 1D tensor or array of fine labels in `0..99` to an `int64` NumPy array."""
    if isinstance(fine_labels, torch.Tensor):
        arr = fine_labels.detach().cpu().numpy()
    elif isinstance(fine_labels, np.ndarray):
        arr = fine_labels
    else:
        raise ValueError(f"Expected torch.Tensor or np.ndarray for fine_labels, got {type(fine_labels).__name__}.")

    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("fine_labels must be a non-empty 1D tensor or ndarray.")
    if not np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.bool_:
        raise ValueError(f"fine_labels must have integer dtype, got {arr.dtype}.")

    min_lbl = int(arr.min())
    max_lbl = int(arr.max())
    if min_lbl < 0 or max_lbl >= 100:
        raise ValueError(f"fine_labels values must be in [0, 99], got range [{min_lbl}, {max_lbl}].")

    return arr.astype(np.int64, copy=False)


def build_scenario1_split(
    fine_labels: torch.Tensor | np.ndarray,
    target_class: int | str,
    pos_fraction: float = 0.10,
    seed: int = 42,
) -> Scenario1Split:
    """Construct a Scenario 1 (Imbalanced Binary One-vs-Rest) split for `target_class`.

    Args:
        fine_labels: 1D tensor or array of CIFAR-100 fine labels (`0..99`).
        target_class: Target positive fine class index (`0..99`) or name (e.g. `"shark"`).
        pos_fraction: Fraction of target class instances to retain in `(0.0, 1.0]`.
        seed: Random seed for deterministic positive subsampling.

    Returns:
        `Scenario1Split` containing subsampled positive indices and all 99 negative class pools.
    """
    set_global_seed(seed)
    target_id, target_name = _resolve_target_class(target_class)

    if isinstance(pos_fraction, bool) or not isinstance(pos_fraction, (int, float)):
        raise ValueError(f"pos_fraction must be a float in (0.0, 1.0], got {pos_fraction!r}.")
    frac = float(pos_fraction)
    if frac <= 0.0 or frac > 1.0:
        raise ValueError(f"pos_fraction must satisfy 0.0 < pos_fraction <= 1.0, got {frac}.")

    labels_np = _to_1d_label_array(fine_labels)
    all_target_indices = np.flatnonzero(labels_np == target_id)
    if all_target_indices.size == 0:
        raise ValueError(f"Target class {target_id} ('{target_name}') has 0 instances in fine_labels.")

    n_pos = max(1, int(np.floor(frac * all_target_indices.size)))
    rng = np.random.default_rng(seed)
    chosen_pos = np.sort(rng.choice(all_target_indices, size=n_pos, replace=False)).astype(np.int64)

    negative_class_to_indices: dict[int, np.ndarray] = {}
    neg_parts: list[np.ndarray] = []
    for cls_id in range(100):
        if cls_id == target_id:
            continue
        cls_indices = np.flatnonzero(labels_np == cls_id).astype(np.int64)
        if cls_indices.size == 0:
            raise ValueError(f"Negative class {cls_id} has 0 instances in fine_labels.")
        negative_class_to_indices[cls_id] = cls_indices
        neg_parts.append(cls_indices)

    negative_indices = np.sort(np.concatenate(neg_parts)).astype(np.int64)
    binary_labels = (labels_np == target_id).astype(np.int64)

    return Scenario1Split(
        target_class_id=target_id,
        target_class_name=target_name,
        pos_fraction=frac,
        positive_indices=chosen_pos,
        negative_indices=negative_indices,
        negative_class_to_indices=negative_class_to_indices,
        binary_labels=binary_labels,
    )


def build_scenario2_split(fine_labels: torch.Tensor | np.ndarray) -> Scenario2Split:
    """Construct a Scenario 2 (100-way Multi-class Curriculum) index pool across all 100 fine classes."""
    labels_np = _to_1d_label_array(fine_labels)
    class_to_indices: dict[int, np.ndarray] = {}

    for cls_id in range(100):
        cls_indices = np.flatnonzero(labels_np == cls_id).astype(np.int64)
        if cls_indices.size == 0:
            raise ValueError(f"Fine class {cls_id} is missing (0 samples) in fine_labels.")
        class_to_indices[cls_id] = cls_indices

    active_indices = np.arange(labels_np.size, dtype=np.int64)
    return Scenario2Split(
        class_to_indices=class_to_indices,
        active_indices=active_indices,
    )
