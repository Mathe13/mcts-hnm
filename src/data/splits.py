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
class StratifiedImbalancedSplit:
    """Operational Imbalanced One-vs-Rest split with fixed unique positives and real negative ratio."""

    target_class_id: int
    target_class_name: str
    target_positive_count: int
    target_neg_pos_ratio: int
    positive_indices: np.ndarray
    negative_indices: np.ndarray
    negative_class_to_indices: dict[int, np.ndarray]
    binary_labels: np.ndarray
    effective_train_ratio: float


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
    classes_to_check = (
        range(100)
        if (int(labels_np.max()) >= 100 or len(np.unique(labels_np)) == 100)
        else [int(c) for c in np.unique(labels_np) if int(c) != target_id]
    )
    for cls_id in classes_to_check:
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


def build_stratified_imbalanced_split(
    fine_labels: torch.Tensor | np.ndarray,
    target_class: int | str,
    target_positive_count: int = 80,
    target_neg_pos_ratio: int = 50,
    seed: int = 42,
    fixed_positive_indices: np.ndarray | None = None,
    nested_negative_parent: dict[int, np.ndarray] | None = None,
) -> StratifiedImbalancedSplit:
    r"""Construct an operational imbalanced split with fixed unique positives and real negative ratio.

    Distinguishes the training set real ratio ($N_{\text{neg}}/N_{\text{pos}}$), the unique positive
    count, and minibatch composition, fulfilling Roadmap v2 Phase A Item A2.

    Args:
        fine_labels: 1D tensor or array of CIFAR-100 fine labels (`0..99`).
        target_class: Target positive fine class index (`0..99`) or name.
        target_positive_count: Number of unique positive instances to select (default: 80).
        target_neg_pos_ratio: Real ratio of negative to positive instances in the training set (e.g. 50, 100, 500).
        seed: Random seed for deterministic selection.
        fixed_positive_indices: Optional pre-selected positive indices to guarantee shared positives across ratios.
        nested_negative_parent: Optional dictionary mapping each negative class to available parent indices,
            ensuring nested sampling $N_{50} \subset N_{100} \subset N_{500}$.

    Returns:
        `StratifiedImbalancedSplit` containing exact positive and negative pools.
    """
    target_id, target_name = _resolve_target_class(target_class)
    labels_np = _to_1d_label_array(fine_labels)

    if target_positive_count <= 0:
        raise ValueError(f"target_positive_count must be > 0, got {target_positive_count}.")
    if target_neg_pos_ratio <= 0:
        raise ValueError(f"target_neg_pos_ratio must be > 0, got {target_neg_pos_ratio}.")

    rng = np.random.default_rng(seed)

    # 1. Resolve positive instances
    if fixed_positive_indices is not None:
        chosen_pos = np.sort(np.asarray(fixed_positive_indices, dtype=np.int64))
        if len(chosen_pos) != target_positive_count:
            raise ValueError(
                f"fixed_positive_indices length ({len(chosen_pos)}) does not match target_positive_count ({target_positive_count})."
            )
    else:
        all_target_indices = np.flatnonzero(labels_np == target_id)
        if len(all_target_indices) < target_positive_count:
            raise ValueError(
                f"Insufficient positive instances for class {target_id} ('{target_name}'): "
                f"requested {target_positive_count}, available {len(all_target_indices)}."
            )
        chosen_pos = np.sort(rng.choice(all_target_indices, size=target_positive_count, replace=False)).astype(np.int64)

    # 2. Resolve negative instances with stratification across all 99 negative classes
    total_negatives_needed = target_positive_count * target_neg_pos_ratio

    # Determine pools per negative class
    available_negative_pools: dict[int, np.ndarray] = {}
    classes_to_check = (
        range(100)
        if (int(labels_np.max()) >= 100 or len(np.unique(labels_np)) == 100)
        else [int(c) for c in np.unique(labels_np) if int(c) != target_id]
    )

    total_available_negatives = 0
    for cls_id in classes_to_check:
        if cls_id == target_id:
            continue
        if nested_negative_parent is not None and cls_id in nested_negative_parent:
            pool = np.asarray(nested_negative_parent[cls_id], dtype=np.int64)
        else:
            pool = np.flatnonzero(labels_np == cls_id).astype(np.int64)
        available_negative_pools[cls_id] = pool
        total_available_negatives += len(pool)

    if total_available_negatives < total_negatives_needed:
        raise ValueError(
            f"Insufficient negative instances across negative classes: "
            f"requested {total_negatives_needed} (80 * {target_neg_pos_ratio}), "
            f"available {total_available_negatives}."
        )

    num_neg_classes = len(available_negative_pools)
    if num_neg_classes == 0:
        raise ValueError("No negative classes available to sample from.")

    # Allocate counts per negative class (stratified quotient + remainder)
    base_per_class = total_negatives_needed // num_neg_classes
    remainder = total_negatives_needed % num_neg_classes

    # Determine which classes receive the extra +1 remainder sample
    neg_class_ids = sorted(available_negative_pools.keys())
    extra_indices = set(rng.choice(len(neg_class_ids), size=remainder, replace=False).tolist())

    negative_class_to_indices: dict[int, np.ndarray] = {}
    all_neg_parts: list[np.ndarray] = []

    for idx, cls_id in enumerate(neg_class_ids):
        pool = available_negative_pools[cls_id]
        count_for_class = base_per_class + (1 if idx in extra_indices else 0)
        if len(pool) < count_for_class:
            raise ValueError(
                f"Negative class {cls_id} has insufficient instances for stratified quota: "
                f"needed {count_for_class}, available {len(pool)}."
            )
        chosen_cls = np.sort(rng.choice(pool, size=count_for_class, replace=False)).astype(np.int64)
        negative_class_to_indices[cls_id] = chosen_cls
        all_neg_parts.append(chosen_cls)

    negative_indices = np.sort(np.concatenate(all_neg_parts)).astype(np.int64)
    effective_ratio = float(len(negative_indices) / len(chosen_pos)) if len(chosen_pos) > 0 else 0.0

    binary_labels = np.zeros(len(labels_np), dtype=np.int64)
    binary_labels[chosen_pos] = 1

    return StratifiedImbalancedSplit(
        target_class_id=target_id,
        target_class_name=target_name,
        target_positive_count=target_positive_count,
        target_neg_pos_ratio=target_neg_pos_ratio,
        positive_indices=chosen_pos,
        negative_indices=negative_indices,
        negative_class_to_indices=negative_class_to_indices,
        binary_labels=binary_labels,
        effective_train_ratio=effective_ratio,
    )


def build_nested_imbalanced_splits(
    fine_labels: torch.Tensor | np.ndarray,
    target_class: int | str,
    target_positive_count: int = 80,
    ratios: tuple[int, ...] = (50, 100, 500),
    seed: int = 42,
) -> dict[int, StratifiedImbalancedSplit]:
    r"""Construct a family of nested imbalanced splits sharing identical positives and nested negatives.

    Guarantees:
    1. Positive instances are identical for all ratios under the same seed and target class.
    2. Negative sets are strictly nested: for sorted ratios $r_1 < r_2 < \dots < r_m$,
       $N_{r_1} \subset N_{r_2} \subset \dots \subset N_{r_m}$.

    Args:
        fine_labels: 1D tensor or array of CIFAR-100 fine labels (`0..99`).
        target_class: Target positive fine class index or name.
        target_positive_count: Number of unique positives (default: 80).
        ratios: Tuple of negative-to-positive integer ratios (e.g. `(50, 100, 500)`).
        seed: Random seed.

    Returns:
        Dictionary mapping ratio -> `StratifiedImbalancedSplit`.
    """
    sorted_desc = sorted(ratios, reverse=True)
    if len(sorted_desc) == 0:
        raise ValueError("ratios must not be empty.")

    # 1. First sample the positive instances once
    base_split = build_stratified_imbalanced_split(
        fine_labels=fine_labels,
        target_class=target_class,
        target_positive_count=target_positive_count,
        target_neg_pos_ratio=sorted_desc[0],
        seed=seed,
    )
    shared_positives = base_split.positive_indices

    results: dict[int, StratifiedImbalancedSplit] = {
        sorted_desc[0]: base_split
    }

    # 2. Iteratively subsample smaller negative ratios strictly from the parent negative pools
    parent_pools = base_split.negative_class_to_indices

    for ratio in sorted_desc[1:]:
        sub_split = build_stratified_imbalanced_split(
            fine_labels=fine_labels,
            target_class=target_class,
            target_positive_count=target_positive_count,
            target_neg_pos_ratio=ratio,
            seed=seed,
            fixed_positive_indices=shared_positives,
            nested_negative_parent=parent_pools,
        )
        results[ratio] = sub_split
        # The new pool becomes the parent for even smaller ratios
        parent_pools = sub_split.negative_class_to_indices

    return results
