"""Mode A: Native CIFAR-100 Semantic Hierarchy builder (Root -> Superclasses -> Fine Classes)."""

from __future__ import annotations

import numpy as np

from src.data.cifar100_taxonomy import CIFAR100Taxonomy, get_cifar100_taxonomy
from src.trees.base_tree import FastHierarchyTree


def _validate_class_to_indices(
    class_to_indices: dict[int, np.ndarray], allow_subclusters: bool = True
) -> dict[int, np.ndarray]:
    """Validate `class_to_indices` keys and non-empty 1D integer index arrays."""
    if not isinstance(class_to_indices, dict) or len(class_to_indices) == 0:
        raise ValueError("class_to_indices must be a non-empty dictionary mapping fine class IDs to index arrays.")

    cleaned: dict[int, np.ndarray] = {}
    for raw_cls_id, raw_indices in sorted(class_to_indices.items(), key=lambda item: int(item[0])):
        if isinstance(raw_cls_id, bool) or not isinstance(raw_cls_id, (int, np.integer)):
            raise ValueError(f"Class ID must be an integer in [0, 99], got {raw_cls_id!r}.")
        cls_id = int(raw_cls_id)
        if cls_id < 0 or (not allow_subclusters and cls_id >= 100):
            raise ValueError(f"CIFAR-100 fine class ID must be in range [0, 99], got {cls_id}.")

        arr = np.asarray(raw_indices)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError(f"Index array for class {cls_id} must be a non-empty 1D array.")
        if not np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.bool_:
            raise ValueError(f"Index array for class {cls_id} must have integer dtype, got {arr.dtype}.")

        cleaned[cls_id] = np.ascontiguousarray(arr, dtype=np.int64)

    return cleaned


def build_semantic_tree(
    class_to_indices: dict[int, np.ndarray],
    taxonomy: CIFAR100Taxonomy | None = None,
) -> FastHierarchyTree:
    r"""Construct the 3-level CIFAR-100 Semantic Hierarchy (`Root -> Superclasses -> Fine Classes`).

    Args:
        class_to_indices: Mapping from active fine class IDs (`0..99`) to 1D sample index arrays
            (e.g., `99` negative classes in Scenario 1 or `100` classes in Scenario 2).
        taxonomy: Optional `CIFAR100Taxonomy` instance; defaults to `get_cifar100_taxonomy()`.

    Returns:
        Validated `FastHierarchyTree` with `tree_type="semantic"` and `root_id=0`.
    """
    cleaned_pools = _validate_class_to_indices(class_to_indices, allow_subclusters=False)
    tax = taxonomy if taxonomy is not None else get_cifar100_taxonomy()

    active_superclasses: list[tuple[int, list[int]]] = []
    for coarse_id in range(len(tax.coarse_names)):
        active_fine = [fid for fid in tax.coarse_to_fine_ids[coarse_id] if fid in cleaned_pools]
        if active_fine:
            active_superclasses.append((coarse_id, active_fine))

    num_super = len(active_superclasses)
    num_leaves = len(cleaned_pools)
    total_nodes = 1 + num_super + num_leaves
    max_branching = max(num_super, max(len(fines) for _, fines in active_superclasses))

    parents = np.full(total_nodes, -1, dtype=np.int64)
    children = np.full((total_nodes, max_branching), -1, dtype=np.int64)
    child_counts = np.zeros(total_nodes, dtype=np.int64)
    is_leaf = np.zeros(total_nodes, dtype=bool)
    depths = np.zeros(total_nodes, dtype=np.int64)
    leaf_to_class_id = np.full(total_nodes, -1, dtype=np.int64)
    instance_counts = np.zeros(total_nodes, dtype=np.int64)
    visits = np.zeros(total_nodes, dtype=np.int64)
    q_values = np.zeros(total_nodes, dtype=np.float64)
    node_names: list[str] = ["root"] + [""] * (total_nodes - 1)

    next_leaf_id = 1 + num_super
    root_total_instances = 0

    for sc_offset, (coarse_id, active_fines) in enumerate(active_superclasses):
        sc_node_id = 1 + sc_offset
        parents[sc_node_id] = 0
        depths[sc_node_id] = 1
        node_names[sc_node_id] = f"superclass:{tax.coarse_names[coarse_id]}"
        children[0, sc_offset] = sc_node_id

        sc_instances = 0
        for leaf_offset, fine_id in enumerate(active_fines):
            leaf_node_id = next_leaf_id
            next_leaf_id += 1

            parents[leaf_node_id] = sc_node_id
            depths[leaf_node_id] = 2
            is_leaf[leaf_node_id] = True
            leaf_to_class_id[leaf_node_id] = fine_id
            node_names[leaf_node_id] = f"class:{tax.fine_names[fine_id]}"

            cnt = int(cleaned_pools[fine_id].size)
            instance_counts[leaf_node_id] = cnt
            sc_instances += cnt
            children[sc_node_id, leaf_offset] = leaf_node_id

        child_counts[sc_node_id] = len(active_fines)
        instance_counts[sc_node_id] = sc_instances
        root_total_instances += sc_instances

    child_counts[0] = num_super
    instance_counts[0] = root_total_instances

    return FastHierarchyTree(
        tree_type="semantic",
        root_id=0,
        parents=parents,
        children=children,
        child_counts=child_counts,
        is_leaf=is_leaf,
        depths=depths,
        leaf_to_class_id=leaf_to_class_id,
        instance_counts=instance_counts,
        visits=visits,
        q_values=q_values,
        node_names=tuple(node_names),
        class_to_indices=cleaned_pools,
    )
