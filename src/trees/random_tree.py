"""Random balanced hierarchy baseline builder matching CIFAR-100 semantic topology (Phase C5)."""

from __future__ import annotations

import numpy as np

from src.trees.base_tree import FastHierarchyTree
from src.trees.semantic_tree import _validate_class_to_indices


def build_random_tree(
    class_to_indices: dict[int, np.ndarray],
    num_superclasses: int = 20,
    seed: int = 42,
) -> FastHierarchyTree:
    r"""Construct a randomized 3-level balanced hierarchy tree (`Root -> Random Groups -> Classes`).

    Preserves the depth and branching profile of the CIFAR-100 Semantic Tree:
    - Root node at depth 0 (node 0)
    - `num_superclasses` internal nodes at depth 1
    - $K$ leaf nodes at depth 2 evenly distributed among the superclasses.

    Args:
        class_to_indices: Mapping from class ID to 1D integer index array.
        num_superclasses: Number of random superclass groups at depth 1 (default: 20).
        seed: Random seed for deterministic class assignment.

    Returns:
        Validated `FastHierarchyTree` with `tree_type="random_balanced"` and `root_id=0`.
    """
    cleaned_pools = _validate_class_to_indices(class_to_indices, allow_subclusters=False)
    num_leaves = len(cleaned_pools)
    if num_leaves < num_superclasses:
        num_super = max(1, num_leaves // 2)
    else:
        num_super = int(num_superclasses)

    total_nodes = 1 + num_super + num_leaves

    rng = np.random.default_rng(seed)
    sorted_classes = sorted(cleaned_pools.keys())
    shuffled_classes = list(sorted_classes)
    rng.shuffle(shuffled_classes)

    # Distribute classes evenly across superclasses
    super_to_classes: list[list[int]] = [[] for _ in range(num_super)]
    for i, cid in enumerate(shuffled_classes):
        super_to_classes[i % num_super].append(cid)

    max_branching = max(num_super, max(len(c_list) for c_list in super_to_classes))

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

    for sc_idx, class_list in enumerate(super_to_classes):
        sc_node_id = 1 + sc_idx
        parents[sc_node_id] = 0
        depths[sc_node_id] = 1
        node_names[sc_node_id] = f"random_group:{sc_idx}"
        children[0, sc_idx] = sc_node_id

        sc_instances = 0
        for leaf_offset, cid in enumerate(class_list):
            leaf_node_id = next_leaf_id
            next_leaf_id += 1

            parents[leaf_node_id] = sc_node_id
            depths[leaf_node_id] = 2
            is_leaf[leaf_node_id] = True
            leaf_to_class_id[leaf_node_id] = cid
            node_names[leaf_node_id] = f"class:{cid}"

            cnt = int(cleaned_pools[cid].size)
            instance_counts[leaf_node_id] = cnt
            sc_instances += cnt
            children[sc_node_id, leaf_offset] = leaf_node_id

        child_counts[sc_node_id] = len(class_list)
        instance_counts[sc_node_id] = sc_instances
        root_total_instances += sc_instances

    child_counts[0] = num_super
    instance_counts[0] = root_total_instances

    return FastHierarchyTree(
        tree_type="random_balanced",
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
