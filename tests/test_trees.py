"""Unit and integration tests for Feature F2: Dual Hierarchy Construction Engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.trees.base_tree import FastHierarchyTree


def _make_sample_tree() -> FastHierarchyTree:
    """Build a small 3-level tree (0 -> [1, 2], 1 -> [3, 4], 2 -> [5]) with 3 leaves."""
    # Nodes:
    # 0: root (children: 1, 2), count = 30
    # 1: internal_A (children: 3, 4), count = 20
    # 2: internal_B (children: 5), count = 10
    # 3: leaf_class_0, count = 12
    # 4: leaf_class_1, count = 8
    # 5: leaf_class_2, count = 10
    parents = np.array([-1, 0, 0, 1, 1, 2], dtype=np.int64)
    children = np.array(
        [
            [1, 2],
            [3, 4],
            [5, -1],
            [-1, -1],
            [-1, -1],
            [-1, -1],
        ],
        dtype=np.int64,
    )
    child_counts = np.array([2, 2, 1, 0, 0, 0], dtype=np.int64)
    is_leaf = np.array([False, False, False, True, True, True], dtype=bool)
    depths = np.array([0, 1, 1, 2, 2, 2], dtype=np.int64)
    leaf_to_class_id = np.array([-1, -1, -1, 0, 1, 2], dtype=np.int64)
    instance_counts = np.array([30, 20, 10, 12, 8, 10], dtype=np.int64)
    visits = np.zeros(6, dtype=np.int64)
    q_values = np.zeros(6, dtype=np.float64)
    node_names = ("root", "super_0", "super_1", "class_0", "class_1", "class_2")
    class_to_indices = {
        0: np.arange(0, 12, dtype=np.int64),
        1: np.arange(12, 20, dtype=np.int64),
        2: np.arange(20, 30, dtype=np.int64),
    }
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
        node_names=node_names,
        class_to_indices=class_to_indices,
    )


def test_base_tree_contiguous_arrays_and_invariants() -> None:
    """C1: FastHierarchyTree stores attributes in C-contiguous NumPy arrays with exact dtypes."""
    tree = _make_sample_tree()
    assert tree.num_nodes == 6
    assert tree.root_id == 0

    for arr, expected_dtype in (
        (tree.parents, np.int64),
        (tree.children, np.int64),
        (tree.child_counts, np.int64),
        (tree.is_leaf, np.bool_),
        (tree.depths, np.int64),
        (tree.leaf_to_class_id, np.int64),
        (tree.instance_counts, np.int64),
        (tree.visits, np.int64),
        (tree.q_values, np.float64),
    ):
        assert isinstance(arr, np.ndarray)
        assert arr.flags["C_CONTIGUOUS"]
        assert arr.dtype == expected_dtype

    assert tree.children.shape == (6, 2)
    assert np.array_equal(tree.get_children(0), np.array([1, 2], dtype=np.int64))
    assert np.array_equal(tree.get_children(2), np.array([5], dtype=np.int64))
    assert np.array_equal(tree.get_children(3), np.array([], dtype=np.int64))
    assert np.array_equal(tree.leaf_node_ids(), np.array([3, 4, 5], dtype=np.int64))
    assert tree.class_id_to_leaf_node_id() == {0: 3, 1: 4, 2: 5}


def test_base_tree_validate_tree_reachability_and_counts() -> None:
    """C2: validate_tree() verifies root, cycle-free reachability, and bottom-up instance count sums."""
    tree = _make_sample_tree()
    tree.validate_tree()

    assert tree.parents[tree.root_id] == -1
    for node_id in range(tree.num_nodes):
        if not tree.is_leaf[node_id]:
            children = tree.get_children(node_id)
            assert children.size >= 1
            assert tree.instance_counts[node_id] == int(tree.instance_counts[children].sum())
        else:
            cls_id = int(tree.leaf_to_class_id[node_id])
            assert cls_id >= 0
            assert tree.instance_counts[node_id] == tree.class_to_indices[cls_id].size


def test_base_tree_invalid_invariants_raise() -> None:
    """C3: FastHierarchyTree raises ValueError on disconnected node, cycle, zero leaf count, or parent count mismatch."""
    # 1. Parent instance_count mismatch
    bad_count = _make_sample_tree()
    bad_count.instance_counts[0] = 999
    with pytest.raises(ValueError, match="instance_counts"):
        bad_count.validate_tree()

    # 2. Non-positive leaf instance_count
    bad_leaf = _make_sample_tree()
    bad_leaf.instance_counts[3] = 0
    bad_leaf.instance_counts[1] = 8
    bad_leaf.instance_counts[0] = 18
    with pytest.raises(ValueError, match="instance_counts"):
        bad_leaf.validate_tree()

    # 3. Disconnected node (node 2 not in root's children)
    disconnected = _make_sample_tree()
    disconnected.children[0, 1] = -1
    disconnected.child_counts[0] = 1
    with pytest.raises(ValueError, match="reachable|parent"):
        disconnected.validate_tree()

    # 4. Cycle in tree (node 3 points back to node 1 as child)
    cyclic = _make_sample_tree()
    cyclic.is_leaf[3] = False
    cyclic.leaf_to_class_id[3] = -1
    cyclic.children[3, 0] = 1
    cyclic.child_counts[3] = 1
    with pytest.raises(ValueError, match="cycle|parent|visited"):
        cyclic.validate_tree()

    # 5. Root parent != -1
    bad_root = _make_sample_tree()
    bad_root.parents[0] = 1
    with pytest.raises(ValueError, match="root"):
        bad_root.validate_tree()


def test_base_tree_reset_mcts_state() -> None:
    """C4: reset_mcts_state() zeros visits and q_values in-place while preserving topology and instance_counts."""
    tree = _make_sample_tree()
    visits_ref = tree.visits
    q_ref = tree.q_values
    orig_counts = tree.instance_counts.copy()
    orig_parents = tree.parents.copy()

    tree.visits[:] = np.array([10, 6, 4, 3, 3, 4], dtype=np.int64)
    tree.q_values[:] = np.array([0.5, 0.6, 0.4, 0.7, 0.5, 0.4], dtype=np.float64)

    tree.reset_mcts_state()

    assert tree.visits is visits_ref
    assert tree.q_values is q_ref
    assert np.all(tree.visits == 0)
    assert np.all(tree.q_values == 0.0)
    assert np.array_equal(tree.instance_counts, orig_counts)
    assert np.array_equal(tree.parents, orig_parents)


def test_base_tree_save_and_load_roundtrip(tmp_path: Path) -> None:
    """C5: tree.save(path) and FastHierarchyTree.load(path) persist and restore bit-identical arrays and index pools."""
    tree = _make_sample_tree()
    tree.visits[:] = np.array([5, 3, 2, 2, 1, 2], dtype=np.int64)
    tree.q_values[:] = np.array([0.42, 0.51, 0.30, 0.55, 0.43, 0.30], dtype=np.float64)

    out_path = tmp_path / "artifacts" / "sample_tree.npz"
    saved_path = tree.save(out_path)
    assert saved_path == out_path
    assert out_path.exists()
    assert not out_path.with_name(f"{out_path.stem}.tmp.npz").exists()

    loaded = FastHierarchyTree.load(out_path)
    assert loaded.tree_type == tree.tree_type
    assert loaded.root_id == tree.root_id
    assert loaded.node_names == tree.node_names
    assert np.array_equal(loaded.parents, tree.parents)
    assert np.array_equal(loaded.children, tree.children)
    assert np.array_equal(loaded.child_counts, tree.child_counts)
    assert np.array_equal(loaded.is_leaf, tree.is_leaf)
    assert np.array_equal(loaded.depths, tree.depths)
    assert np.array_equal(loaded.leaf_to_class_id, tree.leaf_to_class_id)
    assert np.array_equal(loaded.instance_counts, tree.instance_counts)
    assert np.array_equal(loaded.visits, tree.visits)
    assert np.allclose(loaded.q_values, tree.q_values, atol=0.0, rtol=0.0)
    assert set(loaded.class_to_indices.keys()) == set(tree.class_to_indices.keys())
    for k in tree.class_to_indices:
        assert np.array_equal(loaded.class_to_indices[k], tree.class_to_indices[k])


def test_semantic_tree_scenario2_100_classes() -> None:
    """C6: build_semantic_tree() on Scenario 2 (100 classes) builds a 3-level tree with 121 nodes."""
    from src.data.cifar100_taxonomy import get_cifar100_taxonomy
    from src.data.splits import build_scenario2_split
    from src.trees.semantic_tree import build_semantic_tree

    labels = np.repeat(np.arange(100, dtype=np.int64), 450)
    s2 = build_scenario2_split(labels)
    tree = build_semantic_tree(s2.class_to_indices)

    assert tree.tree_type == "semantic"
    assert tree.root_id == 0
    assert tree.num_nodes == 121
    assert tree.num_leaves == 100
    assert int(np.count_nonzero(tree.depths == 0)) == 1
    assert int(np.count_nonzero(tree.depths == 1)) == 20
    assert int(np.count_nonzero(tree.depths == 2)) == 100
    assert int(tree.instance_counts[0]) == 45000

    taxonomy = get_cifar100_taxonomy()
    root_children = tree.get_children(0)
    assert root_children.size == 20

    for idx, sc_node in enumerate(root_children):
        sc_children = tree.get_children(int(sc_node))
        assert sc_children.size == 5
        leaf_classes = tuple(sorted(int(tree.leaf_to_class_id[c]) for c in sc_children))
        assert leaf_classes == taxonomy.coarse_to_fine_ids[idx]
        assert int(tree.instance_counts[sc_node]) == 5 * 450


def test_semantic_tree_scenario1_99_negative_classes() -> None:
    """C7: build_semantic_tree() on Scenario 1 (99 negative classes) builds a 3-level tree with 120 nodes."""
    from src.data.cifar100_taxonomy import get_cifar100_taxonomy
    from src.data.splits import build_scenario1_split
    from src.trees.semantic_tree import build_semantic_tree

    labels = np.repeat(np.arange(100, dtype=np.int64), 450)
    s1 = build_scenario1_split(labels, target_class="shark", pos_fraction=0.02, seed=42)
    tree = build_semantic_tree(s1.negative_class_to_indices)

    assert tree.num_nodes == 120
    assert tree.num_leaves == 99
    assert int(tree.instance_counts[0]) == 99 * 450
    assert s1.target_class_id not in tree.class_id_to_leaf_node_id()

    taxonomy = get_cifar100_taxonomy()
    target_coarse = int(taxonomy.fine_to_coarse_idx[s1.target_class_id])
    root_children = tree.get_children(0)
    assert root_children.size == 20

    child_counts_at_depth1 = [int(tree.child_counts[sc]) for sc in root_children]
    assert child_counts_at_depth1.count(4) == 1
    assert child_counts_at_depth1.count(5) == 19
    assert int(tree.child_counts[root_children[target_coarse]]) == 4


def test_semantic_tree_invalid_inputs_and_empty_superclass() -> None:
    """C8: build_semantic_tree() raises ValueError on invalid inputs and omits empty superclasses."""
    from src.trees.semantic_tree import build_semantic_tree

    with pytest.raises(ValueError, match="non-empty"):
        build_semantic_tree({})

    with pytest.raises(ValueError, match="range"):
        build_semantic_tree({100: np.array([0, 1], dtype=np.int64)})

    with pytest.raises(ValueError, match="empty"):
        build_semantic_tree({0: np.array([], dtype=np.int64)})

    # Subset covering only 2 fine classes from superclass 0 ("beaver"=4, "dolphin"=30)
    subset_tree = build_semantic_tree(
        {
            4: np.array([0, 1, 2], dtype=np.int64),
            30: np.array([3, 4], dtype=np.int64),
        }
    )
    assert subset_tree.num_nodes == 4  # 1 root + 1 active superclass + 2 leaves
    assert subset_tree.num_leaves == 2
    assert int(subset_tree.instance_counts[0]) == 5

