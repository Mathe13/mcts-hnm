"""Vectorized hierarchy tree data structure backed by contiguous NumPy arrays."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass
class FastHierarchyTree:
    r"""Contiguous NumPy-backed tree container for zero-overhead MCTS traversal.

    Attributes:
        tree_type: Identifier of the tree origin (`"semantic"` or `"confusion_ward"`).
        root_id: Integer index of the root node (conventionally `0`).
        parents: 1D `int64` array of shape `[V]` (`parents[root_id] == -1`).
        children: 2D `int64` array of shape `[V, B_max]` padded with `-1`.
        child_counts: 1D `int64` array of shape `[V]` giving valid child count per node.
        is_leaf: 1D `bool` array of shape `[V]`.
        depths: 1D `int64` array of shape `[V]` (`depths[root_id] == 0`).
        leaf_to_class_id: 1D `int64` array of shape `[V]` (`>= 0` for leaves, `-1` for internal).
        instance_counts: 1D `int64` array of shape `[V]` ($M_p = \sum M_c$).
        visits: 1D `int64` array of shape `[V]` tracking MCTS visit counts $n_i$.
        q_values: 1D `float64` array of shape `[V]` tracking MCTS action-value estimates $Q_i$.
        node_names: Tuple of human-readable node labels of length `V`.
        class_to_indices: Mapping from each active fine class ID to its 1D `int64` sample index array.
    """

    tree_type: str
    root_id: int
    parents: np.ndarray
    children: np.ndarray
    child_counts: np.ndarray
    is_leaf: np.ndarray
    depths: np.ndarray
    leaf_to_class_id: np.ndarray
    instance_counts: np.ndarray
    visits: np.ndarray
    q_values: np.ndarray
    node_names: tuple[str, ...]
    class_to_indices: dict[int, np.ndarray]

    def __post_init__(self) -> None:
        """Enforce contiguous memory layout, exact dtypes, and structural invariants."""
        self.root_id = int(self.root_id)
        self.tree_type = str(self.tree_type)
        self.parents = np.ascontiguousarray(self.parents, dtype=np.int64)
        self.children = np.ascontiguousarray(self.children, dtype=np.int64)
        self.child_counts = np.ascontiguousarray(self.child_counts, dtype=np.int64)
        self.is_leaf = np.ascontiguousarray(self.is_leaf, dtype=bool)
        self.depths = np.ascontiguousarray(self.depths, dtype=np.int64)
        self.leaf_to_class_id = np.ascontiguousarray(self.leaf_to_class_id, dtype=np.int64)
        self.instance_counts = np.ascontiguousarray(self.instance_counts, dtype=np.int64)
        self.visits = np.ascontiguousarray(self.visits, dtype=np.int64)
        self.q_values = np.ascontiguousarray(self.q_values, dtype=np.float64)
        self.node_names = tuple(str(name) for name in self.node_names)
        self.class_to_indices = {
            int(k): np.ascontiguousarray(v, dtype=np.int64)
            for k, v in sorted(self.class_to_indices.items(), key=lambda item: int(item[0]))
        }
        self.validate_tree()

    @property
    def num_nodes(self) -> int:
        """Total number of nodes `V` in the tree."""
        return int(self.parents.shape[0])

    @property
    def num_leaves(self) -> int:
        """Total number of leaf nodes in the tree."""
        return int(np.count_nonzero(self.is_leaf))

    def get_children(self, node_id: int) -> np.ndarray:
        """Return the 1D `int64` slice of valid child node IDs for `node_id`."""
        if node_id < 0 or node_id >= self.num_nodes:
            raise ValueError(f"node_id {node_id} out of range [0, {self.num_nodes - 1}].")
        count = int(self.child_counts[node_id])
        return self.children[node_id, :count]

    def leaf_node_ids(self) -> np.ndarray:
        """Return 1D `int64` array of all leaf node IDs."""
        return np.flatnonzero(self.is_leaf).astype(np.int64)

    def class_id_to_leaf_node_id(self) -> dict[int, int]:
        """Return mapping from active fine class ID to its corresponding leaf node ID."""
        leaf_ids = self.leaf_node_ids()
        return {int(self.leaf_to_class_id[nid]): int(nid) for nid in leaf_ids}

    def reset_mcts_state(self) -> None:
        """Reset MCTS `visits` (`0`) and `q_values` (`0.0`) across all nodes in-place."""
        self.visits.fill(0)
        self.q_values.fill(0.0)

    def validate_tree(self) -> None:
        """Verify structural invariants, cycle-free reachability, and instance count conservation."""
        v = self.num_nodes
        if v < 2:
            raise ValueError(f"FastHierarchyTree must have at least 2 nodes, got {v}.")

        for name, arr in (
            ("parents", self.parents),
            ("child_counts", self.child_counts),
            ("is_leaf", self.is_leaf),
            ("depths", self.depths),
            ("leaf_to_class_id", self.leaf_to_class_id),
            ("instance_counts", self.instance_counts),
            ("visits", self.visits),
            ("q_values", self.q_values),
        ):
            if arr.ndim != 1 or arr.shape[0] != v:
                raise ValueError(f"Array '{name}' must have shape ({v},), got {arr.shape}.")

        if self.children.ndim != 2 or self.children.shape[0] != v:
            raise ValueError(
                f"Array 'children' must have 2D shape ({v}, B_max), got {self.children.shape}."
            )

        if len(self.node_names) != v:
            raise ValueError(f"node_names length ({len(self.node_names)}) must equal num_nodes ({v}).")

        if self.root_id < 0 or self.root_id >= v:
            raise ValueError(f"Invalid root_id={self.root_id} for tree with {v} nodes.")
        if int(self.parents[self.root_id]) != -1:
            raise ValueError(
                f"root_id={self.root_id} must have parents[root_id] == -1, got {self.parents[self.root_id]}."
            )
        if int(self.depths[self.root_id]) != 0:
            raise ValueError(f"root_id={self.root_id} must have depth 0, got {self.depths[self.root_id]}.")

        b_max = self.children.shape[1]
        visited = np.zeros(v, dtype=bool)
        stack: list[int] = [self.root_id]

        while stack:
            nid = stack.pop()
            if visited[nid]:
                raise ValueError(f"Cycle or duplicate parent edge detected at node {nid}.")
            visited[nid] = True

            c_count = int(self.child_counts[nid])
            if c_count < 0 or c_count > b_max:
                raise ValueError(f"Invalid child_counts[{nid}]={c_count} (max_branching={b_max}).")

            if bool(self.is_leaf[nid]):
                if c_count != 0:
                    raise ValueError(f"Leaf node {nid} must have child_counts == 0, got {c_count}.")
                if not np.all(self.children[nid] == -1):
                    raise ValueError(f"Leaf node {nid} must have all children entries set to -1.")
                cls_id = int(self.leaf_to_class_id[nid])
                if cls_id < 0:
                    raise ValueError(f"Leaf node {nid} must have leaf_to_class_id >= 0, got {cls_id}.")
                if cls_id not in self.class_to_indices:
                    raise ValueError(f"Leaf node {nid} references class_id {cls_id} absent from class_to_indices.")
                inst_cnt = int(self.instance_counts[nid])
                if inst_cnt <= 0:
                    raise ValueError(
                        f"Leaf node {nid} (class {cls_id}) must have instance_counts > 0, got {inst_cnt}."
                    )
                if inst_cnt != int(self.class_to_indices[cls_id].size):
                    raise ValueError(
                        f"Leaf node {nid} instance_counts ({inst_cnt}) does not match "
                        f"class_to_indices[{cls_id}] length ({self.class_to_indices[cls_id].size})."
                    )
            else:
                if c_count < 1:
                    raise ValueError(f"Internal node {nid} must have at least 1 child, got {c_count}.")
                if int(self.leaf_to_class_id[nid]) != -1:
                    raise ValueError(
                        f"Internal node {nid} must have leaf_to_class_id == -1, got {self.leaf_to_class_id[nid]}."
                    )
                if c_count < b_max and not np.all(self.children[nid, c_count:] == -1):
                    raise ValueError(f"Internal node {nid} padding columns in children must be -1.")

                child_slice = self.children[nid, :c_count]
                for child_raw in child_slice:
                    cid = int(child_raw)
                    if cid < 0 or cid >= v:
                        raise ValueError(f"Node {nid} has out-of-range child ID {cid}.")
                    if visited[cid]:
                        raise ValueError(f"Cycle detected: node {nid} links to already-visited node {cid}.")
                    if int(self.parents[cid]) != nid:
                        raise ValueError(
                            f"Parent-child mismatch: node {nid} lists child {cid}, "
                            f"but parents[{cid}] == {self.parents[cid]}."
                        )
                    if int(self.depths[cid]) != int(self.depths[nid]) + 1:
                        raise ValueError(
                            f"Depth mismatch: child {cid} depth ({self.depths[cid]}) != "
                            f"parent {nid} depth ({self.depths[nid]}) + 1."
                        )
                    stack.append(cid)

        if not np.all(visited):
            unreachable = np.flatnonzero(~visited).tolist()
            raise ValueError(f"Tree contains nodes not reachable from root {self.root_id}: {unreachable}.")

        for nid in range(v):
            if not bool(self.is_leaf[nid]):
                c_count = int(self.child_counts[nid])
                child_slice = self.children[nid, :c_count]
                expected_sum = int(self.instance_counts[child_slice].sum())
                actual_sum = int(self.instance_counts[nid])
                if actual_sum != expected_sum:
                    raise ValueError(
                        f"Internal node {nid} instance_counts ({actual_sum}) != "
                        f"sum of children instance_counts ({expected_sum})."
                    )

        leaf_cls_ids = self.leaf_to_class_id[self.is_leaf]
        if len(np.unique(leaf_cls_ids)) != len(leaf_cls_ids):
            raise ValueError("Duplicate class IDs assigned across leaf nodes.")

    def save(self, path: str | Path) -> Path:
        """Persist the tree and its class index pools atomically to a `.npz` archive."""
        self.validate_tree()
        target_path = Path(path)
        if target_path.suffix != ".npz":
            target_path = target_path.with_suffix(".npz")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_name(f"{target_path.stem}.tmp.npz")

        sorted_cls_ids = np.array(sorted(self.class_to_indices.keys()), dtype=np.int64)
        idx_lengths = np.array([self.class_to_indices[int(c)].size for c in sorted_cls_ids], dtype=np.int64)
        idx_offsets = np.zeros(sorted_cls_ids.size + 1, dtype=np.int64)
        np.cumsum(idx_lengths, out=idx_offsets[1:])
        flat_indices = (
            np.concatenate([self.class_to_indices[int(c)] for c in sorted_cls_ids]).astype(np.int64)
            if sorted_cls_ids.size > 0
            else np.empty(0, dtype=np.int64)
        )

        metadata_json = json.dumps(
            {
                "tree_type": self.tree_type,
                "root_id": self.root_id,
                "node_names": list(self.node_names),
            }
        )

        try:
            np.savez_compressed(
                tmp_path,
                parents=self.parents,
                children=self.children,
                child_counts=self.child_counts,
                is_leaf=self.is_leaf,
                depths=self.depths,
                leaf_to_class_id=self.leaf_to_class_id,
                instance_counts=self.instance_counts,
                visits=self.visits,
                q_values=self.q_values,
                sorted_cls_ids=sorted_cls_ids,
                idx_offsets=idx_offsets,
                flat_indices=flat_indices,
                metadata_json=np.array(metadata_json),
            )
            tmp_path.replace(target_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        return target_path

    @classmethod
    def load(cls, path: str | Path) -> FastHierarchyTree:
        """Load and validate a `FastHierarchyTree` from a `.npz` archive."""
        src_path = Path(path)
        if not src_path.exists():
            raise ValueError(f"Tree archive does not exist at {src_path}.")

        required_keys = {
            "parents",
            "children",
            "child_counts",
            "is_leaf",
            "depths",
            "leaf_to_class_id",
            "instance_counts",
            "visits",
            "q_values",
            "sorted_cls_ids",
            "idx_offsets",
            "flat_indices",
            "metadata_json",
        }
        try:
            with np.load(src_path, allow_pickle=False) as data:
                missing = required_keys.difference(data.files)
                if missing:
                    raise ValueError(f"Corrupted tree archive at {src_path}; missing keys: {sorted(missing)}.")

                meta = json.loads(str(data["metadata_json"].item()))
                sorted_cls_ids = data["sorted_cls_ids"].astype(np.int64)
                idx_offsets = data["idx_offsets"].astype(np.int64)
                flat_indices = data["flat_indices"].astype(np.int64)

                class_to_indices: dict[int, np.ndarray] = {}
                for i, cid in enumerate(sorted_cls_ids):
                    start = int(idx_offsets[i])
                    end = int(idx_offsets[i + 1])
                    class_to_indices[int(cid)] = flat_indices[start:end].copy()

                return cls(
                    tree_type=str(meta["tree_type"]),
                    root_id=int(meta["root_id"]),
                    parents=data["parents"],
                    children=data["children"],
                    child_counts=data["child_counts"],
                    is_leaf=data["is_leaf"],
                    depths=data["depths"],
                    leaf_to_class_id=data["leaf_to_class_id"],
                    instance_counts=data["instance_counts"],
                    visits=data["visits"],
                    q_values=data["q_values"],
                    node_names=tuple(meta["node_names"]),
                    class_to_indices=class_to_indices,
                )
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError(f"Failed to load FastHierarchyTree from {src_path}: {exc}") from exc
