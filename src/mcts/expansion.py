"""Dynamic leaf expansion for MCTS-HNM by model prediction entropy."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.trees.base_tree import FastHierarchyTree

SUBCLUSTER_ID_STRIDE: int = 1000


def resolve_base_class_id(class_id: int) -> int:
    """Resolve a (possibly virtual sub-cluster) class ID back to its base class ID."""
    if class_id < 0:
        raise ValueError(f"class_id must be non-negative, got {class_id}.")
    return int(class_id) % SUBCLUSTER_ID_STRIDE


def compute_prediction_entropy(
    logits: torch.Tensor | np.ndarray,
    is_binary: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute Shannon prediction entropy H(p) for model logits.

    Args:
        logits: Model logits.
            If is_binary=True: 1D array/tensor [N] or 2D [N, 1].
            If is_binary=False: 2D array/tensor [N, C] where C >= 2.
        is_binary: Whether classification is binary or multi-class.
        eps: Small constant to avoid log(0).

    Returns:
        1D float64 numpy array of shape (N,) containing non-negative Shannon entropies.
    """
    if not isinstance(is_binary, bool):
        raise ValueError(f"is_binary must be a bool, got {is_binary!r}.")
    if eps <= 0.0 or not math.isfinite(eps):
        raise ValueError(f"eps must be a positive finite float, got {eps!r}.")

    if isinstance(logits, np.ndarray):
        logits_t = torch.as_tensor(logits, dtype=torch.float64)
    elif isinstance(logits, torch.Tensor):
        logits_t = logits.detach().to(dtype=torch.float64)
    else:
        raise ValueError(f"logits must be a torch.Tensor or np.ndarray, got {type(logits).__name__}.")

    if logits_t.numel() == 0:
        raise ValueError("logits must be non-empty.")
    if not bool(torch.isfinite(logits_t).all()):
        raise ValueError("logits must contain only finite values (no NaN or Inf).")

    if is_binary:
        if logits_t.ndim == 2 and logits_t.shape[1] == 1:
            logits_1d = logits_t.squeeze(1)
        elif logits_t.ndim == 1:
            logits_1d = logits_t
        else:
            raise ValueError(
                f"Binary logits must have shape (N,) or (N, 1), got {tuple(logits_t.shape)}."
            )
        p = torch.sigmoid(logits_1d)
        q = 1.0 - p
        # Binary Shannon entropy in nats: - (p * ln(p) + (1-p) * ln(1-p))
        entropy = -(p * torch.log(p.clamp_min(eps)) + q * torch.log(q.clamp_min(eps)))
        entropy_np = entropy.cpu().numpy().astype(np.float64)
    else:
        if logits_t.ndim != 2 or logits_t.shape[1] < 2:
            raise ValueError(
                f"Multi-class logits must have shape (N, C) with C >= 2, got {tuple(logits_t.shape)}."
            )
        probs = F.softmax(logits_t, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs.clamp_min(eps)), dim=-1)
        entropy_np = entropy.cpu().numpy().astype(np.float64)

    # Ensure non-negative due to float precision
    return np.maximum(entropy_np, 0.0)


def expand_leaf_by_entropy(
    tree: FastHierarchyTree,
    leaf_node_id: int,
    entropies: np.ndarray,
    k_sub: int = 3,
) -> tuple[int, ...]:
    """Dynamically partition a leaf node into k_sub entropy-quantile children in-place.

    Args:
        tree: FastHierarchyTree instance to mutate.
        leaf_node_id: Index of the leaf node to split.
        entropies: 1D float array of shape (instance_counts[leaf_node_id],).
        k_sub: Number of sub-cluster children to produce (k_sub >= 2).

    Returns:
        Tuple of newly created child node IDs.
    """
    if not isinstance(tree, FastHierarchyTree):
        raise ValueError(f"tree must be a FastHierarchyTree, got {type(tree).__name__}.")
    if isinstance(leaf_node_id, bool) or not isinstance(leaf_node_id, (int, np.integer)):
        raise ValueError(f"leaf_node_id must be an integer, got {leaf_node_id!r}.")
    lid = int(leaf_node_id)
    if lid < 0 or lid >= tree.num_nodes:
        raise ValueError(f"leaf_node_id {lid} out of range [0, {tree.num_nodes - 1}].")
    if not bool(tree.is_leaf[lid]):
        raise ValueError(f"Node {lid} is not a leaf node.")

    if isinstance(k_sub, bool) or not isinstance(k_sub, (int, np.integer)):
        raise ValueError(f"k_sub must be an integer >= 2, got {k_sub!r}.")
    k_val = int(k_sub)
    if k_val < 2:
        raise ValueError(f"k_sub must be >= 2, got {k_val}.")

    base_class_id = int(tree.leaf_to_class_id[lid])
    if base_class_id < 0 or base_class_id >= SUBCLUSTER_ID_STRIDE:
        raise ValueError(
            f"Node {lid} has leaf_to_class_id={base_class_id}, which is not an unexpanded base class."
        )

    ent_arr = np.asarray(entropies, dtype=np.float64)
    if ent_arr.ndim != 1:
        raise ValueError(f"entropies must be a 1D array, got shape {ent_arr.shape}.")
    if not np.all(np.isfinite(ent_arr)):
        raise ValueError("entropies contains non-finite values (NaN or Inf).")

    leaf_indices = tree.class_to_indices[base_class_id]
    n_instances = int(leaf_indices.size)
    if ent_arr.size != n_instances:
        raise ValueError(
            f"entropies size ({ent_arr.size}) does not match leaf instance count ({n_instances})."
        )
    if n_instances < k_val:
        raise ValueError(
            f"Cannot split leaf {lid} with {n_instances} instances into k_sub={k_val} sub-clusters."
        )

    # Sort indices deterministically: primary by entropy ascending, secondary by sample index
    order = np.lexsort((leaf_indices, ent_arr))
    sorted_sample_indices = leaf_indices[order]

    sub_buckets = np.array_split(sorted_sample_indices, k_val)
    # Check that all buckets are non-empty
    for b_idx, b in enumerate(sub_buckets):
        if b.size == 0:
            raise ValueError(f"Bucket {b_idx} is empty after split.")

    # Mutate tree in-place
    v_old = tree.num_nodes
    b_max_old = tree.children.shape[1]
    b_max_new = max(b_max_old, k_val)

    new_child_ids = tuple(v_old + i for i in range(k_val))
    v_new = v_old + k_val

    # Resize node arrays
    new_parents = np.full(v_new, -1, dtype=np.int64)
    new_parents[:v_old] = tree.parents
    for cid in new_child_ids:
        new_parents[cid] = lid

    new_children = np.full((v_new, b_max_new), -1, dtype=np.int64)
    new_children[:v_old, :b_max_old] = tree.children
    # Set parent lid's children
    new_children[lid, :k_val] = np.array(new_child_ids, dtype=np.int64)

    new_child_counts = np.zeros(v_new, dtype=np.int64)
    new_child_counts[:v_old] = tree.child_counts
    new_child_counts[lid] = k_val

    new_is_leaf = np.zeros(v_new, dtype=bool)
    new_is_leaf[:v_old] = tree.is_leaf
    new_is_leaf[lid] = False
    for cid in new_child_ids:
        new_is_leaf[cid] = True

    new_depths = np.zeros(v_new, dtype=np.int64)
    new_depths[:v_old] = tree.depths
    for cid in new_child_ids:
        new_depths[cid] = tree.depths[lid] + 1

    new_leaf_to_class_id = np.full(v_new, -1, dtype=np.int64)
    new_leaf_to_class_id[:v_old] = tree.leaf_to_class_id
    new_leaf_to_class_id[lid] = -1

    new_instance_counts = np.zeros(v_new, dtype=np.int64)
    new_instance_counts[:v_old] = tree.instance_counts
    # lid instance count remains the sum of instances

    new_visits = np.zeros(v_new, dtype=np.int64)
    new_visits[:v_old] = tree.visits

    new_q_values = np.zeros(v_new, dtype=np.float64)
    new_q_values[:v_old] = tree.q_values

    new_node_names = list(tree.node_names)
    old_leaf_name = tree.node_names[lid]
    new_node_names[lid] = f"split_{old_leaf_name}"

    # Remove old class_to_indices entry and populate children
    del tree.class_to_indices[base_class_id]

    for i, cid in enumerate(new_child_ids):
        virtual_cls_id = (i + 1) * SUBCLUSTER_ID_STRIDE + base_class_id
        new_leaf_to_class_id[cid] = virtual_cls_id
        inst_count = int(sub_buckets[i].size)
        new_instance_counts[cid] = inst_count
        new_node_names.append(f"{old_leaf_name}:sub_{i}")
        tree.class_to_indices[virtual_cls_id] = np.ascontiguousarray(sub_buckets[i], dtype=np.int64)

    # Assign new arrays to tree
    tree.parents = new_parents
    tree.children = new_children
    tree.child_counts = new_child_counts
    tree.is_leaf = new_is_leaf
    tree.depths = new_depths
    tree.leaf_to_class_id = new_leaf_to_class_id
    tree.instance_counts = new_instance_counts
    tree.visits = new_visits
    tree.q_values = new_q_values
    tree.node_names = tuple(new_node_names)

    # Validate structural consistency
    tree.validate_tree()
    return new_child_ids


def maybe_expand_leaf(
    tree: FastHierarchyTree,
    leaf_node_id: int,
    model: nn.Module,
    train_inputs: torch.Tensor,
    is_binary: bool = True,
    split_threshold: int = 50,
    k_sub: int = 3,
    batch_size: int = 256,
    device: torch.device | None = None,
) -> tuple[int, ...] | None:
    """Evaluate whether leaf_node_id should be expanded and expand it if eligible.

    Conditions for expansion:
    1. tree.is_leaf[leaf_node_id] is True
    2. 0 <= tree.leaf_to_class_id[leaf_node_id] < SUBCLUSTER_ID_STRIDE
    3. tree.visits[leaf_node_id] >= split_threshold
    4. tree.instance_counts[leaf_node_id] >= k_sub

    Returns:
        Tuple of new child node IDs if expanded, or None if conditions not met.
    """
    if isinstance(leaf_node_id, bool) or not isinstance(leaf_node_id, (int, np.integer)):
        raise ValueError(f"leaf_node_id must be an integer, got {leaf_node_id!r}.")
    lid = int(leaf_node_id)
    if lid < 0 or lid >= tree.num_nodes:
        raise ValueError(f"leaf_node_id {lid} out of range [0, {tree.num_nodes - 1}].")

    if not bool(tree.is_leaf[lid]):
        return None

    cls_id = int(tree.leaf_to_class_id[lid])
    if cls_id < 0 or cls_id >= SUBCLUSTER_ID_STRIDE:
        return None

    if split_threshold < 1:
        raise ValueError(f"split_threshold must be >= 1, got {split_threshold}.")
    if k_sub < 2:
        raise ValueError(f"k_sub must be >= 2, got {k_sub}.")

    if int(tree.visits[lid]) < split_threshold:
        return None

    leaf_indices = tree.class_to_indices[cls_id]
    if leaf_indices.size < k_sub:
        return None

    dev = device if device is not None else next(model.parameters()).device
    model_mode = model.training
    model.eval()

    all_logits: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, leaf_indices.size, batch_size):
            batch_idx = leaf_indices[i : i + batch_size]
            batch_x = train_inputs[batch_idx].to(dev)
            logits = model(batch_x)
            all_logits.append(logits.detach().cpu())

    if model_mode:
        model.train()

    logits_t = torch.cat(all_logits, dim=0)
    entropies = compute_prediction_entropy(logits_t, is_binary=is_binary)

    return expand_leaf_by_entropy(
        tree=tree,
        leaf_node_id=lid,
        entropies=entropies,
        k_sub=k_sub,
    )
