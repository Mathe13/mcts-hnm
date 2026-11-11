"""Mode B: Data-Driven Ward Confusion Hierarchy builder from validation confusion matrices."""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import torch

from src.data.cifar100_taxonomy import CIFAR100Taxonomy, get_cifar100_taxonomy
from src.trees.base_tree import FastHierarchyTree
from src.trees.semantic_tree import _validate_class_to_indices
from src.utils.seed import set_global_seed


def validate_confusion_matrix(confusion_matrix: np.ndarray) -> np.ndarray:
    r"""Validate a confusion matrix $C \in \mathbb{R}^{K \times K}$ before distance computation.

    Args:
        confusion_matrix: 2D square matrix of non-negative confusion counts or probabilities.

    Returns:
        Validated 2D `float64` NumPy array of shape `(K, K)`.

    Raises:
        ValueError: If `confusion_matrix` is not a 2D square matrix with $K \ge 2$, contains
            `NaN`, `Inf`, or negative values, or has any zero-sum row.
    """
    arr = np.asarray(confusion_matrix)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 2:
        raise ValueError(
            f"confusion_matrix must be a 2D square matrix with K >= 2, got shape {arr.shape}."
        )
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"confusion_matrix must be numeric, got dtype {arr.dtype}.")

    mat = arr.astype(np.float64, copy=True)
    if not np.all(np.isfinite(mat)):
        raise ValueError("confusion_matrix contains NaN or Inf entries; all entries must be finite.")
    if np.any(mat < 0.0):
        raise ValueError("confusion_matrix contains negative entries; all entries must be >= 0.")

    row_sums = mat.sum(axis=1)
    zero_rows = np.flatnonzero(row_sums <= 0.0)
    if zero_rows.size > 0:
        raise ValueError(
            f"confusion_matrix contains zero-sum row(s) at indices {zero_rows.tolist()}."
        )

    return mat


def is_euclidean_dissimilarity(dist_matrix: np.ndarray, tol: float = 1e-10) -> bool:
    r"""Check if a symmetric dissimilarity matrix can be embedded in Euclidean space via Schoenberg's theorem.

    A dissimilarity matrix $D \in \mathbb{R}^{n \times n}$ is embeddable in Euclidean space iff the
    centering Gram matrix $B = -\frac{1}{2} H (D \odot D) H$ is positive semi-definite (PSD),
    where $H = I - \frac{1}{n} \mathbf{1}\mathbf{1}^T$.
    """
    arr = np.asarray(dist_matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return False
    n = arr.shape[0]
    if n < 2:
        return True
    d_sq = arr ** 2
    h = np.eye(n, dtype=np.float64) - (1.0 / n) * np.ones((n, n), dtype=np.float64)
    b = -0.5 * (h @ d_sq @ h)
    b = 0.5 * (b + b.T)  # enforce exact numerical symmetry
    eigvals = np.linalg.eigvalsh(b)
    return bool(np.all(eigvals >= -tol))


def compute_confusion_distance_matrix(confusion_matrix: np.ndarray) -> np.ndarray:
    r"""Compute the symmetric normalized confusion distance matrix $D \in [0, 1]^{K \times K}$.

    Uses off-diagonal misclassifications $C^{\text{off}}$ ($C^{\text{off}}_{i,i} = 0$):
    $$D_{i,j} = 1 - \frac{C^{\text{off}}_{i,j} + C^{\text{off}}_{j,i}}{2 \cdot \max(C^{\text{off}})}$$
    for $i \neq j$ and $D_{i,i} = 0.0$. If $\max(C^{\text{off}}) == 0$, all off-diagonal
    distances are set to `1.0`.
    """
    mat = validate_confusion_matrix(confusion_matrix)
    k = mat.shape[0]

    c_off = mat.copy()
    np.fill_diagonal(c_off, 0.0)
    max_off = float(c_off.max())

    if max_off <= 0.0:
        dist = np.ones((k, k), dtype=np.float64)
        np.fill_diagonal(dist, 0.0)
        return dist

    sym_sim = (c_off + c_off.T) / (2.0 * max_off)
    dist = np.clip(1.0 - sym_sim, 0.0, 1.0)
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    return np.ascontiguousarray(dist, dtype=np.float64)


@overload
def build_confusion_tree(
    confusion_matrix: np.ndarray,
    class_to_indices: dict[int, np.ndarray],
    return_linkage: Literal[False] = False,
    taxonomy: CIFAR100Taxonomy | None = None,
    linkage_method: str = "ward",
) -> FastHierarchyTree: ...


@overload
def build_confusion_tree(
    confusion_matrix: np.ndarray,
    class_to_indices: dict[int, np.ndarray],
    return_linkage: Literal[True],
    taxonomy: CIFAR100Taxonomy | None = None,
    linkage_method: str = "ward",
) -> tuple[FastHierarchyTree, np.ndarray, np.ndarray]: ...


def build_confusion_tree(
    confusion_matrix: np.ndarray,
    class_to_indices: dict[int, np.ndarray],
    return_linkage: bool = False,
    taxonomy: CIFAR100Taxonomy | None = None,
    linkage_method: str = "ward",
) -> FastHierarchyTree | tuple[FastHierarchyTree, np.ndarray, np.ndarray]:
    """Construct a binary Confusion Hierarchy from `confusion_matrix` and `class_to_indices`.

    Args:
        confusion_matrix: Square confusion matrix of shape `(K, K)` or `(100, 100)`.
        class_to_indices: Mapping from active fine class IDs (`0..99`) to 1D sample index arrays.
        return_linkage: If `True`, returns `(tree, linkage_matrix, distance_matrix)`.
        taxonomy: Optional `CIFAR100Taxonomy` for human-readable class names.
        linkage_method: Hierarchical clustering linkage algorithm (`"ward"` or `"average"`).

    Returns:
        Binary `FastHierarchyTree` with `2K - 1` nodes and `root_id=0` (or tuple if `return_linkage=True`).
    """
    method_lower = str(linkage_method).lower()
    if method_lower not in ("ward", "average", "ward_legacy"):
        raise ValueError(
            f"Unsupported linkage_method: '{linkage_method}'. Must be 'ward', 'average', or 'ward_legacy'."
        )
    cleaned_pools = _validate_class_to_indices(class_to_indices)
    sorted_class_ids = sorted(cleaned_pools.keys())
    k = len(sorted_class_ids)
    if k < 2:
        raise ValueError(f"build_confusion_tree requires at least 2 active classes, got {k}.")

    mat = validate_confusion_matrix(confusion_matrix)
    if mat.shape[0] == k:
        active_mat = mat
    elif mat.shape[0] == 100 and k < 100:
        idx_arr = np.array(sorted_class_ids, dtype=np.int64)
        active_mat = validate_confusion_matrix(mat[np.ix_(idx_arr, idx_arr)])
    else:
        raise ValueError(
            f"confusion_matrix shape {mat.shape} does not match active class count ({k}) or full CIFAR-100 (100)."
        )

    dist_matrix = compute_confusion_distance_matrix(active_mat)
    condensed_dist = squareform(dist_matrix, checks=False)

    if method_lower == "ward":
        # SciPy's Ward variance minimization requires Euclidean metric vectors or distances.
        # Operating directly on row-normalized confusion vectors p_i in R^K (p_i = C_i / sum(C_i))
        # guarantees a mathematically sound Euclidean space embedding, preventing non-Euclidean inversions.
        row_sums = active_mat.sum(axis=1, keepdims=True)
        p_vectors = active_mat / np.clip(row_sums, 1e-12, None)
        linkage_matrix = np.ascontiguousarray(
            linkage(p_vectors, method="ward", metric="euclidean"), dtype=np.float64
        )
    elif method_lower == "ward_legacy":
        # Legacy heuristic: sqrt(2 * D_ij)
        euclidean_condensed = np.sqrt(2.0 * np.clip(condensed_dist, 0.0, None))
        linkage_matrix = np.ascontiguousarray(
            linkage(euclidean_condensed, method="ward"), dtype=np.float64
        )
    elif method_lower == "average":
        # Average linkage (UPGMA) operates directly and consistently on general dissimilarities
        linkage_matrix = np.ascontiguousarray(
            linkage(condensed_dist, method="average"), dtype=np.float64
        )
    else:
        raise ValueError(f"Unsupported linkage_method: '{linkage_method}'.")

    tax = taxonomy if taxonomy is not None else get_cifar100_taxonomy()
    total_nodes = 2 * k - 1
    num_internal = k - 1

    parents = np.full(total_nodes, -1, dtype=np.int64)
    children = np.full((total_nodes, 2), -1, dtype=np.int64)
    child_counts = np.zeros(total_nodes, dtype=np.int64)
    is_leaf = np.zeros(total_nodes, dtype=bool)
    depths = np.zeros(total_nodes, dtype=np.int64)
    leaf_to_class_id = np.full(total_nodes, -1, dtype=np.int64)
    instance_counts = np.zeros(total_nodes, dtype=np.int64)
    visits = np.zeros(total_nodes, dtype=np.int64)
    q_values = np.zeros(total_nodes, dtype=np.float64)
    node_names: list[str] = [""] * total_nodes

    def _scipy_to_tree_id(scipy_idx: int) -> int:
        if scipy_idx < k:
            return num_internal + scipy_idx
        return (2 * k - 2) - scipy_idx

    # Initialize leaf nodes at indices [k - 1 .. 2k - 2]
    for local_idx, cls_id in enumerate(sorted_class_ids):
        leaf_nid = num_internal + local_idx
        is_leaf[leaf_nid] = True
        leaf_to_class_id[leaf_nid] = cls_id
        instance_counts[leaf_nid] = int(cleaned_pools[cls_id].size)
        base_id = cls_id % 1000
        if base_id < len(tax.fine_names):
            c_name = tax.fine_names[base_id]
        else:
            c_name = str(base_id)
        if cls_id >= 1000:
            node_names[leaf_nid] = f"class:{c_name}:sub_{cls_id // 1000}"
        else:
            node_names[leaf_nid] = f"class:{c_name}"

    # Process merges bottom-up (m = 0 .. k - 2); final merge m = k - 2 maps to root_id = 0
    for merge_idx in range(num_internal):
        parent_nid = (k - 2) - merge_idx
        left_scipy = int(linkage_matrix[merge_idx, 0])
        right_scipy = int(linkage_matrix[merge_idx, 1])
        left_nid = _scipy_to_tree_id(left_scipy)
        right_nid = _scipy_to_tree_id(right_scipy)

        children[parent_nid, 0] = left_nid
        children[parent_nid, 1] = right_nid
        child_counts[parent_nid] = 2
        parents[left_nid] = parent_nid
        parents[right_nid] = parent_nid
        instance_counts[parent_nid] = instance_counts[left_nid] + instance_counts[right_nid]
        node_names[parent_nid] = "root" if parent_nid == 0 else f"{method_lower}_cluster:{merge_idx}"

    # Assign depths top-down from root_id = 0
    for nid in range(num_internal):
        left_nid = int(children[nid, 0])
        right_nid = int(children[nid, 1])
        depths[left_nid] = depths[nid] + 1
        depths[right_nid] = depths[nid] + 1

    tree = FastHierarchyTree(
        tree_type="confusion_ward" if "ward" in method_lower else f"confusion_{method_lower}",
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

    if return_linkage:
        return tree, linkage_matrix, dist_matrix
    return tree


def estimate_validation_confusion_matrix(
    train_embeddings: np.ndarray | torch.Tensor | None = None,
    train_labels: np.ndarray | torch.Tensor | None = None,
    val_embeddings: np.ndarray | torch.Tensor | None = None,
    val_labels: np.ndarray | torch.Tensor | None = None,
    num_classes: int = 100,
    smoothing: float = 1e-3,
    seed: int = 42,
    is_test_data: bool = False,
    val_features: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    r"""Estimate a baseline $K \times K$ validation confusion matrix from Tier-1 feature embeddings.

    Fits a deterministic L2-regularized multinomial logistic classifier on `train_embeddings`
    and accumulates predicted probabilities / hard confusion counts on `val_embeddings` with
    additive Laplace `smoothing` so every class row has a strictly positive sum.

    Raises:
        ValueError: If is_test_data is True, or if validation data is missing.
    """
    if is_test_data:
        raise ValueError("Estimating confusion matrix on test data is strictly prohibited to prevent data leakage.")

    # Resolve aliases
    if val_embeddings is None and val_features is not None:
        val_embeddings = val_features

    if val_embeddings is None or val_labels is None:
        raise ValueError("val_embeddings and val_labels must be provided to estimate validation confusion matrix.")

    if train_embeddings is None or train_labels is None:
        train_embeddings = val_embeddings
        train_labels = val_labels

    set_global_seed(seed)
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}.")
    if smoothing < 0.0:
        raise ValueError(f"smoothing must be >= 0.0, got {smoothing}.")

    x_train = (
        train_embeddings.detach().cpu().numpy()
        if isinstance(train_embeddings, torch.Tensor)
        else np.asarray(train_embeddings)
    ).astype(np.float64)
    y_train = (
        train_labels.detach().cpu().numpy()
        if isinstance(train_labels, torch.Tensor)
        else np.asarray(train_labels)
    ).astype(np.int64)
    x_val = (
        val_embeddings.detach().cpu().numpy()
        if isinstance(val_embeddings, torch.Tensor)
        else np.asarray(val_embeddings)
    ).astype(np.float64)
    y_val = (
        val_labels.detach().cpu().numpy()
        if isinstance(val_labels, torch.Tensor)
        else np.asarray(val_labels)
    ).astype(np.int64)

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    clf.fit(x_train, y_train)
    val_preds = clf.predict(x_val)

    c_hard = sk_confusion_matrix(
        y_val,
        val_preds,
        labels=np.arange(num_classes, dtype=np.int64),
    ).astype(np.float64)

    # Combine hard confusion counts with soft class-conditional probability expectations
    # so off-diagonal visual confusion structure is smooth and well-conditioned.
    val_probs = clf.predict_proba(x_val)
    c_soft = np.zeros((num_classes, num_classes), dtype=np.float64)
    active_classes = clf.classes_.astype(np.int64)
    for cls_id in range(num_classes):
        mask = y_val == cls_id
        if np.any(mask):
            c_soft[cls_id, active_classes] = val_probs[mask].sum(axis=0)

    c_combined = c_hard + c_soft + float(smoothing)
    return validate_confusion_matrix(c_combined)
