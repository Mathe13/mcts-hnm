"""Core MCTS-HNM agent: UCB1 selection, Min-Max Q-normalization, and backpropagation."""

from __future__ import annotations

import numpy as np

from src.trees.base_tree import FastHierarchyTree


def _validate_positive_float(value: float, name: str, allow_zero: bool = False) -> float:
    """Validate that `value` is a finite float satisfying positivity constraints."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a numeric float, got {value!r}.")
    val = float(value)
    if not np.isfinite(val):
        raise ValueError(f"{name} must be finite, got {val}.")
    if allow_zero:
        if val < 0.0:
            raise ValueError(f"{name} must be >= 0.0, got {val}.")
    else:
        if val <= 0.0:
            raise ValueError(f"{name} must be > 0.0, got {val}.")
    return val


def normalize_sibling_q_values(q_values: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    r"""Normalize sibling action-value estimates $Q_i$ into $[0, 1]$ via Min-Max scaling.

    Formula:
        $$\bar{Q}_i^{\text{norm}} = \frac{Q_i - Q_{\min}}{Q_{\max} - Q_{\min} + \epsilon}$$

    When all sibling $Q$-values are identical ($Q_{\max} == Q_{\min}$), returns an array
    of `0.0` so selection relies purely on the UCB1 exploration bonus without `NaN`.
    """
    eps_val = _validate_positive_float(eps, "eps", allow_zero=False)
    q_arr = np.asarray(q_values, dtype=np.float64)
    if q_arr.ndim != 1 or q_arr.size == 0:
        raise ValueError(f"q_values must be a non-empty 1D array, got shape {q_arr.shape}.")
    if not np.all(np.isfinite(q_arr)):
        raise ValueError("q_values must contain only finite numbers.")

    q_min = float(np.min(q_arr))
    q_max = float(np.max(q_arr))
    if q_max == q_min:
        return np.zeros_like(q_arr, dtype=np.float64)

    return (q_arr - q_min) / (q_max - q_min + eps_val)


def compute_ucb1_scores(
    q_values: np.ndarray,
    child_visits: np.ndarray,
    parent_visits: int,
    exploration_c: float = 2.0,
    q_norm: bool = True,
    eps: float = 1e-5,
) -> np.ndarray:
    r"""Compute UCB1 scores across sibling child nodes of an internal tree node $s$.

    Formula:
        $$U_i = \tilde{Q}_i + c \sqrt{\frac{\ln \max(N_s, 1)}{n_i + \epsilon}}$$
    where $\tilde{Q}_i = \bar{Q}_i^{\text{norm}}$ when `q_norm=True` and $\tilde{Q}_i = Q_i$
    when `q_norm=False`.
    """
    c_val = _validate_positive_float(exploration_c, "exploration_c", allow_zero=True)
    eps_val = _validate_positive_float(eps, "eps", allow_zero=False)

    if isinstance(parent_visits, bool) or not isinstance(parent_visits, (int, np.integer)):
        raise ValueError(f"parent_visits must be an integer >= 0, got {parent_visits!r}.")
    n_parent = int(parent_visits)
    if n_parent < 0:
        raise ValueError(f"parent_visits must be >= 0, got {n_parent}.")

    q_arr = np.asarray(q_values, dtype=np.float64)
    n_arr = np.asarray(child_visits, dtype=np.float64)
    if q_arr.ndim != 1 or q_arr.size == 0:
        raise ValueError(f"q_values must be a non-empty 1D array, got shape {q_arr.shape}.")
    if n_arr.shape != q_arr.shape:
        raise ValueError(
            f"child_visits shape {n_arr.shape} must match q_values shape {q_arr.shape}."
        )
    if np.any(n_arr < 0.0) or not np.all(np.isfinite(n_arr)):
        raise ValueError("child_visits must contain finite non-negative counts.")
    if not np.all(np.isfinite(q_arr)):
        raise ValueError("q_values must contain only finite values.")

    exploitation = (
        normalize_sibling_q_values(q_arr, eps=eps_val) if bool(q_norm) else q_arr.copy()
    )
    log_parent = np.log(max(n_parent, 1))
    exploration = c_val * np.sqrt(log_parent / (n_arr + eps_val))
    return exploitation + exploration


class MCTSHNMAgent:
    r"""Monte Carlo Tree Search agent for hierarchical hard negative mining (`EPIC-03`).

    Supports both:
    - **Strict GiantHunter mode** (`q_norm=False`): raw $Q_i$ values in UCB1 Eq. (1).
    - **Enhanced MCTS-HNM mode** (`q_norm=True`): Min-Max normalized sibling $\bar{Q}_i^{\text{norm}} \in [0, 1]$.
    """

    def __init__(
        self,
        tree: FastHierarchyTree,
        exploration_c: float = 2.0,
        momentum_alpha: float = 0.3,
        q_norm: bool = True,
        eps: float = 1e-5,
    ) -> None:
        if not isinstance(tree, FastHierarchyTree):
            raise ValueError(f"tree must be a FastHierarchyTree instance, got {type(tree).__name__}.")
        self.tree = tree
        self.exploration_c = _validate_positive_float(
            exploration_c, "exploration_c", allow_zero=True
        )
        self.momentum_alpha = _validate_positive_float(
            momentum_alpha, "momentum_alpha", allow_zero=False
        )
        if self.momentum_alpha > 1.0:
            raise ValueError(f"momentum_alpha must be in (0.0, 1.0], got {self.momentum_alpha}.")
        if not isinstance(q_norm, bool):
            raise ValueError(f"q_norm must be a bool, got {q_norm!r}.")
        self.q_norm = q_norm
        self.eps = _validate_positive_float(eps, "eps", allow_zero=False)

    def select_child(self, node_id: int) -> tuple[int, np.ndarray]:
        """Select a child of internal node `node_id` using UCB1 with unvisited-child priority.

        Returns:
            Tuple `(selected_child_node_id, ucb_scores)` where `ucb_scores` has shape `[B]`.
        """
        if isinstance(node_id, bool) or not isinstance(node_id, (int, np.integer)):
            raise ValueError(f"node_id must be an int, got {node_id!r}.")
        nid = int(node_id)
        if nid < 0 or nid >= self.tree.num_nodes:
            raise ValueError(f"node_id {nid} out of range [0, {self.tree.num_nodes - 1}].")
        if bool(self.tree.is_leaf[nid]):
            raise ValueError(f"Cannot call select_child on leaf node {nid}.")

        children = self.tree.get_children(nid)
        q_children = self.tree.q_values[children]
        n_children = self.tree.visits[children]
        parent_visits = int(self.tree.visits[nid])

        ucb_scores = compute_ucb1_scores(
            q_values=q_children,
            child_visits=n_children,
            parent_visits=parent_visits,
            exploration_c=self.exploration_c,
            q_norm=self.q_norm,
            eps=self.eps,
        )

        unvisited_positions = np.flatnonzero(n_children == 0)
        if unvisited_positions.size > 0:
            chosen_pos = int(unvisited_positions[0])
        else:
            chosen_pos = int(np.argmax(ucb_scores))

        return int(children[chosen_pos]), ucb_scores

    def select_leaf(self) -> tuple[int, list[int]]:
        """Traverse the tree from `root_id` to a leaf node using `select_child()`.

        Returns:
            Tuple `(leaf_node_id, path_node_ids)` where `path_node_ids` starts with
            `tree.root_id` and ends with `leaf_node_id`.
        """
        curr = int(self.tree.root_id)
        path: list[int] = [curr]
        while not bool(self.tree.is_leaf[curr]):
            curr, _ = self.select_child(curr)
            path.append(curr)
        return curr, path

    def reset(self) -> None:
        """Reset all MCTS visit counts (`0`) and action-values (`0.0`) on the underlying tree."""
        self.tree.reset_mcts_state()
