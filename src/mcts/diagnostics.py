"""RL visitation entropy and superclass distribution diagnostics for MCTS-HNM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.cifar100_taxonomy import CIFAR100Taxonomy, get_cifar100_taxonomy
from src.trees.base_tree import FastHierarchyTree


def _validate_tree_arg(tree: FastHierarchyTree) -> None:
    if not isinstance(tree, FastHierarchyTree):
        raise ValueError(f"tree must be a FastHierarchyTree instance, got {type(tree).__name__}.")


def compute_visitation_entropy(tree: FastHierarchyTree) -> tuple[float, float]:
    r"""Compute Shannon visitation entropy $H_t$ and normalized entropy $H_t / \ln K$ over tree leaves.

    Formula:
        $$H_t = -\sum_{i \in \text{Leaves}, \, n_i > 0} p_i \ln p_i, \quad p_i = \frac{n_i}{\sum_{j \in \text{Leaves}} n_j}$$

    Returns:
        Tuple `(entropy_nats, normalized_entropy)`:
        - `entropy_nats`: $H_t \in [0.0, \ln K]$
        - `normalized_entropy`: $H_t / \ln K \in [0.0, 1.0]$
        Returns `(0.0, 0.0)` when total leaf visits equal `0` or $K \le 1$.
    """
    _validate_tree_arg(tree)
    leaf_visits = tree.visits[tree.is_leaf].astype(np.float64)
    total_visits = float(np.sum(leaf_visits))
    num_leaves = int(leaf_visits.size)

    if total_visits <= 0.0 or num_leaves <= 1:
        return 0.0, 0.0

    pos_mask = leaf_visits > 0.0
    probs = leaf_visits[pos_mask] / total_visits
    entropy_nats = float(-np.sum(probs * np.log(probs)))
    if entropy_nats < 0.0 and abs(entropy_nats) < 1e-15:
        entropy_nats = 0.0

    max_entropy = float(np.log(num_leaves))
    normalized_entropy = float(np.clip(entropy_nats / max_entropy, 0.0, 1.0))
    return entropy_nats, normalized_entropy


def compute_leaf_visit_distribution(tree: FastHierarchyTree) -> dict[int, float]:
    """Return mapping from each active fine class ID to its empirical leaf visit proportion."""
    _validate_tree_arg(tree)
    leaf_ids = tree.leaf_node_ids()
    leaf_visits = tree.visits[leaf_ids].astype(np.float64)
    total_visits = float(np.sum(leaf_visits))

    pairs: list[tuple[int, float]] = []
    for idx, nid in enumerate(leaf_ids):
        cls_id = int(tree.leaf_to_class_id[int(nid)])
        prop = float(leaf_visits[idx] / total_visits) if total_visits > 0.0 else 0.0
        pairs.append((cls_id, prop))

    return dict(sorted(pairs, key=lambda item: item[0]))


def compute_superclass_visit_distribution(
    tree: FastHierarchyTree,
    taxonomy: CIFAR100Taxonomy | None = None,
) -> dict[str, float]:
    """Aggregate leaf visit proportions across all 20 CIFAR-100 coarse superclasses.

    Works identically on both Semantic Trees and Ward Confusion Trees by mapping
    each leaf's `leaf_to_class_id` to its canonical CIFAR-100 superclass.
    """
    _validate_tree_arg(tree)
    tax = get_cifar100_taxonomy() if taxonomy is None else taxonomy

    superclass_counts: dict[str, float] = {name: 0.0 for name in tax.coarse_names}
    leaf_ids = tree.leaf_node_ids()
    for raw_nid in leaf_ids:
        nid = int(raw_nid)
        cls_id = int(tree.leaf_to_class_id[nid])
        base_cls_id = int(cls_id % 1000)
        coarse_id = int(tax.fine_to_coarse_idx[base_cls_id])
        coarse_name = tax.coarse_names[coarse_id]
        superclass_counts[coarse_name] += float(tree.visits[nid])

    total_visits = float(sum(superclass_counts.values()))
    if total_visits <= 0.0:
        return {name: 0.0 for name in tax.coarse_names}

    return {name: float(cnt / total_visits) for name, cnt in superclass_counts.items()}


@dataclass(frozen=True)
class MCTSStepDiagnostic:
    """Telemetry snapshot recorded at a single MCTS step."""

    step: int
    leaf_node_id: int
    leaf_class_id: int
    reward: float
    leaf_q_value: float
    root_q_value: float
    entropy_nats: float
    normalized_entropy: float


class MCTSDiagnosticsTracker:
    """Stateful tracker for MCTS visitation entropy, Q-values, and superclass visit distributions."""

    def __init__(
        self,
        tree: FastHierarchyTree,
        taxonomy: CIFAR100Taxonomy | None = None,
    ) -> None:
        _validate_tree_arg(tree)
        self.tree = tree
        self.taxonomy = get_cifar100_taxonomy() if taxonomy is None else taxonomy
        self.history: list[MCTSStepDiagnostic] = []

    def record_step(self, step: int, leaf_node_id: int, reward: float) -> MCTSStepDiagnostic:
        """Record diagnostics after `agent.update(leaf_node_id, reward)` at training `step`."""
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or int(step) < 0:
            raise ValueError(f"step must be a non-negative integer, got {step!r}.")
        if isinstance(leaf_node_id, bool) or not isinstance(leaf_node_id, (int, np.integer)):
            raise ValueError(f"leaf_node_id must be an int, got {leaf_node_id!r}.")
        lid = int(leaf_node_id)
        if lid < 0 or lid >= self.tree.num_nodes or not bool(self.tree.is_leaf[lid]):
            raise ValueError(f"leaf_node_id {lid} must be a valid leaf node ID.")
        if isinstance(reward, bool) or not isinstance(reward, (int, float, np.number)) or not np.isfinite(float(reward)):
            raise ValueError(f"reward must be a finite float, got {reward!r}.")

        entropy_nats, normalized_entropy = compute_visitation_entropy(self.tree)
        entry = MCTSStepDiagnostic(
            step=int(step),
            leaf_node_id=lid,
            leaf_class_id=int(self.tree.leaf_to_class_id[lid]),
            reward=float(reward),
            leaf_q_value=float(self.tree.q_values[lid]),
            root_q_value=float(self.tree.q_values[self.tree.root_id]),
            entropy_nats=entropy_nats,
            normalized_entropy=normalized_entropy,
        )
        self.history.append(entry)
        return entry

    def summary(self) -> dict[str, object]:
        """Export current tree visitation diagnostics and aggregated distributions."""
        entropy_nats, normalized_entropy = compute_visitation_entropy(self.tree)
        leaf_q = self.tree.q_values[self.tree.is_leaf]
        return {
            "num_recorded_steps": len(self.history),
            "total_leaf_visits": int(self.tree.visits[self.tree.is_leaf].sum()),
            "entropy_nats": entropy_nats,
            "normalized_entropy": normalized_entropy,
            "root_q_value": float(self.tree.q_values[self.tree.root_id]),
            "max_leaf_q_value": float(np.max(leaf_q)) if leaf_q.size > 0 else 0.0,
            "mean_leaf_q_value": float(np.mean(leaf_q)) if leaf_q.size > 0 else 0.0,
            "leaf_visit_distribution": compute_leaf_visit_distribution(self.tree),
            "superclass_visit_distribution": compute_superclass_visit_distribution(
                self.tree, taxonomy=self.taxonomy
            ),
        }
