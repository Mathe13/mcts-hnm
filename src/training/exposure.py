"""Sampling Exposure and Class Coverage Diagnostics (Phase B1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class ExposureTracker:
    r"""Tracks class and instance exposure during training minibatches.

    Maintains counters for:
    - Node visits ($N_v$)
    - Leaf selections ($N_l$)
    - Total instances selected ($N_s$)
    - Unique instances selected ($N_u$)
    - Class exposure distribution and normalized exposure entropy:
      $$H_{\text{norm}} = -\frac{\sum_{c=1}^K p_c \ln(p_c)}{\ln(K)} \in [0.0, 1.0]$$
    - Worst-10 class recall on validation data.
    """

    num_classes: int = 100
    total_node_visits: int = 0
    total_leaf_selections: int = 0
    total_instances_selected: int = 0
    unique_instances_selected: set[int] = field(default_factory=set)
    class_exposure_counts: np.ndarray = field(default=None)  # type: ignore[assignment]
    leaf_selection_counts: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.class_exposure_counts is None:
            self.class_exposure_counts = np.zeros(self.num_classes, dtype=np.int64)

    def record_batch(
        self,
        class_ids: np.ndarray | torch.Tensor | list[int],
        instance_indices: np.ndarray | torch.Tensor | list[int] | None = None,
    ) -> None:
        """Record sample instances and class IDs present in a training batch."""
        if isinstance(class_ids, torch.Tensor):
            c_arr = class_ids.detach().cpu().numpy()
        else:
            c_arr = np.asarray(class_ids, dtype=np.int64)

        if c_arr.size == 0:
            return

        self.total_instances_selected += int(c_arr.size)

        for c in c_arr.flat:
            cls_idx = int(c)
            if 0 <= cls_idx < self.num_classes:
                self.class_exposure_counts[cls_idx] += 1

        if instance_indices is not None:
            if isinstance(instance_indices, torch.Tensor):
                idx_arr = instance_indices.detach().cpu().numpy()
            else:
                idx_arr = np.asarray(instance_indices, dtype=np.int64)
            for idx in idx_arr.flat:
                self.unique_instances_selected.add(int(idx))

    def record_leaf_selection(self, leaf_id: int) -> None:
        """Record the selection of a hierarchy leaf node by MCTS."""
        self.total_leaf_selections += 1
        self.leaf_selection_counts[leaf_id] = self.leaf_selection_counts.get(leaf_id, 0) + 1

    def record_node_visit(self, node_id: int) -> None:
        """Record a traversal visit to an internal or leaf node."""
        self.total_node_visits += 1

    def compute_diagnostics(
        self,
        val_class_recalls: dict[int, float] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compute summary exposure metrics, entropy, and Worst-10 recall."""
        total_counts = int(np.sum(self.class_exposure_counts))
        unexposed_count = int(np.sum(self.class_exposure_counts == 0))

        if total_counts <= 0 or self.num_classes <= 1:
            norm_entropy = 0.0
        else:
            p = self.class_exposure_counts.astype(np.float64) / float(total_counts)
            p_pos = p[p > 0.0]
            raw_entropy = -float(np.sum(p_pos * np.log(p_pos)))
            max_entropy = float(np.log(self.num_classes))
            norm_entropy = float(np.clip(raw_entropy / max_entropy, 0.0, 1.0))

        worst10_recall: float | None = None
        if val_class_recalls is not None:
            if isinstance(val_class_recalls, dict):
                recalls = [float(v) for v in val_class_recalls.values()]
            else:
                recalls = [float(v) for v in np.asarray(val_class_recalls, dtype=np.float64).flat]
            if len(recalls) > 0:
                recalls.sort()
                worst_k = recalls[: min(10, len(recalls))]
                worst10_recall = float(np.mean(worst_k))

        return {
            "total_node_visits": self.total_node_visits,
            "total_leaf_selections": self.total_leaf_selections,
            "total_instances_selected": self.total_instances_selected,
            "unique_instances_selected": len(self.unique_instances_selected),
            "class_exposure_counts": self.class_exposure_counts.tolist(),
            "unexposed_classes_count": unexposed_count,
            "normalized_exposure_entropy": norm_entropy,
            "worst10_class_recall": worst10_recall,
        }
