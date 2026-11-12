"""Sampling Exposure and Class Coverage Diagnostics (Phase B1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from src.training.batch_diversity import compute_batch_diversity


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
    batch_entropies: list[float] = field(default_factory=list)
    batch_unique_classes: list[float] = field(default_factory=list)

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

        div = compute_batch_diversity(c_arr, num_classes=self.num_classes)
        self.batch_entropies.append(div["batch_entropy"])
        self.batch_unique_classes.append(div["unique_classes"])

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

        cumulative_coverage = (
            float(np.sum(self.class_exposure_counts > 0)) / float(max(1, self.num_classes))
        )
        mean_batch_entropy = (
            float(np.mean(self.batch_entropies)) if self.batch_entropies else 0.0
        )
        mean_unique_classes = (
            float(np.mean(self.batch_unique_classes)) if self.batch_unique_classes else 0.0
        )

        return {
            "total_node_visits": self.total_node_visits,
            "total_leaf_selections": self.total_leaf_selections,
            "total_instances_selected": self.total_instances_selected,
            "unique_instances_selected": len(self.unique_instances_selected),
            "class_exposure_counts": self.class_exposure_counts.tolist(),
            "unexposed_classes_count": unexposed_count,
            "normalized_exposure_entropy": norm_entropy,
            "worst10_class_recall": worst10_recall,
            "mean_batch_entropy": mean_batch_entropy,
            "mean_unique_classes_per_batch": mean_unique_classes,
            "cumulative_class_coverage": cumulative_coverage,
        }


def compute_exposure_recall_correlation(
    exposure_counts: np.ndarray | list[int] | list[float],
    class_recalls: np.ndarray | list[float],
) -> dict[str, float]:
    r"""Compute Spearman rank and Pearson linear correlation between per-class exposures and recalls.

    Args:
        exposure_counts: Array or list of exposure counts per class ($K$ classes).
        class_recalls: Array or list of test recall scores per class ($K$ classes).

    Returns:
        Dictionary with keys:
        - `spearman_rho`: Spearman rank correlation coefficient in [-1.0, 1.0]
        - `spearman_pvalue`: Two-sided p-value for Spearman correlation
        - `pearson_r`: Pearson correlation coefficient in [-1.0, 1.0]
        - `pearson_pvalue`: Two-sided p-value for Pearson correlation
    """
    exp = np.asarray(exposure_counts, dtype=np.float64)
    rec = np.asarray(class_recalls, dtype=np.float64)

    if exp.size < 2 or rec.size < 2 or exp.size != rec.size:
        return {
            "spearman_rho": 0.0,
            "spearman_pvalue": 1.0,
            "pearson_r": 0.0,
            "pearson_pvalue": 1.0,
        }

    if np.all(exp == exp[0]) or np.all(rec == rec[0]):
        return {
            "spearman_rho": 0.0,
            "spearman_pvalue": 1.0,
            "pearson_r": 0.0,
            "pearson_pvalue": 1.0,
        }

    try:
        from scipy import stats

        s_res = stats.spearmanr(exp, rec)
        p_res = stats.pearsonr(exp, rec)
        s_rho = float(s_res.statistic) if not np.isnan(s_res.statistic) else 0.0
        s_pval = float(s_res.pvalue) if not np.isnan(s_res.pvalue) else 1.0
        p_r = float(p_res.statistic) if not np.isnan(p_res.statistic) else 0.0
        p_pval = float(p_res.pvalue) if not np.isnan(p_res.pvalue) else 1.0
        return {
            "spearman_rho": s_rho,
            "spearman_pvalue": s_pval,
            "pearson_r": p_r,
            "pearson_pvalue": p_pval,
        }
    except Exception:
        return {
            "spearman_rho": 0.0,
            "spearman_pvalue": 1.0,
            "pearson_r": 0.0,
            "pearson_pvalue": 1.0,
        }

