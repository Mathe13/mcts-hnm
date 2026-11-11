"""Minibatch label diversity telemetry (Phase C1)."""

from __future__ import annotations

import numpy as np
import torch


def compute_batch_diversity(
    labels: np.ndarray | torch.Tensor | list[int],
    num_classes: int = 100,
) -> dict[str, float]:
    r"""Calculate batch label diversity metrics: unique class count and Shannon entropy.

    Formulas:
    - Unique classes: $N_{\text{unique}} = |\mathcal{C}_{\text{batch}}|$
    - Batch Shannon entropy: $H_{\text{batch}} = -\sum_{c \in \mathcal{C}_{\text{batch}}} \hat{p}_c \ln(\hat{p}_c)$

    Args:
        labels: 1D or ND array of class integer labels in the batch.
        num_classes: Total number of classes (default: 100).

    Returns:
        Dictionary with keys:
        - `unique_classes`: float count of unique classes present in batch.
        - `batch_entropy`: float Shannon entropy of the batch label distribution (nats).
    """
    if isinstance(labels, torch.Tensor):
        lbls = labels.detach().cpu().numpy()
    else:
        lbls = np.asarray(labels, dtype=np.int64)

    if lbls.size == 0:
        return {"unique_classes": 0.0, "batch_entropy": 0.0}

    lbls_flat = lbls.ravel()
    _, counts = np.unique(lbls_flat, return_counts=True)
    num_unique = float(len(counts))

    if num_unique <= 1:
        return {"unique_classes": num_unique, "batch_entropy": 0.0}

    p = counts.astype(np.float64) / float(lbls_flat.size)
    entropy = -float(np.sum(p * np.log(p)))

    return {"unique_classes": num_unique, "batch_entropy": entropy}
