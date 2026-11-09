"""Evaluation metrics and training loops for Scenario 1 and Scenario 2."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def evaluate_scenario1_metrics(
    model: nn.Module,
    features: torch.Tensor,
    binary_labels: np.ndarray | torch.Tensor,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> dict[str, float]:
    r"""Evaluate test metrics for Scenario 1 (Imbalanced Binary One-vs-Rest).

    Returns:
        Dictionary with keys:
        - `loss`: mean BCE loss
        - `f1`: F1-score at default threshold 0.0 (probability 0.5)
        - `precision`: precision at default threshold
        - `recall`: recall at default threshold
        - `auprc`: Area under Precision-Recall Curve
        - `fpr_at_95_recall`: False Positive Rate at 95% True Positive Rate
    """
    if not isinstance(model, nn.Module):
        raise ValueError("model must be an nn.Module instance.")
    if not isinstance(features, torch.Tensor):
        raise ValueError("features must be a torch.Tensor.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}.")

    y_true = (
        binary_labels.detach().cpu().numpy()
        if isinstance(binary_labels, torch.Tensor)
        else np.asarray(binary_labels)
    ).astype(np.int64)

    if y_true.ndim != 1 or y_true.size == 0:
        raise ValueError("binary_labels must be a non-empty 1D integer array.")
    if features.shape[0] != y_true.size:
        raise ValueError(
            f"features length ({features.shape[0]}) does not match labels length ({y_true.size})."
        )

    dev = device if device is not None else next(model.parameters()).device
    model_mode = model.training
    model.eval()

    all_logits: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, features.shape[0], batch_size):
            batch_x = features[i : i + batch_size].to(dev)
            logits = model(batch_x)
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            all_logits.append(logits.detach().cpu())

    if model_mode:
        model.train()

    logits_t = torch.cat(all_logits, dim=0).to(dtype=torch.float64)
    y_true_t = torch.as_tensor(y_true, dtype=torch.float64)
    loss = float(F.binary_cross_entropy_with_logits(logits_t, y_true_t).item())

    probs = torch.sigmoid(logits_t).numpy()
    preds = (probs >= 0.5).astype(np.int64)

    f1 = float(f1_score(y_true, preds, zero_division=0.0))
    precision = float(precision_score(y_true, preds, zero_division=0.0))
    recall = float(recall_score(y_true, preds, zero_division=0.0))

    # AUPRC
    if np.all(y_true == 0) or np.all(y_true == 1):
        auprc = 0.0
    else:
        auprc = float(average_precision_score(y_true, probs))

    # FPR@95
    if np.sum(y_true == 1) == 0:
        fpr_95 = 1.0
    elif np.sum(y_true == 0) == 0:
        fpr_95 = 0.0
    else:
        fpr, tpr, _ = roc_curve(y_true, probs)
        valid_fpr = fpr[tpr >= 0.95]
        fpr_95 = float(np.min(valid_fpr)) if valid_fpr.size > 0 else 1.0

    return {
        "loss": loss,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auprc": auprc,
        "fpr_at_95_recall": fpr_95,
    }


def evaluate_scenario2_metrics(
    model: nn.Module,
    features: torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    num_classes: int = 100,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> dict[str, float]:
    r"""Evaluate test metrics for Scenario 2 (100-way Multi-Class Curriculum).

    Returns:
        Dictionary with keys:
        - `loss`: mean Cross-Entropy loss
        - `top1_accuracy`: Top-1 accuracy in [0, 1]
        - `macro_f1`: Macro-averaged F1 across all classes
        - `worst_10_class_recall`: Average recall across the 10 lowest-recall classes
    """
    if not isinstance(model, nn.Module):
        raise ValueError("model must be an nn.Module instance.")
    if not isinstance(features, torch.Tensor):
        raise ValueError("features must be a torch.Tensor.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}.")
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}.")

    y_true = (
        labels.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor)
        else np.asarray(labels)
    ).astype(np.int64)

    if y_true.ndim != 1 or y_true.size == 0:
        raise ValueError("labels must be a non-empty 1D integer array.")
    if features.shape[0] != y_true.size:
        raise ValueError(
            f"features length ({features.shape[0]}) does not match labels length ({y_true.size})."
        )

    dev = device if device is not None else next(model.parameters()).device
    model_mode = model.training
    model.eval()

    all_logits: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, features.shape[0], batch_size):
            batch_x = features[i : i + batch_size].to(dev)
            logits = model(batch_x)
            all_logits.append(logits.detach().cpu())

    if model_mode:
        model.train()

    logits_t = torch.cat(all_logits, dim=0).to(dtype=torch.float64)
    y_true_t = torch.as_tensor(y_true, dtype=torch.long)
    loss = float(F.cross_entropy(logits_t, y_true_t).item())

    preds = torch.argmax(logits_t, dim=1).numpy()
    top1 = float(np.mean(preds == y_true))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0.0))

    # Per-class recall
    recalls: list[float] = []
    for c in range(num_classes):
        c_mask = y_true == c
        if np.any(c_mask):
            recalls.append(float(np.mean(preds[c_mask] == c)))
        else:
            recalls.append(0.0)

    recalls_arr = np.array(recalls, dtype=np.float64)
    # 10 worst classes
    worst_k = min(10, num_classes)
    worst_10_recall = float(np.mean(np.sort(recalls_arr)[:worst_k]))

    return {
        "loss": loss,
        "top1_accuracy": top1,
        "macro_f1": macro_f1,
        "worst_10_class_recall": worst_10_recall,
    }
