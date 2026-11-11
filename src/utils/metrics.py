"""Binary Classification Decision Threshold Calibration and Metric Utilities (Phase B3)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_curve


def calibrate_binary_threshold(
    val_probs: np.ndarray | list[float],
    val_targets: np.ndarray | list[int],
    num_thresholds: int = 100,
) -> float:
    r"""Calibrate optimal decision threshold $\tau_{\text{val}} \in [0.01, 0.99]$ on validation data.

    Finds:
        $$\tau_{\text{val}} = \arg\max_{\tau \in [0.01, 0.99]} \text{F1}_{\text{val}}(\tau)$$

    Args:
        val_probs: 1D array of predicted probabilities in $[0, 1]$.
        val_targets: 1D array of ground-truth binary targets in $\{0, 1\}$.
        num_thresholds: Number of threshold evaluation points (default: 100).

    Returns:
        Calibrated float threshold in $[0.01, 0.99]$ maximizing validation F1 (defaults to 0.5 if all F1 are 0).
    """
    probs = np.asarray(val_probs, dtype=np.float64).ravel()
    targets = np.asarray(val_targets, dtype=np.int64).ravel()

    if probs.size != targets.size:
        raise ValueError(f"Size mismatch: probs has {probs.size} while targets has {targets.size}.")
    if probs.size == 0:
        return 0.5

    thresholds = np.linspace(0.01, 0.99, max(num_thresholds, 10))
    best_tau = 0.5
    best_f1 = -1.0

    for tau in thresholds:
        preds = (probs >= tau).astype(np.int64)
        score = float(f1_score(targets, preds, zero_division=0.0))
        if score > best_f1:
            best_f1 = score
            best_tau = float(tau)

    return float(best_tau)


def evaluate_binary_metrics_at_threshold(
    probs: np.ndarray | list[float],
    targets: np.ndarray | list[int],
    threshold: float = 0.5,
) -> dict[str, float]:
    r"""Compute precision, recall, F1, and AUPRC for binary predictions at a given threshold $\tau$."""
    p_arr = np.asarray(probs, dtype=np.float64).ravel()
    y_arr = np.asarray(targets, dtype=np.int64).ravel()

    if p_arr.size != y_arr.size:
        raise ValueError(f"Size mismatch: probs has {p_arr.size} while targets has {y_arr.size}.")

    preds = (p_arr >= threshold).astype(np.int64)
    f1 = float(f1_score(y_arr, preds, zero_division=0.0))
    prec = float(precision_score(y_arr, preds, zero_division=0.0))
    rec = float(recall_score(y_arr, preds, zero_division=0.0))

    if np.all(y_arr == 0) or np.all(y_arr == 1) or p_arr.size == 0:
        auprc = 0.0
    else:
        auprc = float(average_precision_score(y_arr, p_arr))

    return {
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "auprc": auprc,
        "threshold": float(threshold),
    }


def evaluate_scenario1_with_calibration(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    test_probs: np.ndarray,
    test_targets: np.ndarray,
    num_thresholds: int = 100,
) -> dict[str, float]:
    r"""Calibrate threshold on validation data and evaluate test predictions with default and calibrated $\tau$."""
    tau_val = calibrate_binary_threshold(val_probs, val_targets, num_thresholds=num_thresholds)

    default_metrics = evaluate_binary_metrics_at_threshold(test_probs, test_targets, threshold=0.5)
    calibrated_metrics = evaluate_binary_metrics_at_threshold(test_probs, test_targets, threshold=tau_val)

    return {
        "default_f1": default_metrics["f1"],
        "default_precision": default_metrics["precision"],
        "default_recall": default_metrics["recall"],
        "calibrated_f1": calibrated_metrics["f1"],
        "calibrated_precision": calibrated_metrics["precision"],
        "calibrated_recall": calibrated_metrics["recall"],
        "calibrated_threshold": float(tau_val),
        "auprc": default_metrics["auprc"],
    }
