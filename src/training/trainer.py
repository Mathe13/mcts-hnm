"""Evaluation metrics and training loops for Scenario 1 and Scenario 2."""

from __future__ import annotations

import logging
import time
from typing import Literal

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mcts.agent import MCTSHNMAgent, compute_mcts_reward
from src.mcts.diagnostics import MCTSDiagnosticsTracker, compute_superclass_visit_distribution, compute_visitation_entropy
from src.mcts.expansion import maybe_expand_leaf
from src.mcts.sampler import DualStreamBatchSampler, UniformGroupedBatchSampler
from src.models.mlp_head import Tier1MLPHead, create_mlp_optimizer
from src.training.baselines import OHEMSampler, UniformSampler
from src.training.exposure import ExposureTracker
from src.trees.base_tree import FastHierarchyTree
from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)

SUPPORTED_STRATEGIES = (
    "uniform",
    "uniform_grouped",
    "ohem",
    "mcts_strict",
    "mcts_enhanced",
)


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
    worst_k = min(10, num_classes)
    worst_10_recall = float(np.mean(np.sort(recalls_arr)[:worst_k]))

    return {
        "loss": loss,
        "top1_accuracy": top1,
        "macro_f1": macro_f1,
        "worst_10_class_recall": worst_10_recall,
    }


def train_scenario1(
    train_features: torch.Tensor,
    train_binary_labels: np.ndarray | torch.Tensor,
    positive_indices: np.ndarray,
    negative_indices: np.ndarray,
    val_features: torch.Tensor | None = None,
    val_binary_labels: np.ndarray | torch.Tensor | None = None,
    test_features: torch.Tensor | None = None,
    test_binary_labels: np.ndarray | torch.Tensor | None = None,
    strategy: str = "mcts_enhanced",
    tree: FastHierarchyTree | None = None,
    num_steps: int = 1000,
    eval_every: int = 100,
    pos_batch_size: int = 16,
    neg_batch_size: int = 48,
    lr: float = 1e-3,
    exploration_c: float = 2.0,
    momentum_alpha: float = 0.3,
    dual_stream_ratio: float = 0.5,
    enable_leaf_expansion: bool = False,
    split_threshold: int = 50,
    k_sub: int = 3,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict[str, object]:
    """Train downstream MLP classifier under Scenario 1 (Imbalanced Binary One-vs-Rest).

    Periodic evaluations at eval_every steps are computed strictly on validation data (val_features),
    preventing scientific data leakage from the test split.
    """
    if val_features is None or val_binary_labels is None:
        raise ValueError(
            "val_features and val_binary_labels must be provided for validation evaluation during training. "
            "Periodic evaluation on test data is prohibited."
        )
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Supported: {SUPPORTED_STRATEGIES}."
        )
    if num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}.")
    if eval_every <= 0:
        raise ValueError(f"eval_every must be > 0, got {eval_every}.")

    set_global_seed(seed)
    dev = device if device is not None else torch.device("cpu")

    input_dim = int(train_features.shape[1])
    model = Tier1MLPHead(input_dim=input_dim, hidden_dim=128, num_outputs=1, squeeze_binary=True).to(dev)
    optimizer = create_mlp_optimizer(model, lr=lr)

    pos_idx_arr = np.asarray(positive_indices, dtype=np.int64)
    neg_idx_arr = np.asarray(negative_indices, dtype=np.int64)

    agent: MCTSHNMAgent | None = None
    mcts_sampler: DualStreamBatchSampler | None = None
    diagnostics_tracker: MCTSDiagnosticsTracker | None = None
    uni_sampler: UniformSampler | None = None
    ohem_sampler: OHEMSampler | None = None
    uniform_grouped_sampler: UniformGroupedBatchSampler | None = None

    if strategy in ("mcts_strict", "mcts_enhanced"):
        if tree is None:
            raise ValueError(f"tree is required for MCTS strategy '{strategy}'.")
        q_norm = (strategy == "mcts_enhanced")
        beta = dual_stream_ratio if strategy == "mcts_enhanced" else 0.0
        agent = MCTSHNMAgent(
            tree=tree,
            exploration_c=exploration_c,
            momentum_alpha=momentum_alpha,
            q_norm=q_norm,
        )
        mcts_sampler = DualStreamBatchSampler(
            tree=tree,
            neg_batch_size=neg_batch_size,
            dual_stream_ratio=beta,
            positive_indices=pos_idx_arr,
            pos_batch_size=pos_batch_size,
            seed=seed,
        )
        diagnostics_tracker = MCTSDiagnosticsTracker(tree=tree)
    elif strategy == "uniform":
        uni_sampler = UniformSampler(
            negative_indices=neg_idx_arr,
            neg_batch_size=neg_batch_size,
            positive_indices=pos_idx_arr,
            pos_batch_size=pos_batch_size,
            seed=seed,
        )
    elif strategy == "ohem":
        ohem_sampler = OHEMSampler(
            negative_indices=neg_idx_arr,
            neg_batch_size=neg_batch_size,
            ohem_ratio=4,
            positive_indices=pos_idx_arr,
            pos_batch_size=pos_batch_size,
            seed=seed,
        )

    history: list[dict[str, float]] = []
    visitation_snapshots: list[dict[str, Any]] = []
    exposure_tracker = ExposureTracker(num_classes=100)
    start_time = time.time()
    accumulated_train_loss = 0.0
    loss_count = 0

    for step in range(1, num_steps + 1):
        model.train()
        optimizer.zero_grad()

        mcts_leaf_id: int | None = None
        mcts_batch_len: int = 0

        if strategy in ("mcts_strict", "mcts_enhanced"):
            assert agent is not None and mcts_sampler is not None
            mcts_leaf_id, _ = agent.select_leaf()
            batch = mcts_sampler.sample_batch(mcts_leaf_id)
            step_indices = batch.all_indices
            mcts_batch_len = int(batch.mcts_indices.size)
        elif strategy == "uniform":
            assert uni_sampler is not None
            _, _, step_indices = uni_sampler.sample_batch()
        elif strategy == "ohem":
            assert ohem_sampler is not None
            _, _, step_indices = ohem_sampler.sample_batch(
                model=model, features=train_features, device=dev, is_binary=True
            )

        batch_x = train_features[step_indices].to(dev)
        # binary labels: positives are 1.0, negatives are 0.0
        batch_y = torch.as_tensor(
            (train_binary_labels[step_indices] == 1).astype(np.float32), device=dev
        )

        logits = model(batch_x)
        loss = F.binary_cross_entropy_with_logits(logits, batch_y)
        loss.backward()
        optimizer.step()

        accumulated_train_loss += float(loss.item())
        loss_count += 1

        batch_classes = train_binary_labels[step_indices]
        exposure_tracker.record_batch(batch_classes, step_indices)
        if mcts_leaf_id is not None:
            exposure_tracker.record_leaf_selection(mcts_leaf_id)
            if tree is not None:
                exposure_tracker.record_node_visit(mcts_leaf_id)

        if strategy in ("mcts_strict", "mcts_enhanced") and mcts_leaf_id is not None:
            assert agent is not None and diagnostics_tracker is not None
            # Compute MCTS reward on the MCTS sub-batch
            # In DualStreamBatch, all_indices is concat(positives, negatives), where negatives starts with mcts_indices
            # Positives count is pos_batch_size. So mcts logits are at [pos_batch_size : pos_batch_size + mcts_batch_len]
            mcts_start = pos_batch_size
            mcts_end = pos_batch_size + mcts_batch_len
            mcts_logits = logits[mcts_start:mcts_end]
            reward = compute_mcts_reward(mcts_logits, is_binary=True)
            agent.update(mcts_leaf_id, reward)
            diagnostics_tracker.record_step(step, mcts_leaf_id, reward)

            if enable_leaf_expansion and tree is not None:
                maybe_expand_leaf(
                    tree=tree,
                    leaf_node_id=mcts_leaf_id,
                    model=model,
                    train_inputs=train_features,
                    is_binary=True,
                    split_threshold=split_threshold,
                    k_sub=k_sub,
                    device=dev,
                )

        if step % eval_every == 0 or step == num_steps:
            metrics = evaluate_scenario1_metrics(
                model=model,
                features=val_features,
                binary_labels=val_binary_labels,
                device=dev,
            )
            step_train_loss = float(accumulated_train_loss / max(1, loss_count))
            accumulated_train_loss = 0.0
            loss_count = 0
            current_lr = float(optimizer.param_groups[0]["lr"])
            elapsed_time = float(time.time() - start_time)

            metrics["step"] = step
            metrics["train_loss"] = step_train_loss
            metrics["val_loss"] = metrics["loss"]
            metrics["val_F1"] = metrics["f1"]
            metrics["val_AUPRC"] = metrics["auprc"]
            metrics["learning_rate"] = current_lr
            metrics["elapsed_time"] = elapsed_time

            if tree is not None:
                ent_nats, ent_norm = compute_visitation_entropy(tree)
                metrics["entropy_nats"] = ent_nats
                metrics["normalized_entropy"] = ent_norm

                leaf_ids = tree.leaf_node_ids()
                active_visits = {int(nid): int(tree.visits[nid]) for nid in leaf_ids if tree.visits[nid] > 0}
                sc_dist = compute_superclass_visit_distribution(tree)
                visitation_snapshots.append({
                    "step": int(step),
                    "tree_id": getattr(tree, "tree_id", "semantic"),
                    "seed": int(seed),
                    "visited_node_ids": list(active_visits.keys()),
                    "visit_counts": active_visits,
                    "superclass_visit_distribution": sc_dist,
                    "normalization_method": "ucb1_minmax" if getattr(tree, "normalize_q", True) else "standard_ucb1",
                })
            history.append(metrics)

    final_val_eval = history[-1] if history else evaluate_scenario1_metrics(model, val_features, val_binary_labels, device=dev)
    final_test_eval = None
    if test_features is not None and test_binary_labels is not None:
        final_test_eval = evaluate_scenario1_metrics(model, test_features, test_binary_labels, device=dev)

    final_eval = final_test_eval if final_test_eval is not None else final_val_eval
    exposure_diag = exposure_tracker.compute_diagnostics()
    return {
        "strategy": strategy,
        "history": history,
        "step_history": history,
        "exposure_diagnostics": exposure_diag,
        "final_metrics": final_eval,
        "val_metrics": final_val_eval,
        "test_metrics": final_test_eval,
        "model": model,
        "diagnostics": diagnostics_tracker.summary() if diagnostics_tracker is not None else None,
        "final_num_leaves": tree.num_leaves if tree is not None else None,
        "visitation_checkpoints": visitation_snapshots,
    }


def train_scenario2(
    train_features: torch.Tensor,
    train_labels: np.ndarray | torch.Tensor,
    val_features: torch.Tensor | None = None,
    val_labels: np.ndarray | torch.Tensor | None = None,
    test_features: torch.Tensor | None = None,
    test_labels: np.ndarray | torch.Tensor | None = None,
    strategy: str = "mcts_enhanced",
    tree: FastHierarchyTree | None = None,
    num_classes: int = 100,
    num_steps: int = 1000,
    eval_every: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    label_smoothing: float = 0.1,
    exploration_c: float = 2.0,
    momentum_alpha: float = 0.3,
    dual_stream_ratio: float = 0.5,
    enable_leaf_expansion: bool = False,
    split_threshold: int = 50,
    k_sub: int = 3,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict[str, object]:
    """Train downstream MLP classifier under Scenario 2 (100-way Multi-Class Curriculum).

    Periodic evaluations at eval_every steps are computed strictly on validation data (val_features),
    preventing scientific data leakage from the test split.
    """
    if val_features is None or val_labels is None:
        raise ValueError(
            "val_features and val_labels must be provided for validation evaluation during training. "
            "Periodic evaluation on test data is prohibited."
        )
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Supported: {SUPPORTED_STRATEGIES}."
        )
    if num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}.")
    if eval_every <= 0:
        raise ValueError(f"eval_every must be > 0, got {eval_every}.")

    set_global_seed(seed)
    dev = device if device is not None else torch.device("cpu")

    input_dim = int(train_features.shape[1])
    model = Tier1MLPHead(input_dim=input_dim, hidden_dim=128, num_outputs=num_classes).to(dev)
    optimizer = create_mlp_optimizer(model, lr=lr)

    train_lbl_arr = (
        train_labels.detach().cpu().numpy()
        if isinstance(train_labels, torch.Tensor)
        else np.asarray(train_labels)
    ).astype(np.int64)

    all_indices = np.arange(train_lbl_arr.size, dtype=np.int64)

    agent: MCTSHNMAgent | None = None
    mcts_sampler: DualStreamBatchSampler | None = None
    diagnostics_tracker: MCTSDiagnosticsTracker | None = None
    uni_sampler: UniformSampler | None = None
    ohem_sampler: OHEMSampler | None = None
    uniform_grouped_sampler: UniformGroupedBatchSampler | None = None

    if strategy in ("mcts_strict", "mcts_enhanced"):
        if tree is None:
            raise ValueError(f"tree is required for MCTS strategy '{strategy}'.")
        q_norm = (strategy == "mcts_enhanced")
        beta = dual_stream_ratio if strategy == "mcts_enhanced" else 0.0
        agent = MCTSHNMAgent(
            tree=tree,
            exploration_c=exploration_c,
            momentum_alpha=momentum_alpha,
            q_norm=q_norm,
        )
        mcts_sampler = DualStreamBatchSampler(
            tree=tree,
            neg_batch_size=batch_size,
            dual_stream_ratio=beta,
            pos_batch_size=0,
            seed=seed,
        )
        diagnostics_tracker = MCTSDiagnosticsTracker(tree=tree)
    elif strategy == "uniform":
        uni_sampler = UniformSampler(
            negative_indices=all_indices,
            neg_batch_size=batch_size,
            pos_batch_size=0,
            seed=seed,
        )
    elif strategy == "ohem":
        ohem_sampler = OHEMSampler(
            negative_indices=all_indices,
            neg_batch_size=batch_size,
            ohem_ratio=4,
            pos_batch_size=0,
            seed=seed,
        )
    elif strategy == "uniform_grouped":
        c2i = (
            tree.class_to_indices
            if tree is not None
            else {c: np.where(train_lbl_arr == c)[0] for c in range(num_classes)}
        )
        uniform_grouped_sampler = UniformGroupedBatchSampler(
            class_to_indices=c2i,
            batch_size=batch_size,
            dual_stream_ratio=dual_stream_ratio,
            seed=seed,
        )

    history: list[dict[str, float]] = []
    visitation_snapshots: list[dict[str, Any]] = []
    exposure_tracker = ExposureTracker(num_classes=num_classes)
    start_time = time.time()
    accumulated_train_loss = 0.0
    loss_count = 0

    for step in range(1, num_steps + 1):
        model.train()
        optimizer.zero_grad()

        mcts_leaf_id: int | None = None
        mcts_batch_len: int = 0
        leaf_cls_id: int | None = None

        if strategy in ("mcts_strict", "mcts_enhanced"):
            assert agent is not None and mcts_sampler is not None
            mcts_leaf_id, _ = agent.select_leaf()
            batch = mcts_sampler.sample_batch(mcts_leaf_id)
            step_indices = batch.all_indices
            mcts_batch_len = int(batch.mcts_indices.size)
            leaf_cls_id = batch.leaf_class_id
        elif strategy == "uniform":
            assert uni_sampler is not None
            _, _, step_indices = uni_sampler.sample_batch()
        elif strategy == "ohem":
            assert ohem_sampler is not None
            _, _, step_indices = ohem_sampler.sample_batch(
                model=model,
                features=train_features,
                device=dev,
                is_binary=False,
                target_labels=train_lbl_arr,
            )
        elif strategy == "uniform_grouped":
            assert uniform_grouped_sampler is not None
            batch = uniform_grouped_sampler.sample_batch()
            step_indices = batch.all_indices
            leaf_cls_id = batch.leaf_class_id

        batch_x = train_features[step_indices].to(dev)
        batch_y = torch.as_tensor(train_lbl_arr[step_indices], dtype=torch.long, device=dev)

        logits = model(batch_x)
        loss = F.cross_entropy(logits, batch_y, label_smoothing=label_smoothing)
        loss.backward()
        optimizer.step()

        accumulated_train_loss += float(loss.item())
        loss_count += 1

        exposure_tracker.record_batch(train_lbl_arr[step_indices], step_indices)
        if mcts_leaf_id is not None:
            exposure_tracker.record_leaf_selection(mcts_leaf_id)
            if tree is not None:
                exposure_tracker.record_node_visit(mcts_leaf_id)

        if strategy in ("mcts_strict", "mcts_enhanced") and mcts_leaf_id is not None:
            assert agent is not None and diagnostics_tracker is not None
            mcts_logits = logits[:mcts_batch_len]
            mcts_targets = batch_y[:mcts_batch_len]
            reward = compute_mcts_reward(mcts_logits, is_binary=False, target_labels=mcts_targets)
            agent.update(mcts_leaf_id, reward)
            diagnostics_tracker.record_step(step, mcts_leaf_id, reward)

            if enable_leaf_expansion and tree is not None:
                maybe_expand_leaf(
                    tree=tree,
                    leaf_node_id=mcts_leaf_id,
                    model=model,
                    train_inputs=train_features,
                    is_binary=False,
                    split_threshold=split_threshold,
                    k_sub=k_sub,
                    device=dev,
                )

        if step % eval_every == 0 or step == num_steps:
            metrics = evaluate_scenario2_metrics(
                model=model,
                features=val_features,
                labels=val_labels,
                num_classes=num_classes,
                device=dev,
            )
            step_train_loss = float(accumulated_train_loss / max(1, loss_count))
            accumulated_train_loss = 0.0
            loss_count = 0
            current_lr = float(optimizer.param_groups[0]["lr"])
            elapsed_time = float(time.time() - start_time)

            metrics["step"] = step
            metrics["train_loss"] = step_train_loss
            metrics["val_loss"] = metrics["loss"]
            metrics["val_F1"] = metrics["macro_f1"]
            metrics["val_AUPRC"] = 0.0
            metrics["macro_f1"] = metrics["macro_f1"]
            metrics["accuracy"] = metrics["top1_accuracy"]
            metrics["learning_rate"] = current_lr
            metrics["elapsed_time"] = elapsed_time

            if tree is not None:
                ent_nats, ent_norm = compute_visitation_entropy(tree)
                metrics["entropy_nats"] = ent_nats
                metrics["normalized_entropy"] = ent_norm

                leaf_ids = tree.leaf_node_ids()
                active_visits = {int(nid): int(tree.visits[nid]) for nid in leaf_ids if tree.visits[nid] > 0}
                sc_dist = compute_superclass_visit_distribution(tree)
                visitation_snapshots.append({
                    "step": int(step),
                    "tree_id": getattr(tree, "tree_id", "semantic"),
                    "seed": int(seed),
                    "visited_node_ids": list(active_visits.keys()),
                    "visit_counts": active_visits,
                    "superclass_visit_distribution": sc_dist,
                    "normalization_method": "ucb1_minmax" if getattr(tree, "normalize_q", True) else "standard_ucb1",
                })
            history.append(metrics)

    final_val_eval = history[-1] if history else evaluate_scenario2_metrics(model, val_features, val_labels, num_classes=num_classes, device=dev)
    final_test_eval = None
    if test_features is not None and test_labels is not None:
        final_test_eval = evaluate_scenario2_metrics(model, test_features, test_labels, num_classes=num_classes, device=dev)

    final_eval = final_test_eval if final_test_eval is not None else final_val_eval
    exposure_diag = exposure_tracker.compute_diagnostics()
    return {
        "strategy": strategy,
        "history": history,
        "step_history": history,
        "exposure_diagnostics": exposure_diag,
        "final_metrics": final_eval,
        "val_metrics": final_val_eval,
        "test_metrics": final_test_eval,
        "model": model,
        "diagnostics": diagnostics_tracker.summary() if diagnostics_tracker is not None else None,
        "final_num_leaves": tree.num_leaves if tree is not None else None,
        "visitation_checkpoints": visitation_snapshots,
    }
