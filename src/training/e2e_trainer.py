"""Tier-2 End-to-End ResNet-18 Vision Pipeline and Training Engine."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mcts.agent import MCTSHNMAgent, compute_mcts_reward
from src.mcts.diagnostics import MCTSDiagnosticsTracker, compute_visitation_entropy
from src.mcts.expansion import maybe_expand_leaf
from src.mcts.sampler import DualStreamBatchSampler
from src.training.baselines import OHEMSampler, UniformSampler
from src.training.trainer import (
    SUPPORTED_STRATEGIES,
    evaluate_scenario1_metrics,
    evaluate_scenario2_metrics,
)
from src.trees.base_tree import FastHierarchyTree
from src.trees.confusion_tree import build_confusion_tree, validate_confusion_matrix
from src.utils.experiment_config import (
    ExecutionMode,
    ExperimentConfig,
    save_provenance_metadata,
)
from src.utils.seed import set_global_seed
from src.utils.statistics import compute_paired_statistical_tests

logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock with 2 3x3 convolutions and skip connection."""

    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class Tier2ResNet18(nn.Module):
    """CIFAR-adapted ResNet-18 vision architecture for 32x32 images."""

    def __init__(
        self,
        num_outputs: int = 1,
        width_multiplier: float = 1.0,
        dropout: float = 0.1,
        squeeze_binary: bool = True,
    ) -> None:
        super().__init__()
        if num_outputs <= 0:
            raise ValueError(f"num_outputs must be >= 1, got {num_outputs}.")
        if width_multiplier <= 0.0:
            raise ValueError(f"width_multiplier must be > 0.0, got {width_multiplier}.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}.")

        self.num_outputs = num_outputs
        self.squeeze_binary = squeeze_binary

        # Base channel widths scaled by width_multiplier
        base_c = int(round(64 * width_multiplier))
        c1 = max(8, base_c)
        c2 = max(16, c1 * 2)
        c3 = max(32, c2 * 2)
        c4 = max(64, c3 * 2)

        self.in_planes = c1
        # CIFAR-specific: 3x3 conv with stride 1, padding 1 (no maxpool)
        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)

        self.layer1 = self._make_layer(c1, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(c2, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(c3, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(c4, num_blocks=2, stride=2)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.fc = nn.Linear(c4 * BasicBlock.expansion, num_outputs)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: list[nn.Module] = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global average pooled feature vectors [B, D]."""
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Input must be 4D tensor with 3 channels [B, 3, H, W], got {tuple(x.shape)}.")
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        feats = self.forward_features(x)
        dropped = self.dropout(feats)
        logits = self.fc(dropped)
        if self.squeeze_binary and self.num_outputs == 1:
            return logits.squeeze(-1)
        return logits


def refresh_confusion_tree_from_model(
    model: nn.Module,
    val_images: torch.Tensor,
    val_labels: np.ndarray | torch.Tensor,
    class_to_indices: dict[int, np.ndarray],
    num_classes: int = 100,
    is_binary: bool = False,
    batch_size: int = 256,
    smoothing: float = 1e-3,
    device: torch.device | None = None,
    seed: int = 42,
) -> FastHierarchyTree:
    """Evaluate model on validation images and rebuild the Ward Confusion Tree."""
    set_global_seed(seed)
    dev = device if device is not None else next(model.parameters()).device
    model_mode = model.training
    model.eval()

    y_val = (
        val_labels.detach().cpu().numpy()
        if isinstance(val_labels, torch.Tensor)
        else np.asarray(val_labels)
    ).astype(np.int64)

    all_logits: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, val_images.shape[0], batch_size):
            batch_x = val_images[i : i + batch_size].to(dev)
            logits = model(batch_x)
            all_logits.append(logits.detach().cpu())

    if model_mode:
        model.train()

    logits_t = torch.cat(all_logits, dim=0)

    if is_binary:
        # Construct confusion between active classes
        k = len(class_to_indices)
        c_mat = np.eye(k, dtype=np.float64) + float(smoothing)
    else:
        probs = F.softmax(logits_t, dim=-1).numpy()
        preds = np.argmax(probs, axis=1)
        c_hard = sk_confusion_matrix(
            y_val, preds, labels=np.arange(num_classes, dtype=np.int64)
        ).astype(np.float64)
        c_base = c_hard + float(smoothing)

        # Map each active class (including sub-clusters) to base class id
        sorted_active = sorted(class_to_indices.keys())
        k = len(sorted_active)
        c_mat = np.zeros((k, k), dtype=np.float64)
        for i, cid_i in enumerate(sorted_active):
            base_i = cid_i % 1000
            for j, cid_j in enumerate(sorted_active):
                base_j = cid_j % 1000
                if base_i < num_classes and base_j < num_classes:
                    c_mat[i, j] = c_base[base_i, base_j]
                else:
                    c_mat[i, j] = 1.0 if i == j else float(smoothing)

    c_valid = validate_confusion_matrix(c_mat)
    return build_confusion_tree(
        confusion_matrix=c_valid,
        class_to_indices=class_to_indices,
    )


def train_e2e_scenario1(
    train_images: torch.Tensor,
    train_binary_labels: np.ndarray | torch.Tensor,
    positive_indices: np.ndarray,
    negative_indices: np.ndarray,
    test_images: torch.Tensor,
    test_binary_labels: np.ndarray | torch.Tensor,
    strategy: str = "mcts_enhanced",
    tree: FastHierarchyTree | None = None,
    val_images: torch.Tensor | None = None,
    val_binary_labels: np.ndarray | torch.Tensor | None = None,
    num_steps: int = 500,
    eval_every: int = 50,
    refresh_tree_every: int | None = None,
    pos_batch_size: int = 16,
    neg_batch_size: int = 48,
    lr: float = 1e-3,
    width_multiplier: float = 1.0,
    dropout: float = 0.1,
    exploration_c: float = 2.0,
    momentum_alpha: float = 0.3,
    dual_stream_ratio: float = 0.5,
    enable_leaf_expansion: bool = False,
    split_threshold: int = 50,
    k_sub: int = 3,
    seed: int = 42,
    device: torch.device | None = None,
    checkpoints_dir: str | Path | None = None,
) -> dict[str, object]:
    """Train Tier2ResNet18 end-to-end under Scenario 1 (Binary One-vs-Rest)."""
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Supported: {SUPPORTED_STRATEGIES}.")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}.")
    if eval_every <= 0:
        raise ValueError(f"eval_every must be > 0, got {eval_every}.")
    if train_images.ndim != 4 or train_images.shape[1] != 3:
        raise ValueError(f"train_images must be 4D RGB tensor, got {train_images.shape}.")

    set_global_seed(seed)
    dev = device if device is not None else torch.device("cpu")

    model = Tier2ResNet18(
        num_outputs=1,
        width_multiplier=width_multiplier,
        dropout=dropout,
        squeeze_binary=True,
    ).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pos_idx_arr = np.asarray(positive_indices, dtype=np.int64)
    neg_idx_arr = np.asarray(negative_indices, dtype=np.int64)

    agent: MCTSHNMAgent | None = None
    mcts_sampler: DualStreamBatchSampler | None = None
    diagnostics_tracker: MCTSDiagnosticsTracker | None = None
    uni_sampler: UniformSampler | None = None
    ohem_sampler: OHEMSampler | None = None

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
                model=model, features=train_images, device=dev, is_binary=True
            )

        batch_x = train_images[step_indices].to(dev)
        batch_y = torch.as_tensor(
            (train_binary_labels[step_indices] == 1).astype(np.float32), device=dev
        )

        logits = model(batch_x)
        loss = F.binary_cross_entropy_with_logits(logits, batch_y)
        loss.backward()
        optimizer.step()

        if strategy in ("mcts_strict", "mcts_enhanced") and mcts_leaf_id is not None:
            assert agent is not None and diagnostics_tracker is not None
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
                    train_inputs=train_images,
                    is_binary=True,
                    split_threshold=split_threshold,
                    k_sub=k_sub,
                    device=dev,
                )

        # Periodic tree refresh
        if (
            refresh_tree_every is not None
            and step % refresh_tree_every == 0
            and tree is not None
            and tree.tree_type == "confusion_ward"
            and val_images is not None
            and val_binary_labels is not None
        ):
            new_tree = refresh_confusion_tree_from_model(
                model=model,
                val_images=val_images,
                val_labels=val_binary_labels,
                class_to_indices=tree.class_to_indices,
                is_binary=True,
                device=dev,
                seed=seed + step,
            )
            # Reassign tree references
            tree.parents = new_tree.parents
            tree.children = new_tree.children
            tree.child_counts = new_tree.child_counts
            tree.is_leaf = new_tree.is_leaf
            tree.depths = new_tree.depths
            tree.leaf_to_class_id = new_tree.leaf_to_class_id
            tree.instance_counts = new_tree.instance_counts
            tree.visits = new_tree.visits
            tree.q_values = new_tree.q_values
            tree.node_names = new_tree.node_names

        if step % eval_every == 0 or step == num_steps:
            metrics = evaluate_scenario1_metrics(
                model=model,
                features=test_images,
                binary_labels=test_binary_labels,
                device=dev,
            )
            metrics["step"] = step
            if tree is not None:
                ent_nats, ent_norm = compute_visitation_entropy(tree)
                metrics["entropy_nats"] = ent_nats
                metrics["normalized_entropy"] = ent_norm
            history.append(metrics)

    final_eval = history[-1] if history else evaluate_scenario1_metrics(model, test_images, test_binary_labels, device=dev)
    checkpoint_path: str | None = None
    if checkpoints_dir is not None:
        cdir = Path(checkpoints_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        ckpt_file = cdir / f"tier2_resnet18_scenario1_{strategy}_seed{seed}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "final_metrics": final_eval,
            "seed": seed,
            "strategy": strategy,
        }, ckpt_file)
        checkpoint_path = str(ckpt_file)

    return {
        "strategy": strategy,
        "history": history,
        "final_metrics": final_eval,
        "model": model,
        "checkpoint_path": checkpoint_path,
        "diagnostics": diagnostics_tracker.summary() if diagnostics_tracker is not None else None,
        "final_num_leaves": tree.num_leaves if tree is not None else None,
    }


def train_e2e_scenario2(
    train_images: torch.Tensor,
    train_labels: np.ndarray | torch.Tensor,
    test_images: torch.Tensor,
    test_labels: np.ndarray | torch.Tensor,
    strategy: str = "mcts_enhanced",
    tree: FastHierarchyTree | None = None,
    val_images: torch.Tensor | None = None,
    val_labels: np.ndarray | torch.Tensor | None = None,
    num_classes: int = 100,
    num_steps: int = 500,
    eval_every: int = 50,
    refresh_tree_every: int | None = None,
    batch_size: int = 64,
    lr: float = 1e-3,
    width_multiplier: float = 1.0,
    dropout: float = 0.1,
    label_smoothing: float = 0.1,
    exploration_c: float = 2.0,
    momentum_alpha: float = 0.3,
    dual_stream_ratio: float = 0.5,
    enable_leaf_expansion: bool = False,
    split_threshold: int = 50,
    k_sub: int = 3,
    seed: int = 42,
    device: torch.device | None = None,
    checkpoints_dir: str | Path | None = None,
) -> dict[str, object]:
    """Train Tier2ResNet18 end-to-end under Scenario 2 (100-way Multi-Class)."""
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Supported: {SUPPORTED_STRATEGIES}.")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}.")
    if eval_every <= 0:
        raise ValueError(f"eval_every must be > 0, got {eval_every}.")
    if train_images.ndim != 4 or train_images.shape[1] != 3:
        raise ValueError(f"train_images must be 4D RGB tensor, got {train_images.shape}.")

    set_global_seed(seed)
    dev = device if device is not None else torch.device("cpu")

    model = Tier2ResNet18(
        num_outputs=num_classes,
        width_multiplier=width_multiplier,
        dropout=dropout,
    ).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

    history: list[dict[str, float]] = []

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
                model=model,
                features=train_images,
                device=dev,
                is_binary=False,
                target_labels=train_lbl_arr,
            )

        batch_x = train_images[step_indices].to(dev)
        batch_y = torch.as_tensor(train_lbl_arr[step_indices], dtype=torch.long, device=dev)

        logits = model(batch_x)
        loss = F.cross_entropy(logits, batch_y, label_smoothing=label_smoothing)
        loss.backward()
        optimizer.step()

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
                    train_inputs=train_images,
                    is_binary=False,
                    split_threshold=split_threshold,
                    k_sub=k_sub,
                    device=dev,
                )

        # Periodic tree refresh
        if (
            refresh_tree_every is not None
            and step % refresh_tree_every == 0
            and tree is not None
            and tree.tree_type == "confusion_ward"
            and val_images is not None
            and val_labels is not None
        ):
            new_tree = refresh_confusion_tree_from_model(
                model=model,
                val_images=val_images,
                val_labels=val_labels,
                class_to_indices=tree.class_to_indices,
                num_classes=num_classes,
                is_binary=False,
                device=dev,
                seed=seed + step,
            )
            tree.parents = new_tree.parents
            tree.children = new_tree.children
            tree.child_counts = new_tree.child_counts
            tree.is_leaf = new_tree.is_leaf
            tree.depths = new_tree.depths
            tree.leaf_to_class_id = new_tree.leaf_to_class_id
            tree.instance_counts = new_tree.instance_counts
            tree.visits = new_tree.visits
            tree.q_values = new_tree.q_values
            tree.node_names = new_tree.node_names

        if step % eval_every == 0 or step == num_steps:
            metrics = evaluate_scenario2_metrics(
                model=model,
                features=test_images,
                labels=test_labels,
                num_classes=num_classes,
                device=dev,
            )
            metrics["step"] = step
            if tree is not None:
                ent_nats, ent_norm = compute_visitation_entropy(tree)
                metrics["entropy_nats"] = ent_nats
                metrics["normalized_entropy"] = ent_norm
            history.append(metrics)

    final_eval = history[-1] if history else evaluate_scenario2_metrics(model, test_images, test_labels, num_classes=num_classes, device=dev)
    checkpoint_path: str | None = None
    if checkpoints_dir is not None:
        cdir = Path(checkpoints_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        ckpt_file = cdir / f"tier2_resnet18_scenario2_{strategy}_seed{seed}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "final_metrics": final_eval,
            "seed": seed,
            "strategy": strategy,
        }, ckpt_file)
        checkpoint_path = str(ckpt_file)

    return {
        "strategy": strategy,
        "history": history,
        "final_metrics": final_eval,
        "model": model,
        "checkpoint_path": checkpoint_path,
        "diagnostics": diagnostics_tracker.summary() if diagnostics_tracker is not None else None,
        "final_num_leaves": tree.num_leaves if tree is not None else None,
    }


@dataclass
class E2EExperimentConfig:
    """Configuration for multi-seed Tier-2 end-to-end experiments."""

    scenario: int = 1  # 1 or 2
    strategies: tuple[str, ...] = ("uniform", "mcts_enhanced")
    seeds: tuple[int, ...] = (42, 43)
    num_steps: int = 100
    eval_every: int = 25
    pos_batch_size: int = 4
    neg_batch_size: int = 12
    lr: float = 1e-3
    width_multiplier: float = 0.25
    enable_leaf_expansion: bool = False
    split_threshold: int = 10
    k_sub: int = 2
    refresh_tree_every: int | None = None
    output_dir: str = "results"
    checkpoints_dir: str | None = None


def run_e2e_experiment(
    config: E2EExperimentConfig,
    train_images: torch.Tensor,
    train_labels: np.ndarray | torch.Tensor,
    test_images: torch.Tensor,
    test_labels: np.ndarray | torch.Tensor,
    tree_factory: Any = None,
    val_images: torch.Tensor | None = None,
    val_labels: np.ndarray | torch.Tensor | None = None,
    positive_indices: np.ndarray | None = None,
    negative_indices: np.ndarray | None = None,
    device: torch.device | None = None,
    experiment_config: ExperimentConfig | None = None,
) -> dict[str, object]:
    """Execute multi-seed Tier-2 E2E experiments, run statistical tests, and export JSON/CSV."""
    if config.scenario not in (1, 2):
        raise ValueError(f"scenario must be 1 or 2, got {config.scenario}.")
    if not config.strategies:
        raise ValueError("strategies cannot be empty.")
    if not config.seeds:
        raise ValueError("seeds cannot be empty.")

    out_path = Path(config.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results_by_strategy: dict[str, list[dict[str, float]]] = {s: [] for s in config.strategies}

    for strat in config.strategies:
        for seed in config.seeds:
            tree = tree_factory() if tree_factory is not None else None
            if config.scenario == 1:
                assert positive_indices is not None and negative_indices is not None
                res = train_e2e_scenario1(
                    train_images=train_images,
                    train_binary_labels=train_labels,
                    positive_indices=positive_indices,
                    negative_indices=negative_indices,
                    test_images=test_images,
                    test_binary_labels=test_labels,
                    strategy=strat,
                    tree=tree,
                    val_images=val_images,
                    val_binary_labels=val_labels,
                    num_steps=config.num_steps,
                    eval_every=config.eval_every,
                    refresh_tree_every=config.refresh_tree_every,
                    pos_batch_size=config.pos_batch_size,
                    neg_batch_size=config.neg_batch_size,
                    lr=config.lr,
                    width_multiplier=config.width_multiplier,
                    enable_leaf_expansion=config.enable_leaf_expansion,
                    split_threshold=config.split_threshold,
                    k_sub=config.k_sub,
                    seed=seed,
                    device=device,
                    checkpoints_dir=config.checkpoints_dir,
                )
            else:
                num_classes = int(np.max(train_labels) + 1)
                res = train_e2e_scenario2(
                    train_images=train_images,
                    train_labels=train_labels,
                    test_images=test_images,
                    test_labels=test_labels,
                    strategy=strat,
                    tree=tree,
                    val_images=val_images,
                    val_labels=val_labels,
                    num_classes=num_classes,
                    num_steps=config.num_steps,
                    eval_every=config.eval_every,
                    refresh_tree_every=config.refresh_tree_every,
                    batch_size=config.neg_batch_size,
                    lr=config.lr,
                    width_multiplier=config.width_multiplier,
                    enable_leaf_expansion=config.enable_leaf_expansion,
                    split_threshold=config.split_threshold,
                    k_sub=config.k_sub,
                    seed=seed,
                    device=device,
                    checkpoints_dir=config.checkpoints_dir,
                )
            results_by_strategy[strat].append(res["final_metrics"])

    # Aggregate and compute statistics against mcts_enhanced (or first strategy)
    primary_metric = "f1" if config.scenario == 1 else "top1_accuracy"
    ref_strat = "mcts_enhanced" if "mcts_enhanced" in config.strategies else config.strategies[0]
    ref_scores = [m[primary_metric] for m in results_by_strategy[ref_strat]]

    summary: dict[str, object] = {
        "scenario": config.scenario,
        "config": asdict(config),
        "strategies": {},
        "statistical_tests": {},
    }

    csv_rows: list[dict[str, object]] = []

    for strat, runs in results_by_strategy.items():
        strat_scores = [r[primary_metric] for r in runs]
        mean_val = float(np.mean(strat_scores))
        std_val = float(np.std(strat_scores))
        summary["strategies"][strat] = {
            "mean": mean_val,
            "std": std_val,
            "runs": runs,
        }
        csv_rows.append({"strategy": strat, "mean": mean_val, "std": std_val})

        if strat != ref_strat and len(config.seeds) >= 2:
            stats = compute_paired_statistical_tests(ref_scores, strat_scores)
            summary["statistical_tests"][f"{ref_strat}_vs_{strat}"] = stats

    json_file = out_path / f"e2e_scenario_{config.scenario}_results.json"
    csv_file = out_path / f"e2e_scenario_{config.scenario}_results.csv"

    # Atomic write JSON
    tmp_json = json_file.with_suffix(".tmp.json")
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    tmp_json.replace(json_file)

    # Atomic write CSV
    tmp_csv = csv_file.with_suffix(".tmp.csv")
    with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "mean", "std"])
        writer.writeheader()
        writer.writerows(csv_rows)
    tmp_csv.replace(csv_file)

    active_config = (
        experiment_config
        if experiment_config is not None
        else ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    )
    save_provenance_metadata(json_file, active_config)
    save_provenance_metadata(csv_file, active_config)

    return summary
