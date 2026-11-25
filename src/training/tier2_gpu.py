"""Tier-2 GPU campaign: fine-tuning the Tier-1 backbone under the Tier-1 evaluation protocol.

Tier 1 trains an MLP head on *frozen* ImageNet ResNet-18 embeddings. This module trains the
same head on top of the same ImageNet ResNet-18, but with the backbone *unfrozen*, on the same
stratified train/validation split, the same Scenario-1 positive subsampling, the same batch
composition (16 positives + 48 negatives) and the same evaluation protocol (validation-calibrated
threshold, best-validation-AUPRC checkpoint, test scored only after training). The only intended
difference between tiers is therefore whether the representation can adapt to the mined negatives,
which is the question left open by the frozen-embedding benchmark.

GPU-oriented choices (none of them change the sampling algorithms):
- images are kept on the device as uint8 32x32 tensors and preprocessed per batch on the GPU
  (optional random crop + horizontal flip at 32x32, bilinear upsampling to 224, ImageNet normalisation);
- automatic mixed precision (bf16 when supported, otherwise fp16 with gradient scaling);
- one JSON file per run, so an interrupted campaign resumes by skipping finished runs.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import platform
import time
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.data.cifar100_images import load_cifar100_raw_images
from src.data.cifar100_taxonomy import CIFAR100_FINE_CLASSES
from src.data.extract_embeddings import create_stratified_train_val_indices
from src.data.splits import IMBALANCE_RATIOS, build_scenario1_split
from src.mcts.agent import MCTSHNMAgent, compute_mcts_reward
from src.mcts.diagnostics import MCTSDiagnosticsTracker, compute_visitation_entropy
from src.mcts.sampler import DualStreamBatchSampler
from src.models.mlp_head import Tier1MLPHead
from src.training.baselines import OHEMSampler, UniformSampler
from src.training.trainer import MCTS_STRATEGIES, resolve_mcts_hyperparameters
from src.trees.base_tree import FastHierarchyTree
from src.trees.random_tree import build_random_tree
from src.trees.semantic_tree import build_semantic_tree
from src.utils.experiment_config import ExecutionMode, ExperimentConfig
from src.utils.metrics import calibrate_binary_threshold, evaluate_binary_metrics_at_threshold
from src.utils.seed import set_global_seed
from src.utils.statistics import compute_paired_statistical_tests

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TIER2_STRATEGIES = ("uniform", "ohem") + MCTS_STRATEGIES
TIER2_TREE_TYPES = ("semantic", "random")


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------


@dataclass
class Tier2Data:
    """CIFAR-100 images kept on the training device as uint8 ``[N, 3, 32, 32]`` tensors.

    ``train_*`` and ``val_*`` are the stratified 90/10 partition of the official training set
    used by Tier 1 (same ``val_fraction`` and ``split_seed``), so Scenario-1 index pools built
    from ``train_labels`` coincide with the Tier-1 pools for the same target class and seed.
    """

    train_images: torch.Tensor
    train_labels: np.ndarray
    val_images: torch.Tensor
    val_labels: np.ndarray
    test_images: torch.Tensor
    test_labels: np.ndarray
    split_seed: int
    val_fraction: float


def _to_uint8(images: torch.Tensor) -> torch.Tensor:
    if images.dtype == torch.uint8:
        return images
    return (images.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def prepare_tier2_data(
    data_dir: str | Path = "data",
    device: torch.device | str = "cpu",
    val_fraction: float = 0.1,
    split_seed: int = 42,
    config: ExperimentConfig | None = None,
    download: bool = True,
) -> Tier2Data:
    """Load CIFAR-100, rebuild the Tier-1 stratified validation split and move images to ``device``."""
    cfg = config if config is not None else ExperimentConfig(execution_mode=ExecutionMode.REAL)
    dev = torch.device(device)
    full_train, full_train_labels = load_cifar100_raw_images(
        data_dir=data_dir, split="train", download=download, config=cfg
    )
    test_images, test_labels = load_cifar100_raw_images(
        data_dir=data_dir, split="test", download=download, config=cfg
    )
    train_idx, val_idx = create_stratified_train_val_indices(
        full_train_labels, val_fraction=val_fraction, seed=split_seed
    )
    full_train_u8 = _to_uint8(full_train)
    return Tier2Data(
        train_images=full_train_u8[torch.as_tensor(train_idx)].contiguous().to(dev),
        train_labels=np.asarray(full_train_labels)[train_idx].astype(np.int64),
        val_images=full_train_u8[torch.as_tensor(val_idx)].contiguous().to(dev),
        val_labels=np.asarray(full_train_labels)[val_idx].astype(np.int64),
        test_images=_to_uint8(test_images).contiguous().to(dev),
        test_labels=np.asarray(test_labels).astype(np.int64),
        split_seed=int(split_seed),
        val_fraction=float(val_fraction),
    )


def random_crop_flip(
    images: torch.Tensor, generator: torch.Generator | None = None, padding: int = 4
) -> torch.Tensor:
    """Vectorised CIFAR augmentation: reflect-pad, random ``HxW`` crop and random horizontal flip."""
    n, _, h, w = images.shape
    dev = images.device
    padded = F.pad(images, (padding, padding, padding, padding), mode="reflect")
    off_y = torch.randint(0, 2 * padding + 1, (n,), device=dev, generator=generator)
    off_x = torch.randint(0, 2 * padding + 1, (n,), device=dev, generator=generator)
    rows = (off_y[:, None] + torch.arange(h, device=dev)[None, :])[:, None, :, None]
    cols = (off_x[:, None] + torch.arange(w, device=dev)[None, :])[:, None, None, :]
    batch_ix = torch.arange(n, device=dev)[:, None, None, None]
    chan_ix = torch.arange(images.shape[1], device=dev)[None, :, None, None]
    out = padded[batch_ix, chan_ix, rows, cols]
    flip = torch.rand(n, device=dev, generator=generator) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


class GPUPreprocessor(nn.Module):
    """uint8 32x32 -> (augment) -> bilinear upsample -> ImageNet normalisation, all on device."""

    def __init__(self, image_size: int = 224) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(
        self, images_u8: torch.Tensor, augment: bool = False, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        x = images_u8.float().div_(255.0)
        if augment:
            x = random_crop_flip(x, generator=generator)
        if self.image_size != x.shape[-1]:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std


# --------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------


class FineTunedResNet18Binary(nn.Module):
    """ImageNet ResNet-18 backbone (512-d pooled features) + the Tier-1 MLP head (512->128->1)."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False, dropout: float = 0.1) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        backbone.fc = nn.Identity()  # type: ignore[assignment]
        self.backbone = backbone
        self.head = Tier1MLPHead(input_dim=512, hidden_dim=128, num_outputs=1, dropout=dropout, squeeze_binary=True)
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True) -> "FineTunedResNet18Binary":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()  # frozen backbone == Tier-1 feature extractor (BN in eval mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# --------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------


@dataclass
class Tier2RunConfig:
    """Hyperparameters of one Tier-2 run (defaults follow the Tier-1 principal benchmark)."""

    num_steps: int = 3000
    eval_every: int = 150
    pos_batch_size: int = 16
    neg_batch_size: int = 48
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.0
    exploration_c: float = 2.0
    momentum_alpha: float = 0.3
    dual_stream_ratio: float = 0.5
    ohem_ratio: int = 4
    image_size: int = 224
    augment: bool = True
    amp: bool = True
    freeze_backbone: bool = False
    pretrained: bool = True
    eval_batch_size: int = 500
    cudnn_benchmark: bool = True


def _amp_dtype(device: torch.device) -> torch.dtype | None:
    """bf16 only on GPUs with native bf16 tensor cores (Ampere+, e.g. A100); fp16 otherwise.

    ``torch.cuda.is_bf16_supported()`` is True on a T4 (Turing) too, but there bf16 is emulated
    and runs several times slower than fp16.
    """
    if device.type != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 and torch.cuda.is_bf16_supported() else torch.float16


@torch.no_grad()
def predict_probs(
    model: nn.Module,
    pre: GPUPreprocessor,
    images_u8: torch.Tensor,
    batch_size: int,
    amp_dtype: torch.dtype | None,
) -> np.ndarray:
    """Positive-class probabilities for ``images_u8`` (model left in its original mode)."""
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    out: list[torch.Tensor] = []
    for i in range(0, images_u8.shape[0], batch_size):
        x = pre(images_u8[i : i + batch_size].to(dev))
        with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        out.append(logits.float().view(-1).cpu())
    model.train(was_training)
    return torch.sigmoid(torch.cat(out)).numpy().astype(np.float64)


def binary_metrics(probs: np.ndarray, targets: np.ndarray, threshold: float) -> dict[str, float]:
    """F1/precision/recall at ``threshold`` plus AUPRC, FPR at 95% recall and BCE loss."""
    y = np.asarray(targets, dtype=np.int64)
    m = evaluate_binary_metrics_at_threshold(probs, y, threshold=threshold)
    if 0 < y.sum() < y.size:
        fpr, tpr, _ = roc_curve(y, probs)
        m["fpr_at_95_recall"] = float(np.min(fpr[tpr >= 0.95]))
        m["auprc"] = float(average_precision_score(y, probs))
    else:
        m["fpr_at_95_recall"] = float("nan")
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    m["loss"] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return m


def train_tier2_scenario1(
    data: Tier2Data,
    positive_indices: np.ndarray,
    negative_indices: np.ndarray,
    target_class: int,
    strategy: str,
    tree: FastHierarchyTree | None,
    cfg: Tier2RunConfig,
    seed: int,
    device: torch.device,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Train one Tier-2 Scenario-1 run and return a JSON-serialisable record (no model weights)."""
    if strategy not in TIER2_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Supported: {TIER2_STRATEGIES}.")
    if strategy in MCTS_STRATEGIES and tree is None:
        raise ValueError(f"tree is required for MCTS strategy '{strategy}'.")

    set_global_seed(seed)
    if device.type == "cuda" and cfg.cudnn_benchmark:
        # Faster convolutions; results are then reproducible up to cuDNN non-determinism.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    amp_dtype = _amp_dtype(device) if cfg.amp else None
    use_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    aug_gen = torch.Generator(device=device).manual_seed(seed) if cfg.augment else None

    pre = GPUPreprocessor(cfg.image_size).to(device)
    model = FineTunedResNet18Binary(pretrained=cfg.pretrained, freeze_backbone=cfg.freeze_backbone).to(device)
    model = model.to(memory_format=torch.channels_last)
    param_groups: list[dict[str, Any]] = [{"params": model.head.parameters(), "lr": cfg.lr_head}]
    if not cfg.freeze_backbone:
        param_groups.append({"params": model.backbone.parameters(), "lr": cfg.lr_backbone})
    optimizer = torch.optim.Adam(param_groups, weight_decay=cfg.weight_decay)

    train_bin = (data.train_labels == target_class).astype(np.int64)
    val_bin = (data.val_labels == target_class).astype(np.int64)
    test_bin = (data.test_labels == target_class).astype(np.int64)
    pos_arr = np.asarray(positive_indices, dtype=np.int64)
    neg_arr = np.asarray(negative_indices, dtype=np.int64)

    agent: MCTSHNMAgent | None = None
    mcts_sampler: DualStreamBatchSampler | None = None
    tracker: MCTSDiagnosticsTracker | None = None
    uni_sampler: UniformSampler | None = None
    ohem_sampler: OHEMSampler | None = None
    if strategy in MCTS_STRATEGIES:
        assert tree is not None
        q_norm, beta = resolve_mcts_hyperparameters(strategy, dual_stream_ratio=cfg.dual_stream_ratio)
        agent = MCTSHNMAgent(tree=tree, exploration_c=cfg.exploration_c, momentum_alpha=cfg.momentum_alpha, q_norm=q_norm)
        mcts_sampler = DualStreamBatchSampler(
            tree=tree, neg_batch_size=cfg.neg_batch_size, dual_stream_ratio=beta,
            positive_indices=pos_arr, pos_batch_size=cfg.pos_batch_size, seed=seed,
        )
        tracker = MCTSDiagnosticsTracker(tree=tree)
    elif strategy == "uniform":
        uni_sampler = UniformSampler(
            negative_indices=neg_arr, neg_batch_size=cfg.neg_batch_size,
            positive_indices=pos_arr, pos_batch_size=cfg.pos_batch_size, seed=seed,
        )
    else:
        ohem_sampler = OHEMSampler(
            negative_indices=neg_arr, neg_batch_size=cfg.neg_batch_size, ohem_ratio=cfg.ohem_ratio,
            positive_indices=pos_arr, pos_batch_size=cfg.pos_batch_size, seed=seed,
        )

    class _Mined(nn.Module):
        """Adapter so OHEMSampler scores raw uint8 candidates with the full preprocessing.

        ``inner`` is a registered submodule, so OHEMSampler's ``eval()`` reaches the backbone and
        candidate scoring never updates BatchNorm statistics.
        """

        def __init__(self) -> None:
            super().__init__()
            self.inner = model
            self.pre = pre

        def forward(self, x_u8: torch.Tensor) -> torch.Tensor:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                return self.inner(self.pre(x_u8).contiguous(memory_format=torch.channels_last)).float()

    mined_scorer = _Mined()

    history: list[dict[str, float]] = []
    best_auprc, best_tau, best_state, best_step = -1.0, 0.5, None, 0
    tau_val = 0.5
    grad_samples = mining_forwards = 0
    running_loss, running_n = 0.0, 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    for step in range(1, cfg.num_steps + 1):
        model.train()
        leaf_id: int | None = None
        mcts_len = 0
        if agent is not None and mcts_sampler is not None:
            leaf_id, _ = agent.select_leaf()
            batch = mcts_sampler.sample_batch(leaf_id)
            idx = batch.all_indices
            mcts_len = int(batch.mcts_indices.size)
        elif uni_sampler is not None:
            _, _, idx = uni_sampler.sample_batch()
        else:
            assert ohem_sampler is not None
            _, _, idx = ohem_sampler.sample_batch(
                model=mined_scorer, features=data.train_images, device=device, is_binary=True
            )
            mining_forwards += cfg.ohem_ratio * cfg.neg_batch_size
        grad_samples += int(idx.size)

        idx_t = torch.as_tensor(idx, device=device)
        x = pre(data.train_images[idx_t], augment=cfg.augment, generator=aug_gen)
        x = x.contiguous(memory_format=torch.channels_last)
        y = torch.as_tensor(train_bin[idx].astype(np.float32), device=device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        logits = logits.float()
        loss = F.binary_cross_entropy_with_logits(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += float(loss.detach())
        running_n += 1

        if agent is not None and leaf_id is not None and tracker is not None:
            mcts_logits = logits.detach()[cfg.pos_batch_size : cfg.pos_batch_size + mcts_len]
            reward = compute_mcts_reward(mcts_logits, is_binary=True)
            agent.update(leaf_id, reward)
            tracker.record_step(step, leaf_id, reward)

        if step % cfg.eval_every == 0 or step == cfg.num_steps:
            val_probs = predict_probs(model, pre, data.val_images, cfg.eval_batch_size, amp_dtype)
            tau_val = calibrate_binary_threshold(val_probs, val_bin)
            m = binary_metrics(val_probs, val_bin, tau_val)
            rec: dict[str, float] = {
                "step": step,
                "train_loss": running_loss / max(1, running_n),
                "val_loss": m["loss"],
                "val_F1": m["f1"],
                "val_AUPRC": m["auprc"],
                "calibrated_threshold": float(tau_val),
                "elapsed_time": time.time() - t0,
            }
            running_loss, running_n = 0.0, 0
            if tree is not None:
                ent_nats, ent_norm = compute_visitation_entropy(tree)
                rec["entropy_nats"], rec["normalized_entropy"] = float(ent_nats), float(ent_norm)
            history.append(rec)
            if m["auprc"] > best_auprc:
                best_auprc, best_tau, best_step = float(m["auprc"]), float(tau_val), step
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            if progress is not None:
                progress(
                    f"step {step}/{cfg.num_steps} | train_loss {rec['train_loss']:.4f} | "
                    f"val_F1 {m['f1']:.3f} | val_AUPRC {m['auprc']:.3f} | {rec['elapsed_time']:.0f}s"
                )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    # Test is scored only after training; never used for thresholds or checkpoint selection.
    test_probs_last = predict_probs(model, pre, data.test_images, cfg.eval_batch_size, amp_dtype)
    last_metrics = binary_metrics(test_probs_last, test_bin, tau_val)
    last_metrics["f1_at_0.5"] = binary_metrics(test_probs_last, test_bin, 0.5)["f1"]
    if best_state is not None:
        model.load_state_dict(best_state)
    test_probs_best = predict_probs(model, pre, data.test_images, cfg.eval_batch_size, amp_dtype)
    best_metrics = binary_metrics(test_probs_best, test_bin, best_tau)
    best_metrics["f1_at_0.5"] = binary_metrics(test_probs_best, test_bin, 0.5)["f1"]
    best_metrics["selected_step"] = best_step

    return {
        "strategy": strategy,
        "final_metrics": last_metrics,
        "best_validation_metrics": best_metrics,
        "history": history,
        "computational_costs": {
            "total_updates": cfg.num_steps,
            "gradient_samples": grad_samples,
            "mining_forwards": mining_forwards,
            "elapsed_seconds": elapsed,
        },
        "mcts_diagnostics": tracker.summary() if tracker is not None else None,
    }


# --------------------------------------------------------------------------------------------
# Campaign
# --------------------------------------------------------------------------------------------


@dataclass
class Tier2CampaignConfig:
    """Grid of Tier-2 Scenario-1 runs; every cell is one JSON record in ``output_dir``."""

    campaign: str = "tier2_finetune"
    target_classes: tuple[int | str, ...] = (0,)
    ratios: tuple[str, ...] = ("1:50",)
    strategies: tuple[str, ...] = ("uniform", "ohem", "mcts_strict", "mcts_enhanced")
    tree_type: str = "semantic"
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    run: Tier2RunConfig = field(default_factory=Tier2RunConfig)
    output_dir: str = "results/real/tier2"


def run_id_for(campaign: str, target: int, ratio: str, tree_type: str, strategy: str, seed: int) -> str:
    tree_tag = tree_type if strategy in MCTS_STRATEGIES else "none"
    return f"{campaign}_cls{target}_r{ratio.replace(':', '_')}_{tree_tag}_{strategy}_s{seed}"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)!r}")


def _environment(device: torch.device) -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "platform": platform.platform(),
    }
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(device)
        env["amp_dtype"] = str(_amp_dtype(device))
    try:
        import subprocess

        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
            cwd=Path(__file__).resolve().parents[2],
        ).stdout.strip() or None
    except Exception:  # pragma: no cover - git missing
        env["git_commit"] = None
    return env


def run_tier2_campaign(
    cfg: Tier2CampaignConfig,
    data: Tier2Data,
    device: torch.device,
    log: Callable[[str], None] = print,
    verbose_steps: bool = False,
) -> list[dict[str, Any]]:
    """Run (or resume) every grid cell, writing ``<run_id>.json`` per run. Returns all records."""
    if cfg.tree_type not in TIER2_TREE_TYPES:
        raise ValueError(f"tree_type must be one of {TIER2_TREE_TYPES}, got {cfg.tree_type!r}.")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _environment(device)
    records: list[dict[str, Any]] = []
    grid = [
        (tc, ratio, strat, seed)
        for tc in cfg.target_classes
        for ratio in cfg.ratios
        for strat in cfg.strategies
        for seed in cfg.seeds
    ]
    for i, (tc, ratio, strat, seed) in enumerate(grid, start=1):
        split = build_scenario1_split(
            fine_labels=data.train_labels, target_class=tc,
            pos_fraction=IMBALANCE_RATIOS[ratio], seed=seed,
        )
        rid = run_id_for(cfg.campaign, split.target_class_id, ratio, cfg.tree_type, strat, seed)
        path = out_dir / f"{rid}.json"
        if path.exists():
            records.append(json.loads(path.read_text(encoding="utf-8")))
            log(f"[{i}/{len(grid)}] {rid}: already done, skipping")
            continue

        tree: FastHierarchyTree | None = None
        if strat in MCTS_STRATEGIES:
            if cfg.tree_type == "semantic":
                tree = build_semantic_tree(split.negative_class_to_indices)
            else:
                tree = build_random_tree(split.negative_class_to_indices, seed=seed)

        log(f"[{i}/{len(grid)}] {rid}: training ({split.positive_indices.size} positives)")
        res = train_tier2_scenario1(
            data=data, positive_indices=split.positive_indices, negative_indices=split.negative_indices,
            target_class=split.target_class_id, strategy=strat, tree=tree, cfg=cfg.run,
            seed=seed, device=device, progress=(lambda s: log("    " + s)) if verbose_steps else None,
        )
        record = {
            "run_id": rid,
            "campaign": cfg.campaign,
            "tier": "tier2_finetune" if not cfg.run.freeze_backbone else "tier2_frozen_backbone",
            "scenario": "scenario1",
            "target_class": split.target_class_id,
            "target_class_name": CIFAR100_FINE_CLASSES[split.target_class_id],
            "imbalance_ratio": ratio,
            "num_positives": int(split.positive_indices.size),
            "tree_type": cfg.tree_type if strat in MCTS_STRATEGIES else None,
            "seed": seed,
            "split_seed": data.split_seed,
            "run_config": asdict(cfg.run),
            "environment": env,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **res,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=1, default=_json_default), encoding="utf-8")
        tmp.replace(path)
        fm, bm = record["final_metrics"], record["best_validation_metrics"]
        log(
            f"    done in {res['computational_costs']['elapsed_seconds']:.0f}s | "
            f"test F1 {fm['f1']:.3f} AUPRC {fm['auprc']:.3f} | best-val ckpt F1 {bm['f1']:.3f} AUPRC {bm['auprc']:.3f}"
        )
        records.append(record)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return records


def load_tier2_records(output_dir: str | Path, campaign: str | None = None) -> list[dict[str, Any]]:
    """Load every per-run JSON in ``output_dir`` (optionally filtered by campaign name)."""
    recs = []
    for p in sorted(Path(output_dir).glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if campaign is None or r.get("campaign") == campaign:
            recs.append(r)
    return recs


def summarize_tier2(
    records: Sequence[dict[str, Any]],
    reference: str = "mcts_strict",
    metrics: Sequence[str] = ("f1", "auprc", "fpr_at_95_recall"),
    protocol: str = "final_metrics",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Mean/SD per (class, ratio, strategy) and seed-paired tests of each strategy vs ``reference``."""
    groups: dict[tuple[int, str, str], dict[int, dict[str, float]]] = {}
    for r in records:
        key = (int(r["target_class"]), str(r["imbalance_ratio"]), str(r["strategy"]))
        groups.setdefault(key, {})[int(r["seed"])] = r[protocol]

    summary_rows: list[dict[str, Any]] = []
    for (tc, ratio, strat), by_seed in sorted(groups.items()):
        row: dict[str, Any] = {"target_class": tc, "ratio": ratio, "strategy": strat, "n_seeds": len(by_seed)}
        for m in metrics:
            vals = np.array([v[m] for v in by_seed.values()], dtype=np.float64)
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_sd"] = float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan")
        summary_rows.append(row)

    test_rows: list[dict[str, Any]] = []
    for (tc, ratio, strat), by_seed in sorted(groups.items()):
        ref = groups.get((tc, ratio, reference))
        if strat == reference or ref is None:
            continue
        seeds = sorted(set(by_seed) & set(ref))
        if len(seeds) < 2:
            continue
        for m in metrics[:2]:
            stats = compute_paired_statistical_tests(
                [by_seed[s][m] for s in seeds], [ref[s][m] for s in seeds], keys_a=seeds, keys_b=seeds
            )
            test_rows.append({
                "target_class": tc, "ratio": ratio, "comparison": f"{strat} - {reference}", "metric": m,
                "n_pairs": len(seeds),
                **{k: v for k, v in stats.items() if isinstance(v, (int, float, str, type(None)))},
            })
    return summary_rows, test_rows


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def pooled_tier2_tests(
    records: Sequence[dict[str, Any]],
    reference: str = "mcts_strict",
    metric: str = "auprc",
    protocol: str = "final_metrics",
) -> list[dict[str, Any]]:
    """Each strategy vs ``reference`` over all (target class, ratio, seed) pairs, Holm-corrected.

    Pooling across target classes gives enough pairs for a Wilcoxon signed-rank test (with five
    seeds per class its smallest attainable two-sided p is 0.0625). Pairs from the same class are
    not independent draws, so these tests describe the evaluated classes only.
    """
    from scipy.stats import ttest_1samp, wilcoxon

    by_key: dict[tuple[int, str, int], dict[str, float]] = {}
    for r in records:
        key = (int(r["target_class"]), str(r["imbalance_ratio"]), int(r["seed"]))
        by_key.setdefault(key, {})[str(r["strategy"])] = float(r[protocol][metric])
    strategies = sorted({s for v in by_key.values() for s in v} - {reference})
    rows: list[dict[str, Any]] = []
    for strat in strategies:
        diffs = np.array([v[strat] - v[reference] for v in by_key.values() if strat in v and reference in v])
        if diffs.size < 2:
            continue
        rows.append({
            "comparison": f"{strat} - {reference}", "metric": metric, "protocol": protocol,
            "n_pairs": int(diffs.size), "mean_diff": float(diffs.mean()), "sd_diff": float(diffs.std(ddof=1)),
            "reference_wins": int((diffs < 0).sum()),
            "t_pvalue": float(ttest_1samp(diffs, 0.0).pvalue),
            "wilcoxon_pvalue": float(wilcoxon(diffs).pvalue) if np.any(diffs != 0) else 1.0,
        })
    for row, adj in zip(rows, holm_adjust([r["wilcoxon_pvalue"] for r in rows])):
        row["wilcoxon_holm"] = adj
    return rows


_STRATEGY_LABELS = {
    "uniform": "Uniform Random",
    "ohem": "OHEM",
    "mcts_strict": "Strict MCTS",
    "mcts_enhanced": "Enhanced MCTS-HNM",
}


def _write_lines(lines: list[str], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_tier2_latex_table(
    records: Sequence[dict[str, Any]], path: str | Path, label: str = "tab:tier2_finetune"
) -> Path:
    """Per-class AUPRC/F1 table for both checkpoint protocols (bold = best mean per class and column)."""
    protocols = ("final_metrics", "best_validation_metrics")
    summaries = {p: summarize_tier2(records, protocol=p)[0] for p in protocols}
    classes = sorted({int(r["target_class"]) for r in summaries["final_metrics"]})
    names = {int(r["target_class"]): str(r["target_class_name"]) for r in records}
    order = [s for s in _STRATEGY_LABELS if any(r["strategy"] == s for r in summaries["final_metrics"])]
    cols = [(p, m) for p in protocols for m in ("auprc", "f1")]

    def cell(row: dict[str, Any], m: str, best: bool) -> str:
        scale, fmt = (100.0, "{:.1f}") if m == "f1" else (1.0, "{:.3f}")
        txt = f"{fmt.format(row[m + '_mean'] * scale)} $\\pm$ {fmt.format(row[m + '_sd'] * scale)}"
        return f"\\textbf{{{txt}}}" if best else txt

    lines = [
        "% Auto-generated by src/training/tier2_gpu.py::write_tier2_latex_table",
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Tier 2 (fine-tuned ResNet-18), Scenario 1 at 2\% positive retention ($1{:}50$, nine positive "
        r"training images), five paired seeds per target class. Test AUPRC and minority F1 (\%, "
        r"validation-calibrated threshold) as mean $\pm$ sample SD, at the final step and at the checkpoint "
        r"with the highest validation AUPRC. Bold marks the best mean per class and column; it does not "
        r"indicate significance (see Table~\ref{tab:tier2_pooled}).}",
        f"\\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llcccc}", r"\toprule",
        r" & & \multicolumn{2}{c}{Final step} & \multicolumn{2}{c}{Validation-selected checkpoint} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Target class & Sampler & AUPRC & F1 (\%) & AUPRC & F1 (\%) \\",
        r"\midrule",
    ]
    for ci, tc in enumerate(classes):
        rows = {p: {r["strategy"]: r for r in summaries[p] if int(r["target_class"]) == tc} for p in protocols}
        best = {(p, m): max(rows[p].values(), key=lambda r: r[m + "_mean"])["strategy"] for p, m in cols}
        for si, strat in enumerate(order):
            first = names.get(tc, str(tc)).replace("_", " ") if si == 0 else ""
            cells = [cell(rows[p][strat], m, best[(p, m)] == strat) for p, m in cols]
            lines.append(f"  {first} & {_STRATEGY_LABELS[strat]} & " + " & ".join(cells) + r" \\")
        if ci < len(classes) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}"]
    return _write_lines(lines, path)


def _fmt_p(p: float) -> str:
    return "$<0.001$" if p < 0.001 else f"{p:.3f}"


def write_tier2_pooled_latex_table(
    records: Sequence[dict[str, Any]], path: str | Path, label: str = "tab:tier2_pooled"
) -> Path:
    """Pooled seed-paired comparisons against Strict MCTS for both protocols and both metrics."""
    n_classes = len({int(r["target_class"]) for r in records})
    n_seeds = len({int(r["seed"]) for r in records})
    lines = [
        "% Auto-generated by src/training/tier2_gpu.py::write_tier2_pooled_latex_table",
        r"\begin{table}[t]", r"\centering", r"\small",
        f"\\caption{{Tier 2: paired differences to Strict MCTS pooled over {n_classes} target classes $\\times$ "
        f"{n_seeds} seeds ($n={n_classes * n_seeds}$ pairs). Wins count pairs in which Strict MCTS scores higher. "
        r"$p_{\text{Holm}}$: Wilcoxon signed-rank test, Holm-adjusted over the three comparisons within each "
        r"protocol and metric. Pairs from the same class are not independent draws, so the tests describe the "
        r"evaluated classes only.}",
        f"\\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrcc}", r"\toprule",
        r"Protocol, metric & Sampler $-$ Strict & $\Delta$ mean & Strict wins & $p_{\text{Holm}}$ \\",
        r"\midrule",
    ]
    blocks = [
        ("final_metrics", "auprc", "Final, AUPRC"), ("final_metrics", "f1", r"Final, F1 (\%)"),
        ("best_validation_metrics", "auprc", "Val.-selected, AUPRC"),
        ("best_validation_metrics", "f1", r"Val.-selected, F1 (\%)"),
    ]
    for bi, (proto, metric, title) in enumerate(blocks):
        rows = pooled_tier2_tests(records, metric=metric, protocol=proto)
        rows.sort(key=lambda r: list(_STRATEGY_LABELS).index(r["comparison"].split(" - ")[0]))
        for ri, r in enumerate(rows):
            scale, fmt = (100.0, "{:+.1f}") if metric == "f1" else (1.0, "{:+.3f}")
            strat = r["comparison"].split(" - ")[0]
            lines.append(
                f"  {title if ri == 0 else ''} & {_STRATEGY_LABELS[strat]} & ${fmt.format(r['mean_diff'] * scale)}$ & "
                f"{r['reference_wins']}/{r['n_pairs']} & {_fmt_p(r['wilcoxon_holm'])} \\\\"
            )
        if bi < len(blocks) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"]
    return _write_lines(lines, path)


def write_rows_csv(rows: Sequence[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# Moving results from a remote kernel to the local checkout
# ---------------------------------------------------------------------------
# With the VS Code Colab extension neither drive.mount nor files.download works, but cell outputs are
# saved into the local .ipynb. The notebook therefore prints the results zip as base64 between two
# markers, and the helper below turns those markers back into files on the local machine.

ZIP_BEGIN_MARKER = "BEGIN_ZIP"
ZIP_END_MARKER = "END_ZIP"


def print_results_zip(output_dir: str | Path, archive_base: str | Path, line_width: int = 100) -> Path:
    """Zip `output_dir` and print it as base64 lines between ZIP_BEGIN_MARKER and ZIP_END_MARKER."""
    import base64
    import shutil

    zip_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=str(output_dir)))
    encoded = base64.b64encode(zip_path.read_bytes()).decode("ascii")
    print(f"zip bytes={zip_path.stat().st_size} files={len(list(Path(output_dir).glob('*.json')))}")
    print(ZIP_BEGIN_MARKER)
    for i in range(0, len(encoded), line_width):
        print(encoded[i:i + line_width])
    print(ZIP_END_MARKER)
    return zip_path


def extract_results_zip_from_notebook(notebook_path: str | Path, output_dir: str | Path) -> list[str]:
    """Decode the last zip printed by `print_results_zip` in a saved notebook into `output_dir`."""
    import base64
    import io
    import re
    import zipfile

    nb = json.loads(Path(notebook_path).read_text(encoding="utf-8"))
    payload = None
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            text = out.get("text", "")
            text = "".join(text) if isinstance(text, list) else text
            if ZIP_BEGIN_MARKER in text and ZIP_END_MARKER in text:
                payload = text.split(ZIP_BEGIN_MARKER, 1)[1].split(ZIP_END_MARKER, 1)[0]
    if payload is None:
        raise ValueError(f"No {ZIP_BEGIN_MARKER}/{ZIP_END_MARKER} block in {notebook_path}")
    archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(re.sub(r"\s", "", payload))))
    bad = archive.testzip()
    if bad is not None:
        raise ValueError(f"Corrupted member in embedded zip: {bad}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    archive.extractall(output_dir)
    return archive.namelist()


if __name__ == "__main__":  # python -m src.training.tier2_gpu NOTEBOOK [OUTPUT_DIR]
    import sys

    nb_arg = sys.argv[1] if len(sys.argv) > 1 else "notebooks/06_tier2_gpu_campaign.ipynb"
    out_arg = sys.argv[2] if len(sys.argv) > 2 else "results/real/tier2"
    names = extract_results_zip_from_notebook(nb_arg, out_arg)
    print(f"Extracted {len(names)} files into {out_arg}")
