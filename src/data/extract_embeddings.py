"""Frozen ResNet-18 embedding extraction and atomic disk cache management for CIFAR-100."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

from src.data.cifar100_taxonomy import fine_to_coarse_tensor
from src.utils.seed import get_torch_generator, set_global_seed
from src.utils.experiment_config import (
    ExecutionMode,
    ExperimentConfig,
    get_isolated_paths,
    save_provenance_metadata,
    validate_provenance_metadata,
)

EMBEDDING_DIM: int = 512
REQUIRED_SPLITS: tuple[str, ...] = ("train", "val", "test")
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def create_stratified_train_val_indices(
    labels: np.ndarray | Sequence[int],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition dataset indices into disjoint stratified train and validation splits.

    Args:
        labels: 1D sequence or array of integer class labels.
        val_fraction: Fraction of each class allocated to validation (`0.0 < val_fraction < 1.0`).
        seed: Non-negative integer seed for deterministic per-class shuffling.

    Returns:
        Tuple `(train_indices, val_indices)` of sorted 1D `int64` NumPy arrays.

    Raises:
        ValueError: If `val_fraction` is outside `(0.0, 1.0)` or `labels` is empty.
    """
    set_global_seed(seed)
    if not (0.0 < float(val_fraction) < 1.0):
        raise ValueError(f"val_fraction must be in (0.0, 1.0), got {val_fraction}.")

    label_arr = np.asarray(labels, dtype=np.int64)
    if label_arr.ndim != 1 or label_arr.size == 0:
        raise ValueError("labels must be a non-empty 1D sequence of integer class IDs.")

    rng = np.random.default_rng(seed)
    unique_classes = np.unique(label_arr)

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []

    for cls_id in unique_classes:
        cls_indices = np.flatnonzero(label_arr == cls_id)
        if cls_indices.size < 2:
            raise ValueError(f"Class {cls_id} has fewer than 2 samples ({cls_indices.size}); cannot split.")
        permuted = rng.permutation(cls_indices)
        n_val = int(round(cls_indices.size * val_fraction))
        n_val = max(1, min(cls_indices.size - 1, n_val))
        val_parts.append(permuted[:n_val])
        train_parts.append(permuted[n_val:])

    train_indices = np.sort(np.concatenate(train_parts)).astype(np.int64)
    val_indices = np.sort(np.concatenate(val_parts)).astype(np.int64)
    return train_indices, val_indices


def build_cifar100_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Return the deterministic preprocessing transform for ResNet-18 feature extraction."""
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_frozen_resnet18_extractor(
    device: str | torch.device = "cpu",
    pretrained: bool = True,
) -> nn.Module:
    """Construct a frozen ResNet-18 feature extractor outputting 512-d pooled vectors."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = models.resnet18(weights=weights)
    backbone.fc = nn.Identity()  # type: ignore[assignment]
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone.to(device)


def _extract_split_tensors(
    dataset: Dataset[Any],
    indices: np.ndarray,
    backbone: nn.Module,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Run batched forward pass over `Subset(dataset, indices)` and collect 512-d embeddings."""
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        generator=get_torch_generator(seed),
    )

    emb_batches: list[torch.Tensor] = []
    fine_batches: list[torch.Tensor] = []

    backbone.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            feats = backbone(images)
            if feats.ndim > 2:
                feats = torch.flatten(feats, start_dim=1)
            emb_batches.append(feats.detach().cpu().to(dtype=torch.float32))
            fine_batches.append(torch.as_tensor(targets, dtype=torch.int64))

    embeddings = torch.cat(emb_batches, dim=0)
    fine_labels = torch.cat(fine_batches, dim=0)
    coarse_labels = fine_to_coarse_tensor(fine_labels)
    idx_tensor = torch.as_tensor(indices, dtype=torch.int64)

    return {
        "embeddings": embeddings,
        "fine_labels": fine_labels,
        "coarse_labels": coarse_labels,
        "indices": idx_tensor,
    }


def validate_embedding_bundle(bundle: Any) -> dict[str, Any]:
    """Validate structure, shapes, dtypes, and numerical integrity of an embedding cache dict."""
    if not isinstance(bundle, dict):
        raise ValueError(f"Embedding cache bundle must be a dict, got {type(bundle).__name__}.")

    for split_name in REQUIRED_SPLITS:
        if split_name not in bundle or not isinstance(bundle[split_name], dict):
            raise ValueError(f"Embedding cache missing required split '{split_name}'.")

    for split_name in REQUIRED_SPLITS:
        split_data = bundle[split_name]
        for required_key in ("embeddings", "fine_labels", "coarse_labels", "indices"):
            if required_key not in split_data:
                raise ValueError(f"Split '{split_name}' missing required tensor key '{required_key}'.")

        emb = split_data["embeddings"]
        fine = split_data["fine_labels"]
        coarse = split_data["coarse_labels"]
        indices = split_data["indices"]

        if not isinstance(emb, torch.Tensor) or emb.ndim != 2 or emb.shape[1] != EMBEDDING_DIM:
            shape_desc = getattr(emb, "shape", None)
            raise ValueError(
                f"Split '{split_name}' embeddings must have shape [N, {EMBEDDING_DIM}], got {shape_desc}."
            )
        if emb.dtype != torch.float32:
            raise ValueError(f"Split '{split_name}' embeddings must have dtype float32, got {emb.dtype}.")
        if not torch.isfinite(emb).all():
            raise ValueError(f"Split '{split_name}' embeddings contain NaN or Inf values.")

        n_samples = emb.shape[0]
        if n_samples == 0:
            raise ValueError(f"Split '{split_name}' cannot be empty.")

        for name, tensor in (("fine_labels", fine), ("coarse_labels", coarse), ("indices", indices)):
            if not isinstance(tensor, torch.Tensor) or tensor.shape != (n_samples,):
                raise ValueError(
                    f"Split '{split_name}' {name} must have shape ({n_samples},), got {getattr(tensor, 'shape', None)}."
                )

        expected_coarse = fine_to_coarse_tensor(fine)
        if not torch.equal(coarse, expected_coarse):
            raise ValueError(f"Split '{split_name}' coarse_labels do not match fine_to_coarse_tensor(fine_labels).")

    return bundle


def load_embedding_cache(cache_path: str | Path) -> dict[str, Any]:
    """Load and validate an embedding bundle from disk.

    Raises:
        ValueError: If the file does not exist, cannot be deserialized, or fails integrity checks.
    """
    path = Path(cache_path)
    if not path.is_file():
        raise ValueError(f"Embedding cache file does not exist at '{path}'.")
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"Failed to load embedding cache from '{path}': {exc}") from exc

    return validate_embedding_bundle(bundle)


def extract_cifar100_embeddings(
    data_dir: str | Path = "data",
    cache_path: str | Path = "embeddings/cifar100_resnet18.pt",
    batch_size: int = 256,
    val_fraction: float = 0.1,
    seed: int = 42,
    device: str | torch.device | None = None,
    overwrite: bool = False,
    dataset_factory: Callable[[Path, bool, Callable[..., Any]], Dataset[Any]] | None = None,
    backbone: nn.Module | None = None,
) -> dict[str, Any]:
    """Extract or load cached frozen ResNet-18 `[N, 512]` embeddings for CIFAR-100.

    Args:
        data_dir: Directory where raw CIFAR-100 files are stored/downloaded.
        cache_path: Target path for the serialized `.pt` embedding bundle.
        batch_size: Mini-batch size used during feature extraction.
        val_fraction: Stratified validation fraction carved from the training set.
        seed: Random seed for stratified splitting and DataLoader determinism.
        device: Compute device (`"cpu"`, `"cuda"`, or `None` for auto-detection).
        overwrite: If False and `cache_path` exists, validates and returns the cached bundle.
        dataset_factory: Optional callable `(root, train, transform) -> Dataset` for testing.
        backbone: Optional pre-built feature extractor module outputting `[B, 512]` tensors.

    Returns:
        Validated dictionary containing `"train"`, `"val"`, `"test"`, and `"metadata"` entries.
    """
    set_global_seed(seed)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    target_cache = Path(cache_path)
    if target_cache.exists() and not overwrite:
        return load_embedding_cache(target_cache)

    resolved_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    transform = build_cifar100_eval_transform()
    root_dir = Path(data_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    if dataset_factory is not None:
        full_train_ds = dataset_factory(root_dir, True, transform)
        test_ds = dataset_factory(root_dir, False, transform)
    else:
        full_train_ds = datasets.CIFAR100(root=str(root_dir), train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR100(root=str(root_dir), train=False, download=True, transform=transform)

    train_targets = getattr(full_train_ds, "targets", None)
    if train_targets is None:
        train_targets = [int(full_train_ds[i][1]) for i in range(len(full_train_ds))]  # type: ignore[arg-type]

    train_indices, val_indices = create_stratified_train_val_indices(
        train_targets, val_fraction=val_fraction, seed=seed
    )
    test_indices = np.arange(len(test_ds), dtype=np.int64)  # type: ignore[arg-type]

    extractor = (
        backbone.to(resolved_device)
        if backbone is not None
        else build_frozen_resnet18_extractor(device=resolved_device, pretrained=True)
    )

    train_split = _extract_split_tensors(
        full_train_ds, train_indices, extractor, batch_size, resolved_device, seed
    )
    val_split = _extract_split_tensors(
        full_train_ds, val_indices, extractor, batch_size, resolved_device, seed
    )
    test_split = _extract_split_tensors(
        test_ds, test_indices, extractor, batch_size, resolved_device, seed
    )

    bundle: dict[str, Any] = {
        "train": train_split,
        "val": val_split,
        "test": test_split,
        "metadata": {
            "backbone": "resnet18_imagenet1k_v1" if backbone is None else type(backbone).__name__,
            "embedding_dim": EMBEDDING_DIM,
            "seed": seed,
            "val_fraction": val_fraction,
            "train_count": int(train_indices.size),
            "val_count": int(val_indices.size),
            "test_count": int(test_indices.size),
        },
    }

    validate_embedding_bundle(bundle)

    target_cache.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_cache.with_suffix(target_cache.suffix + ".tmp")
    torch.save(bundle, tmp_path)
    tmp_path.replace(target_cache)

    return bundle


def extract_or_generate_embeddings(
    config: ExperimentConfig,
    data_dir: str | Path = "data",
    cache_path: str | Path | None = None,
    batch_size: int = 256,
    val_fraction: float = 0.1,
    seed: int | None = None,
    device: str | torch.device | None = None,
    overwrite: bool = False,
    dataset_factory: Callable[[Path, bool, Callable[..., Any]], Dataset[Any]] | None = None,
    backbone: nn.Module | None = None,
    samples_per_class: int = 5,
) -> dict[str, Any]:
    """Extract real CIFAR-100 embeddings or generate synthetic fixtures based on config.execution_mode.

    In REAL mode:
        Extracts real CIFAR-100 embeddings or loads validated real cache.
        If real CIFAR-100 data cannot be accessed, raises FileNotFoundError.
        Writes provenance metadata with synthetic_data=False.

    In SMOKE mode:
        Generates a small deterministic synthetic embedding bundle with 100 classes.
        Writes provenance metadata with synthetic_data=True.
    """
    effective_seed = seed if seed is not None else (config.seed if isinstance(config.seed, int) else 42)
    paths = get_isolated_paths(config)

    resolved_cache = Path(cache_path) if cache_path is not None else (paths.embeddings_dir / "cifar100_resnet18.pt")

    if resolved_cache.is_file() and not overwrite:
        meta_file = resolved_cache.parent / f"{resolved_cache.name}.meta.json"
        if meta_file.exists():
            validate_provenance_metadata(meta_file, expected_mode=config.execution_mode)
        elif config.execution_mode == ExecutionMode.REAL:
            raise FileNotFoundError(f"Missing mandatory provenance metadata in REAL mode: {meta_file}")
        bundle = load_embedding_cache(resolved_cache)
        return bundle

    if config.execution_mode == ExecutionMode.REAL:
        try:
            bundle = extract_cifar100_embeddings(
                data_dir=data_dir,
                cache_path=resolved_cache,
                batch_size=batch_size,
                val_fraction=val_fraction,
                seed=effective_seed,
                device=device,
                overwrite=overwrite,
                dataset_factory=dataset_factory,
                backbone=backbone,
            )
        except Exception as exc:
            if isinstance(exc, FileNotFoundError):
                raise
            raise FileNotFoundError(
                f"Failed to extract real CIFAR-100 embeddings in REAL mode: {exc}. "
                "Ensure CIFAR-100 raw data is downloaded to data_dir."
            ) from exc

        save_provenance_metadata(
            resolved_cache,
            config,
            extra={
                "data_source": str(resolved_cache.name),
                "synthetic_data": False,
                "dataset": "CIFAR-100",
            },
        )
        return bundle

    # SMOKE mode: generate deterministic synthetic bundle
    rng = np.random.default_rng(effective_seed)
    n_classes = 100
    train_n = n_classes * samples_per_class
    val_n = n_classes * max(1, samples_per_class // 5)
    test_n = n_classes * max(1, samples_per_class // 5)

    def _make_split(n_samples: int) -> dict[str, torch.Tensor]:
        labels = np.array([i % n_classes for i in range(n_samples)], dtype=np.int64)
        feats = rng.standard_normal((n_samples, EMBEDDING_DIM)).astype(np.float32)
        fine_tensor = torch.from_numpy(labels)
        coarse_tensor = fine_to_coarse_tensor(fine_tensor)
        return {
            "embeddings": torch.from_numpy(feats),
            "fine_labels": fine_tensor,
            "coarse_labels": coarse_tensor,
            "indices": torch.arange(n_samples, dtype=torch.int64),
        }

    bundle = {
        "train": _make_split(train_n),
        "val": _make_split(val_n),
        "test": _make_split(test_n),
        "metadata": {
            "backbone": "synthetic_smoke_generator",
            "embedding_dim": EMBEDDING_DIM,
            "seed": effective_seed,
            "synthetic_data": True,
        },
    }

    validate_embedding_bundle(bundle)

    resolved_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, resolved_cache)

    save_provenance_metadata(
        resolved_cache,
        config,
        extra={
            "data_source": str(resolved_cache.name),
            "synthetic_data": True,
            "dataset": "CIFAR-100-Synthetic-Smoke",
        },
    )
    return bundle
