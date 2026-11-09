"""Unit tests for DATA-02: stratified train/val splitting, embedding extraction, and cache verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.data.extract_embeddings import (
    EMBEDDING_DIM,
    create_stratified_train_val_indices,
    extract_cifar100_embeddings,
    load_embedding_cache,
)


class _SyntheticCIFAR100Dataset(Dataset[tuple[torch.Tensor, int]]):
    """Deterministic in-memory dataset fixture with 100 fine classes."""

    def __init__(self, samples_per_class: int = 10, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        total = 100 * samples_per_class
        self.targets: list[int] = [cls_id for cls_id in range(100) for _ in range(samples_per_class)]
        # Generate deterministic [N, 3, 8, 8] tensors correlated with class id
        base = rng.standard_normal((total, 3, 8, 8)).astype(np.float32)
        self.images = torch.from_numpy(base)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], self.targets[index]


class _Deterministic512Backbone(nn.Module):
    """Lightweight deterministic backbone projecting [B, 3, H, W] -> [B, 512]."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(123)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.proj = nn.Linear(3 * 4 * 4, EMBEDDING_DIM, bias=False)
        for p in self.parameters():
            p.requires_grad = False
        self.forward_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        flat = torch.flatten(self.pool(x), start_dim=1)
        return self.proj(flat)


def test_stratified_train_val_indices() -> None:
    """Verify DATA-02 AC 1: 50,000 labels split into 45,000 train (450/class) and 5,000 val (50/class)."""
    labels = np.repeat(np.arange(100, dtype=np.int64), 500)
    train_idx, val_idx = create_stratified_train_val_indices(labels, val_fraction=0.1, seed=42)

    assert train_idx.shape == (45000,)
    assert val_idx.shape == (5000,)
    assert np.intersect1d(train_idx, val_idx).size == 0
    assert np.array_equal(np.union1d(train_idx, val_idx), np.arange(50000))

    train_counts = np.bincount(labels[train_idx], minlength=100)
    val_counts = np.bincount(labels[val_idx], minlength=100)
    assert np.all(train_counts == 450)
    assert np.all(val_counts == 50)

    # Verify determinism across repeated calls with same seed
    train_idx_2, val_idx_2 = create_stratified_train_val_indices(labels, val_fraction=0.1, seed=42)
    assert np.array_equal(train_idx, train_idx_2)
    assert np.array_equal(val_idx, val_idx_2)


def test_stratified_train_val_invalid_fraction_raises() -> None:
    """Verify DATA-02 Edge Case 2: val_fraction outside (0.0, 1.0) raises ValueError."""
    labels = np.repeat(np.arange(100, dtype=np.int64), 10)
    for bad_frac in (0.0, -0.2, 1.0, 1.5):
        with pytest.raises(ValueError, match="val_fraction"):
            create_stratified_train_val_indices(labels, val_fraction=bad_frac, seed=42)


def test_extract_cifar100_embeddings_creates_cache(tmp_path: Path) -> None:
    """Verify DATA-02 AC 2: extracts float32 [N, 512] tensors and saves atomic bundle to cache_path."""
    cache_file = tmp_path / "embeddings" / "cifar100_resnet18.pt"
    backbone = _Deterministic512Backbone()

    def factory(_root: Path, is_train: bool, _transform: Any) -> Dataset[Any]:
        return _SyntheticCIFAR100Dataset(samples_per_class=10 if is_train else 2, seed=42)

    bundle = extract_cifar100_embeddings(
        data_dir=tmp_path / "data",
        cache_path=cache_file,
        batch_size=128,
        val_fraction=0.2,
        seed=42,
        device="cpu",
        overwrite=False,
        dataset_factory=factory,
        backbone=backbone,
    )

    assert cache_file.is_file()
    assert not cache_file.with_suffix(".pt.tmp").exists()
    assert backbone.forward_calls > 0

    assert bundle["train"]["embeddings"].shape == (800, 512)
    assert bundle["val"]["embeddings"].shape == (200, 512)
    assert bundle["test"]["embeddings"].shape == (200, 512)

    for split_name in ("train", "val", "test"):
        split = bundle[split_name]
        assert split["embeddings"].dtype == torch.float32
        assert split["fine_labels"].dtype == torch.int64
        assert split["coarse_labels"].dtype == torch.int64
        assert split["coarse_labels"].min().item() == 0
        assert split["coarse_labels"].max().item() == 19


def test_extract_cifar100_embeddings_reuses_valid_cache(tmp_path: Path) -> None:
    """Verify DATA-02 AC 3: when cache exists and overwrite=False, reuses cache without running backbone."""
    cache_file = tmp_path / "embeddings" / "cifar100_resnet18.pt"
    backbone = _Deterministic512Backbone()

    def factory(_root: Path, is_train: bool, _transform: Any) -> Dataset[Any]:
        return _SyntheticCIFAR100Dataset(samples_per_class=5 if is_train else 2, seed=42)

    first_bundle = extract_cifar100_embeddings(
        data_dir=tmp_path / "data",
        cache_path=cache_file,
        batch_size=100,
        val_fraction=0.2,
        seed=42,
        overwrite=False,
        dataset_factory=factory,
        backbone=backbone,
    )
    calls_after_first = backbone.forward_calls
    assert calls_after_first > 0

    def exploding_factory(_root: Path, _is_train: bool, _transform: Any) -> Dataset[Any]:
        raise AssertionError("dataset_factory should not be called on cache hit")

    second_bundle = extract_cifar100_embeddings(
        data_dir=tmp_path / "data",
        cache_path=cache_file,
        batch_size=100,
        val_fraction=0.2,
        seed=42,
        overwrite=False,
        dataset_factory=exploding_factory,
        backbone=backbone,
    )

    assert backbone.forward_calls == calls_after_first
    assert torch.equal(first_bundle["train"]["embeddings"], second_bundle["train"]["embeddings"])
    assert torch.equal(first_bundle["val"]["embeddings"], second_bundle["val"]["embeddings"])
    assert torch.equal(first_bundle["test"]["embeddings"], second_bundle["test"]["embeddings"])


def test_load_embedding_cache_corrupted_raises(tmp_path: Path) -> None:
    """Verify DATA-02 AC 4: corrupted file, missing split keys, wrong dim, or NaN/Inf raises ValueError."""
    missing_path = tmp_path / "nonexistent.pt"
    with pytest.raises(ValueError, match="does not exist"):
        load_embedding_cache(missing_path)

    corrupt_bytes_path = tmp_path / "corrupt.pt"
    corrupt_bytes_path.write_bytes(b"not a valid pytorch archive")
    with pytest.raises(ValueError, match="Failed to load"):
        load_embedding_cache(corrupt_bytes_path)

    missing_split_path = tmp_path / "missing_split.pt"
    torch.save({"train": {}, "val": {}}, missing_split_path)
    with pytest.raises(ValueError, match="missing required split 'test'"):
        load_embedding_cache(missing_split_path)

    # Wrong feature dimension (256 != 512)
    wrong_dim_path = tmp_path / "wrong_dim.pt"
    bad_split = {
        "embeddings": torch.zeros((10, 256), dtype=torch.float32),
        "fine_labels": torch.zeros(10, dtype=torch.int64),
        "coarse_labels": torch.zeros(10, dtype=torch.int64),
        "indices": torch.arange(10, dtype=torch.int64),
    }
    torch.save({"train": bad_split, "val": bad_split, "test": bad_split}, wrong_dim_path)
    with pytest.raises(ValueError, match="512"):
        load_embedding_cache(wrong_dim_path)

    # NaN entries in embeddings
    nan_path = tmp_path / "nan_cache.pt"
    nan_emb = torch.zeros((10, 512), dtype=torch.float32)
    nan_emb[0, 0] = float("nan")
    nan_split = {
        "embeddings": nan_emb,
        "fine_labels": torch.zeros(10, dtype=torch.int64),
        "coarse_labels": torch.full((10,), 4, dtype=torch.int64),
        "indices": torch.arange(10, dtype=torch.int64),
    }
    torch.save({"train": nan_split, "val": nan_split, "test": nan_split}, nan_path)
    with pytest.raises(ValueError, match="NaN or Inf"):
        load_embedding_cache(nan_path)
