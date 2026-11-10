"""Raw CIFAR-100 32x32 image dataset loading with execution mode contract."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torchvision import datasets, transforms

from src.utils.experiment_config import ExecutionMode, ExperimentConfig


def load_cifar100_raw_images(
    data_dir: Path | str = "data",
    split: Literal["train", "test", "val"] = "train",
    download: bool = True,
    config: ExperimentConfig | None = None,
    num_smoke_samples: int = 100,
) -> tuple[torch.Tensor, np.ndarray]:
    """Load raw 32x32 CIFAR-100 images and label arrays adhering to ExecutionMode contract.

    Args:
        data_dir: Directory containing or destination for CIFAR-100.
        split: One of "train", "test", or "val".
        download: Whether to allow torchvision to download if missing.
        config: Centralized experiment config enforcing REAL vs SMOKE mode.
        num_smoke_samples: Number of synthetic instances in SMOKE mode.

    Returns:
        Tuple of `(images_tensor, labels_array)` where images have shape `[N, 3, 32, 32]`.

    Raises:
        FileNotFoundError: If in `ExecutionMode.REAL` and dataset is not present or cannot be downloaded.
        ValueError: If split is invalid.
    """
    if split not in ("train", "test", "val"):
        raise ValueError(f"split must be 'train', 'test', or 'val', got '{split}'.")

    active_config = config if config is not None else ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    path = Path(data_dir)

    if active_config.execution_mode == ExecutionMode.REAL:
        cifar_dir = path / "cifar-100-python"
        if not download and not cifar_dir.is_dir():
            raise FileNotFoundError(
                f"Real CIFAR-100 dataset not found at {path}. Enable download=True or download beforehand."
            )

        try:
            is_train = (split in ("train", "val"))
            ds = datasets.CIFAR100(
                root=str(path),
                train=is_train,
                download=download,
            )
            # Convert [N, 32, 32, 3] uint8 to [N, 3, 32, 32] float32 in [0, 1]
            raw_data = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float() / 255.0
            raw_targets = np.array(ds.targets, dtype=np.int64)
            return raw_data, raw_targets
        except Exception as exc:
            raise FileNotFoundError(
                f"Real CIFAR-100 dataset not found or failed to load from {path}: {exc}"
            ) from exc

    # ExecutionMode.SMOKE: Return deterministic synthetic fixture
    seed = active_config.seed
    rng = np.random.default_rng(seed)
    synthetic_imgs = rng.normal(size=(num_smoke_samples, 3, 32, 32)).astype(np.float32)
    synthetic_labels = np.array([i % 100 for i in range(num_smoke_samples)], dtype=np.int64)

    return torch.from_numpy(synthetic_imgs), synthetic_labels
