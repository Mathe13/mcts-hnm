"""Non-hierarchical baseline sampling strategies: Uniform and Online Hard Example Mining (OHEM)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.seed import set_global_seed


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    val = int(value)
    if val <= 0:
        raise ValueError(f"{name} must be > 0, got {val}.")
    return val


def _validate_non_negative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    val = int(value)
    if val < 0:
        raise ValueError(f"{name} must be >= 0, got {val}.")
    return val


class UniformSampler:
    """Samples mini-batches uniformly at random from available pools."""

    def __init__(
        self,
        negative_indices: np.ndarray,
        neg_batch_size: int = 48,
        positive_indices: np.ndarray | None = None,
        pos_batch_size: int = 0,
        seed: int = 42,
    ) -> None:
        self.neg_batch_size = _validate_positive_int(neg_batch_size, "neg_batch_size")
        self.pos_batch_size = _validate_non_negative_int(pos_batch_size, "pos_batch_size")

        neg_arr = np.asarray(negative_indices, dtype=np.int64)
        if neg_arr.ndim != 1 or neg_arr.size == 0:
            raise ValueError("negative_indices must be a non-empty 1D integer array.")
        self.negative_pool = np.ascontiguousarray(neg_arr)

        if self.pos_batch_size > 0:
            if positive_indices is None:
                raise ValueError("positive_indices is required when pos_batch_size > 0.")
            pos_arr = np.asarray(positive_indices, dtype=np.int64)
            if pos_arr.ndim != 1 or pos_arr.size == 0:
                raise ValueError("positive_indices must be a non-empty 1D integer array.")
            self.positive_pool = np.ascontiguousarray(pos_arr)
        else:
            self.positive_pool = np.empty(0, dtype=np.int64)

        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def sample_batch(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample mini-batch indices.

        Returns:
            Tuple `(positive_indices, negative_indices, all_indices)`.
        """
        replace_neg = self.negative_pool.size < self.neg_batch_size
        neg_idx = self._rng.choice(
            self.negative_pool, size=self.neg_batch_size, replace=replace_neg
        ).astype(np.int64, copy=False)

        if self.pos_batch_size > 0:
            replace_pos = self.positive_pool.size < self.pos_batch_size
            pos_idx = self._rng.choice(
                self.positive_pool, size=self.pos_batch_size, replace=replace_pos
            ).astype(np.int64, copy=False)
        else:
            pos_idx = np.empty(0, dtype=np.int64)

        all_idx = np.concatenate((pos_idx, neg_idx)).astype(np.int64, copy=False)
        return pos_idx, neg_idx, all_idx


class OHEMSampler:
    r"""Online Hard Example Mining (OHEM) sampler.

    Draws a candidate negative pool of size $M_{\text{cand}} = \min(|\mathcal{D}_{\text{neg}}|, r_{\text{ohem}} \cdot B_{\text{neg}})$,
    evaluates model loss under `torch.no_grad()`, and selects the top-$B_{\text{neg}}$ samples with highest loss.
    """

    def __init__(
        self,
        negative_indices: np.ndarray,
        neg_batch_size: int = 48,
        ohem_ratio: int = 4,
        positive_indices: np.ndarray | None = None,
        pos_batch_size: int = 0,
        seed: int = 42,
    ) -> None:
        self.neg_batch_size = _validate_positive_int(neg_batch_size, "neg_batch_size")
        self.pos_batch_size = _validate_non_negative_int(pos_batch_size, "pos_batch_size")
        self.ohem_ratio = _validate_positive_int(ohem_ratio, "ohem_ratio")

        neg_arr = np.asarray(negative_indices, dtype=np.int64)
        if neg_arr.ndim != 1 or neg_arr.size == 0:
            raise ValueError("negative_indices must be a non-empty 1D integer array.")
        self.negative_pool = np.ascontiguousarray(neg_arr)

        if self.pos_batch_size > 0:
            if positive_indices is None:
                raise ValueError("positive_indices is required when pos_batch_size > 0.")
            pos_arr = np.asarray(positive_indices, dtype=np.int64)
            if pos_arr.ndim != 1 or pos_arr.size == 0:
                raise ValueError("positive_indices must be a non-empty 1D integer array.")
            self.positive_pool = np.ascontiguousarray(pos_arr)
        else:
            self.positive_pool = np.empty(0, dtype=np.int64)

        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def sample_batch(
        self,
        model: nn.Module,
        features: torch.Tensor,
        device: torch.device | None = None,
        is_binary: bool = True,
        target_labels: torch.Tensor | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mine hardest negative samples using current model predictions.

        Returns:
            Tuple `(positive_indices, negative_indices, all_indices)`.
        """
        if not isinstance(model, nn.Module):
            raise ValueError("model must be an nn.Module instance.")
        if not isinstance(features, torch.Tensor):
            raise ValueError("features must be a torch.Tensor.")

        cand_size = min(self.negative_pool.size, self.ohem_ratio * self.neg_batch_size)
        replace_cand = self.negative_pool.size < cand_size
        cand_indices = self._rng.choice(
            self.negative_pool, size=cand_size, replace=replace_cand
        ).astype(np.int64, copy=False)

        dev = device if device is not None else next(model.parameters()).device
        cand_features = features[cand_indices].to(dev)

        model_training_mode = model.training
        model.eval()
        with torch.no_grad():
            logits = model(cand_features)
            if is_binary:
                if logits.ndim == 2 and logits.shape[1] == 1:
                    logits_1d = logits.squeeze(1)
                else:
                    logits_1d = logits
                # Negative targets are 0. Loss = -log(1 - sigma(z))
                targets = torch.zeros_like(logits_1d)
                sample_losses = F.binary_cross_entropy_with_logits(
                    logits_1d, targets, reduction="none"
                ).cpu().numpy()
            else:
                if target_labels is None:
                    raise ValueError("target_labels is required when is_binary=False.")
                cand_targets = torch.as_tensor(target_labels, dtype=torch.long, device=dev)[cand_indices]
                sample_losses = F.cross_entropy(logits, cand_targets, reduction="none").cpu().numpy()

        if model_training_mode:
            model.train()

        # Select top neg_batch_size candidate samples
        if cand_size <= self.neg_batch_size:
            if cand_size < self.neg_batch_size:
                # Need replacement
                choice_pos = self._rng.choice(cand_size, size=self.neg_batch_size, replace=True)
                neg_idx = cand_indices[choice_pos]
            else:
                neg_idx = cand_indices
        else:
            top_positions = np.argpartition(-sample_losses, self.neg_batch_size - 1)[: self.neg_batch_size]
            neg_idx = cand_indices[top_positions]

        if self.pos_batch_size > 0:
            replace_pos = self.positive_pool.size < self.pos_batch_size
            pos_idx = self._rng.choice(
                self.positive_pool, size=self.pos_batch_size, replace=replace_pos
            ).astype(np.int64, copy=False)
        else:
            pos_idx = np.empty(0, dtype=np.int64)

        all_idx = np.concatenate((pos_idx, neg_idx)).astype(np.int64, copy=False)
        return pos_idx, neg_idx, all_idx
