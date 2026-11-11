"""Deterministic Checkpoint Manager for Last and Best-Validation States (Phase B3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.trees.base_tree import FastHierarchyTree


@dataclass
class CheckpointManager:
    """Manages saving and loading of model, optimizer, tree, and RNG state checkpoints."""

    checkpoint_dir: Path
    run_id: str

    def __post_init__(self) -> None:
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(self, checkpoint_type: str) -> Path:
        """Get the filesystem path for a checkpoint of given type ('last' or 'best_validation')."""
        return self.checkpoint_dir / f"{self.run_id}_{checkpoint_type}.pt"

    def save_checkpoint(
        self,
        checkpoint_type: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        metrics: dict[str, Any],
        tree: FastHierarchyTree | None = None,
        seed_states: dict[str, Any] | None = None,
    ) -> Path:
        """Save a complete state checkpoint for resumption and evaluation."""
        if checkpoint_type not in ("last", "best_validation"):
            raise ValueError(
                f"checkpoint_type must be 'last' or 'best_validation', got '{checkpoint_type}'."
            )

        path = self.get_checkpoint_path(checkpoint_type)
        tree_state = None
        if tree is not None:
            tree_state = {
                "visits": tree.visits.copy(),
                "q_values": tree.q_values.copy(),
            }

        rng_state = seed_states if seed_states is not None else {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
        }

        checkpoint = {
            "run_id": self.run_id,
            "step": step,
            "checkpoint_type": checkpoint_type,
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "tree_state": tree_state,
            "seed_states": rng_state,
        }

        torch.save(checkpoint, path)
        return path

    @staticmethod
    def load_checkpoint(
        path: Path | str,
        model: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        tree: FastHierarchyTree | None = None,
        restore_rng: bool = False,
    ) -> dict[str, Any]:
        """Load and restore state from a saved checkpoint file."""
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if model is not None and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if tree is not None and checkpoint.get("tree_state") is not None:
            t_state = checkpoint["tree_state"]
            tree.visits[:] = t_state["visits"]
            tree.q_values[:] = t_state["q_values"]

        if restore_rng and "seed_states" in checkpoint:
            s_states = checkpoint["seed_states"]
            if "torch" in s_states:
                torch.set_rng_state(s_states["torch"])
            if "numpy" in s_states:
                np.random.set_state(s_states["numpy"])

        return checkpoint
