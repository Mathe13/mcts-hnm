"""Immutable resolved training configuration and parameter governance across runners."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class ResolvedTrainingConfig:
    """Immutable resolved hyperparameter configuration shared across runners.

    Enforces that benchmark runs, ablations, and direct training loops operate under
    identical resolved defaults and parameter names, fulfilling Roadmap v2 Phase A Item A4.
    """

    scenario: str
    target_class: int = 0
    imbalance_ratio: str | int = "1:50"
    target_positive_count: int = 80
    pos_batch_size: int = 16
    neg_batch_size: int = 48
    num_steps: int = 500
    eval_every: int = 100
    lr: float = 1e-3
    exploration_c: float = 2.0
    momentum_alpha: float = 0.3
    dual_stream_ratio: float = 0.5
    seed: int = 42
    strategy: str = "mcts_enhanced"
    tree_type: str = "semantic"
    device: str = "cpu"
    execution_mode: str = "real"

    def __post_init__(self) -> None:
        if self.scenario not in ("scenario1", "scenario2"):
            raise ValueError(f"scenario must be 'scenario1' or 'scenario2', got '{self.scenario}'.")
        if self.pos_batch_size < 0:
            raise ValueError(f"pos_batch_size must be >= 0, got {self.pos_batch_size}.")
        if self.neg_batch_size <= 0:
            raise ValueError(f"neg_batch_size must be > 0, got {self.neg_batch_size}.")
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, got {self.num_steps}.")
        if self.eval_every <= 0:
            raise ValueError(f"eval_every must be > 0, got {self.eval_every}.")
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0.0, got {self.lr}.")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize configuration to a canonical JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    def compute_config_hash(self) -> str:
        """Compute deterministic SHA-256 digest of resolved configuration."""
        canonical_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
