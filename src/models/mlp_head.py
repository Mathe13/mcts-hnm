"""Downstream MLP classification heads and optimizer factories for Tier-1 evaluation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    val = int(value)
    if val <= 0:
        raise ValueError(f"{name} must be > 0, got {val}.")
    return val


def _validate_float_range(value: float, name: str, low: float = 0.0, high: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a float, got {value!r}.")
    val = float(value)
    if not np.isfinite(val) or val < low or val >= high:
        raise ValueError(f"{name} must be in [{low}, {high}), got {val}.")
    return val


class Tier1MLPHead(nn.Module):
    r"""2-layer MLP classification head for frozen Tier-1 feature representations.

    Architecture:
        $$\text{Linear}(d_{\text{in}}, d_{\text{hid}}) \rightarrow \text{ReLU}() \rightarrow \text{Dropout}(p) \rightarrow \text{Linear}(d_{\text{hid}}, K)$$

    Matches GiantHunter's supervision setup and Section 5 of ARCHITECTURE.md.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        num_outputs: int = 1,
        dropout: float = 0.1,
        squeeze_binary: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = _validate_positive_int(input_dim, "input_dim")
        self.hidden_dim = _validate_positive_int(hidden_dim, "hidden_dim")
        self.num_outputs = _validate_positive_int(num_outputs, "num_outputs")
        self.dropout_rate = _validate_float_range(dropout, "dropout", 0.0, 1.0)
        if not isinstance(squeeze_binary, bool):
            raise ValueError(f"squeeze_binary must be a bool, got {squeeze_binary!r}.")
        self.squeeze_binary = squeeze_binary

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass over feature batch.

        Args:
            x: 2D Tensor of shape `[B, input_dim]`.

        Returns:
            Logits Tensor of shape `[B, num_outputs]` or `[B]` (if `num_outputs == 1` and `squeeze_binary`).
        """
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor input, got {type(x).__name__}.")
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected 2D tensor of shape [B, {self.input_dim}], got {tuple(x.shape)}."
            )

        h = self.fc1(x)
        h = self.act(h)
        h = self.dropout(h)
        out = self.fc2(h)

        if self.num_outputs == 1 and self.squeeze_binary:
            return out.squeeze(-1)
        return out


def create_mlp_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> torch.optim.Adam:
    """Instantiate standard Adam optimizer matching GiantHunter setup."""
    if not isinstance(model, nn.Module):
        raise ValueError(f"Expected nn.Module, got {type(model).__name__}.")
    lr_val = float(lr)
    if not np.isfinite(lr_val) or lr_val <= 0.0:
        raise ValueError(f"lr must be > 0.0, got {lr_val}.")
    wd_val = float(weight_decay)
    if not np.isfinite(wd_val) or wd_val < 0.0:
        raise ValueError(f"weight_decay must be >= 0.0, got {wd_val}.")

    return torch.optim.Adam(model.parameters(), lr=lr_val, weight_decay=wd_val)
