"""Tests for downstream models, baseline samplers, training loops, and benchmark suite."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.mlp_head import Tier1MLPHead, create_mlp_optimizer


def test_tier1_mlp_head_forward_and_optimizer() -> None:
    # Binary head with squeeze
    model_bin = Tier1MLPHead(input_dim=512, hidden_dim=128, num_outputs=1, dropout=0.1, squeeze_binary=True)
    x = torch.randn(16, 512, requires_grad=True)
    out_bin = model_bin(x)
    assert out_bin.shape == (16,)
    loss_bin = out_bin.sum()
    loss_bin.backward()
    assert x.grad is not None

    # Binary head without squeeze
    model_bin_nosq = Tier1MLPHead(input_dim=512, hidden_dim=128, num_outputs=1, squeeze_binary=False)
    out_nosq = model_bin_nosq(torch.randn(8, 512))
    assert out_nosq.shape == (8, 1)

    # Multi-class head
    model_multi = Tier1MLPHead(input_dim=512, hidden_dim=64, num_outputs=100)
    out_multi = model_multi(torch.randn(10, 512))
    assert out_multi.shape == (10, 100)

    # Optimizer creation
    opt = create_mlp_optimizer(model_multi, lr=1e-3, weight_decay=1e-4)
    assert isinstance(opt, torch.optim.Adam)
    assert opt.defaults["lr"] == 1e-3
    assert opt.defaults["weight_decay"] == 1e-4


def test_model_and_sampler_invalid_args_raise() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        Tier1MLPHead(input_dim=0)
    with pytest.raises(ValueError, match="hidden_dim"):
        Tier1MLPHead(hidden_dim=-5)
    with pytest.raises(ValueError, match="num_outputs"):
        Tier1MLPHead(num_outputs=0)
    with pytest.raises(ValueError, match="dropout"):
        Tier1MLPHead(dropout=1.0)
    with pytest.raises(ValueError, match="dropout"):
        Tier1MLPHead(dropout=-0.1)
    with pytest.raises(ValueError, match="Expected 2D tensor"):
        m = Tier1MLPHead(input_dim=512)
        m(torch.randn(16, 256))
    with pytest.raises(ValueError, match="lr"):
        create_mlp_optimizer(Tier1MLPHead(), lr=-1e-3)
    with pytest.raises(ValueError, match="weight_decay"):
        create_mlp_optimizer(Tier1MLPHead(), weight_decay=-1.0)
