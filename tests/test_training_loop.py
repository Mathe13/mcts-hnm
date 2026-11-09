"""Tests for downstream models, baseline samplers, training loops, and benchmark suite."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.mlp_head import Tier1MLPHead, create_mlp_optimizer
from src.training.baselines import OHEMSampler, UniformSampler


def test_uniform_sampler_batch_generation() -> None:
    neg_pool = np.arange(100, dtype=np.int64)
    pos_pool = np.arange(10, dtype=np.int64)

    # Scenario 1 (pos + neg)
    sampler = UniformSampler(negative_indices=neg_pool, neg_batch_size=24, positive_indices=pos_pool, pos_batch_size=8, seed=42)
    pos_idx, neg_idx, all_idx = sampler.sample_batch()
    assert len(pos_idx) == 8
    assert len(neg_idx) == 24
    assert len(all_idx) == 32
    assert np.all(np.isin(pos_idx, pos_pool))
    assert np.all(np.isin(neg_idx, neg_pool))

    # Scenario 2 (neg only across all classes)
    sampler_s2 = UniformSampler(negative_indices=neg_pool, neg_batch_size=48, pos_batch_size=0, seed=42)
    p2, n2, a2 = sampler_s2.sample_batch()
    assert len(p2) == 0
    assert len(n2) == 48
    assert len(a2) == 48


def test_ohem_sampler_selects_highest_loss() -> None:
    # 50 negative samples, input dim 4
    torch.manual_seed(42)
    num_samples = 50
    features = torch.randn(num_samples, 4)
    neg_pool = np.arange(num_samples, dtype=np.int64)

    # Linear model: logit = x[:, 0]
    # For binary negatives, loss = BCE(z, 0). Larger logit z -> higher loss!
    model = Tier1MLPHead(input_dim=4, hidden_dim=8, num_outputs=1, squeeze_binary=True)
    # Mock model weights to isolate 1st feature
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        # Set first hidden neuron weight to 10.0 and output to 1.0
        model.fc1.weight[0, 0] = 10.0
        model.fc2.weight[0, 0] = 1.0

    # Put high positive value in sample index 7 so it has highest loss
    features[7, 0] = 5.0
    features[12, 0] = 4.0

    sampler = OHEMSampler(
        negative_indices=neg_pool,
        neg_batch_size=2,
        ohem_ratio=25,  # evaluate all 50
        seed=42,
    )
    _, neg_idx, _ = sampler.sample_batch(model, features, is_binary=True)
    assert len(neg_idx) == 2
    assert 7 in neg_idx
    assert 12 in neg_idx


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
    with pytest.raises(ValueError, match="neg_batch_size"):
        UniformSampler(negative_indices=np.array([1, 2]), neg_batch_size=0)
    with pytest.raises(ValueError, match="positive_indices"):
        UniformSampler(negative_indices=np.array([1, 2]), pos_batch_size=2)
    with pytest.raises(ValueError, match="ohem_ratio"):
        OHEMSampler(negative_indices=np.array([1, 2]), ohem_ratio=0)
