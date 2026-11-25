"""Tests for the Tier-2 GPU fine-tuning campaign (run on CPU with tiny synthetic data)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.training.tier2_gpu import (
    FineTunedResNet18Binary,
    GPUPreprocessor,
    Tier2CampaignConfig,
    Tier2Data,
    Tier2RunConfig,
    binary_metrics,
    extract_results_zip_from_notebook,
    holm_adjust,
    pooled_tier2_tests,
    write_tier2_latex_table,
    write_tier2_pooled_latex_table,
    load_tier2_records,
    random_crop_flip,
    run_tier2_campaign,
    summarize_tier2,
)


def _synthetic_data(per_class: int = 6) -> Tier2Data:
    g = torch.Generator().manual_seed(0)

    def imgs(n: int) -> torch.Tensor:
        return torch.randint(0, 256, (n, 3, 32, 32), generator=g, dtype=torch.uint8)

    train_labels = np.repeat(np.arange(100), per_class).astype(np.int64)
    small_labels = np.repeat(np.arange(100), 2).astype(np.int64)
    return Tier2Data(
        train_images=imgs(train_labels.size), train_labels=train_labels,
        val_images=imgs(small_labels.size), val_labels=small_labels,
        test_images=imgs(small_labels.size), test_labels=small_labels.copy(),
        split_seed=42, val_fraction=0.1,
    )


def _tiny_run_config(**kw: object) -> Tier2RunConfig:
    base = dict(
        num_steps=3, eval_every=2, pos_batch_size=2, neg_batch_size=6, image_size=32,
        pretrained=False, amp=False, eval_batch_size=64, cudnn_benchmark=False,
    )
    base.update(kw)
    return Tier2RunConfig(**base)  # type: ignore[arg-type]


def test_random_crop_flip_preserves_shape_and_is_seeded() -> None:
    x = torch.rand(5, 3, 32, 32)
    a = random_crop_flip(x, generator=torch.Generator().manual_seed(1))
    b = random_crop_flip(x, generator=torch.Generator().manual_seed(1))
    assert a.shape == x.shape
    assert torch.equal(a, b)


def test_preprocessor_upsamples_and_normalises() -> None:
    pre = GPUPreprocessor(image_size=64)
    out = pre(torch.full((2, 3, 32, 32), 255, dtype=torch.uint8))
    assert out.shape == (2, 3, 64, 64)
    expected = (1.0 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    assert torch.allclose(out[0, :, 0, 0], expected, atol=1e-5)


def test_frozen_backbone_has_no_trainable_params_and_stays_in_eval() -> None:
    m = FineTunedResNet18Binary(pretrained=False, freeze_backbone=True)
    m.train()
    assert not any(p.requires_grad for p in m.backbone.parameters())
    assert not m.backbone.training and m.head.training


def test_binary_metrics_perfect_ranking() -> None:
    probs = np.array([0.1, 0.2, 0.9, 0.8])
    y = np.array([0, 0, 1, 1])
    m = binary_metrics(probs, y, threshold=0.5)
    assert m["f1"] == pytest.approx(1.0)
    assert m["auprc"] == pytest.approx(1.0)
    assert m["fpr_at_95_recall"] == pytest.approx(0.0)


@pytest.mark.parametrize("tree_type", ["semantic", "random"])
def test_campaign_runs_all_strategies_and_resumes(tmp_path, tree_type: str) -> None:
    data = _synthetic_data()
    cfg = Tier2CampaignConfig(
        campaign="t", target_classes=(0,), ratios=("1:10",),
        strategies=("uniform", "ohem", "mcts_strict", "mcts_enhanced"),
        tree_type=tree_type, seeds=(42, 43), run=_tiny_run_config(), output_dir=str(tmp_path),
    )
    logs: list[str] = []
    recs = run_tier2_campaign(cfg, data, torch.device("cpu"), log=logs.append)
    assert len(recs) == 8
    files = sorted(tmp_path.glob("*.json"))
    assert len(files) == 8
    rec = json.loads(files[0].read_text(encoding="utf-8"))
    for key in ("final_metrics", "best_validation_metrics", "history", "computational_costs", "run_config"):
        assert key in rec
    assert rec["computational_costs"]["total_updates"] == 3
    ohem = [r for r in recs if r["strategy"] == "ohem"][0]
    assert ohem["computational_costs"]["mining_forwards"] == 3 * 4 * 6
    mcts = [r for r in recs if r["strategy"] == "mcts_strict"][0]
    assert mcts["mcts_diagnostics"]["total_leaf_visits"] == 3

    # Second invocation must skip every finished run.
    logs.clear()
    recs2 = run_tier2_campaign(cfg, data, torch.device("cpu"), log=logs.append)
    assert len(recs2) == 8
    assert all("skipping" in line for line in logs)

    summary, tests = summarize_tier2(load_tier2_records(tmp_path, campaign="t"))
    assert {r["strategy"] for r in summary} == set(cfg.strategies)
    assert all(r["n_seeds"] == 2 for r in summary)
    assert {t["comparison"] for t in tests} == {
        "uniform - mcts_strict", "ohem - mcts_strict", "mcts_enhanced - mcts_strict"
    }


def test_holm_adjust_matches_hand_computation() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def _fake_records() -> list[dict]:
    recs = []
    for tc, name in ((0, "apple"), (4, "beaver")):
        for seed in range(42, 47):
            for strat, bonus in (("uniform", 0.0), ("ohem", -0.01), ("mcts_strict", 0.05), ("mcts_enhanced", 0.02)):
                m = {"auprc": 0.5 + bonus + 0.001 * seed, "f1": 0.4 + bonus, "fpr_at_95_recall": 0.2}
                recs.append({"target_class": tc, "target_class_name": name, "imbalance_ratio": "1:50",
                             "strategy": strat, "seed": seed, "final_metrics": m,
                             "best_validation_metrics": dict(m)})
    return recs


def test_pooled_tests_and_latex_tables(tmp_path) -> None:
    recs = _fake_records()
    rows = pooled_tier2_tests(recs, metric="auprc")
    assert {r["comparison"] for r in rows} == {"uniform - mcts_strict", "ohem - mcts_strict",
                                               "mcts_enhanced - mcts_strict"}
    assert all(r["n_pairs"] == 10 and r["reference_wins"] == 10 and r["mean_diff"] < 0 for r in rows)
    t1 = write_tier2_latex_table(recs, tmp_path / "a.tex").read_text(encoding="utf-8")
    t2 = write_tier2_pooled_latex_table(recs, tmp_path / "b.tex").read_text(encoding="utf-8")
    assert "beaver" in t1 and t1.count(r"\textbf{") == 8
    assert "10/10" in t2 and r"\label{tab:tier2_pooled}" in t2


def test_results_zip_roundtrip_through_notebook_output(tmp_path, capsys) -> None:
    from src.training.tier2_gpu import print_results_zip

    src_dir = tmp_path / "res"
    src_dir.mkdir()
    (src_dir / "run.json").write_text('{"a": 1}', encoding="utf-8")
    print_results_zip(src_dir, tmp_path / "bundle")
    text = capsys.readouterr().out
    nb = {"cells": [{"cell_type": "code", "outputs": [{"output_type": "stream", "text": text.splitlines(True)}]}]}
    nb_path = tmp_path / "nb.ipynb"
    nb_path.write_text(json.dumps(nb), encoding="utf-8")
    names = extract_results_zip_from_notebook(nb_path, tmp_path / "out")
    assert names == ["run.json"]
    assert (tmp_path / "out" / "run.json").read_text(encoding="utf-8") == '{"a": 1}'
