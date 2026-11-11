"""Diagnostic Pilot Runner for Scenarios 1 and 2 (Phase B4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.splits import build_nested_imbalanced_splits
from src.training.trainer import train_scenario1, train_scenario2
from src.trees.semantic_tree import build_semantic_tree
from src.utils.experiment_config import assert_path_mutable


def run_diagnostic_pilot(
    output_dir: Path | str,
    seeds: tuple[int, ...] = (42, 43),
    num_steps: int = 10,
    eval_every: int = 5,
    target_class: int = 0,
    ratios: tuple[int, ...] = (50, 100, 500),
    num_positives: int = 80,
    execution_mode: str = "smoke",
    features: torch.Tensor | None = None,
    labels: np.ndarray | None = None,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    r"""Execute short diagnostic pilot runs across Scenarios 1 and 2.

    Runs:
    - Scenario 1 (One-vs-Rest): ratios 50, 100, 500 with Uniform and MCTS Enhanced across `seeds`.
    - Scenario 2 (100-way): Uniform and MCTS Enhanced across `seeds`.
    """
    out_path = Path(output_dir)
    assert_path_mutable(out_path / "pilot_run.json")
    out_path.mkdir(parents=True, exist_ok=True)

    num_classes = 100
    samples_per_class = 450

    if features is None or labels is None:
        # Fast synthetic fixture: 100 classes with 450 samples each, small feature dimension
        dim = 16
        total_samples = num_classes * samples_per_class
        feat_tensor = torch.randn(total_samples, dim)
        label_arr = np.repeat(np.arange(num_classes, dtype=np.int64), samples_per_class)
    else:
        feat_tensor = features
        label_arr = labels

    val_dim = 16
    val_samples_per_class = 5
    val_feat = torch.randn(num_classes * val_samples_per_class, val_dim)
    val_lbl = np.repeat(np.arange(num_classes, dtype=np.int64), val_samples_per_class)

    test_feat = torch.randn(num_classes * 5, val_dim)
    test_lbl = np.repeat(np.arange(num_classes, dtype=np.int64), 5)

    all_results: list[dict[str, Any]] = []

    # 1. Scenario 1 pilot runs
    for seed in seeds:
        splits = build_nested_imbalanced_splits(
            fine_labels=label_arr,
            target_class=target_class,
            target_positive_count=num_positives,
            ratios=ratios,
            seed=seed,
        )

        for ratio in ratios:
            split = splits[ratio]
            for strat in ("uniform", "mcts_enhanced"):
                run_id = f"pilot_scen1_cls{target_class}_r{ratio}_{strat}_s{seed}"
                tree = build_semantic_tree(split.negative_class_to_indices) if strat == "mcts_enhanced" else None

                val_binary = (val_lbl == target_class).astype(np.int64)
                test_binary = (test_lbl == target_class).astype(np.int64)

                res = train_scenario1(
                    train_features=feat_tensor,
                    train_binary_labels=split.binary_labels,
                    positive_indices=split.positive_indices,
                    negative_indices=split.negative_indices,
                    val_features=val_feat,
                    val_binary_labels=val_binary,
                    test_features=test_feat,
                    test_binary_labels=test_binary,
                    strategy=strat,
                    tree=tree,
                    num_steps=num_steps,
                    eval_every=eval_every,
                    seed=seed,
                    device=device,
                )

                record: dict[str, Any] = {
                    "run_id": run_id,
                    "scenario": "scenario1",
                    "target_class": int(target_class),
                    "imbalance_ratio": int(ratio),
                    "strategy": strat,
                    "seed": int(seed),
                    "num_positives": len(split.positive_indices),
                    "num_negatives": len(split.negative_indices),
                    "step_history": res.get("step_history", []),
                    "exposure_diagnostics": res.get("exposure_diagnostics", {}),
                    "final_metrics": res.get("final_metrics", {}),
                    "val_metrics": res.get("val_metrics", {}),
                }

                rec_file = out_path / f"{run_id}.json"
                with open(rec_file, "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2, default=str)

                all_results.append(record)

    # 2. Scenario 2 pilot runs
    scen2_class_to_indices = {
        c: np.flatnonzero(label_arr == c).astype(np.int64)
        for c in range(num_classes)
    }
    for seed in seeds:
        for strat in ("uniform", "mcts_enhanced"):
            run_id = f"pilot_scen2_cls100_{strat}_s{seed}"
            tree = build_semantic_tree(scen2_class_to_indices) if strat == "mcts_enhanced" else None

            res = train_scenario2(
                train_features=feat_tensor,
                train_labels=label_arr,
                val_features=val_feat,
                val_labels=val_lbl,
                test_features=test_feat,
                test_labels=test_lbl,
                strategy=strat,
                tree=tree,
                num_classes=num_classes,
                num_steps=num_steps,
                eval_every=eval_every,
                seed=seed,
                device=device,
            )

            record = {
                "run_id": run_id,
                "scenario": "scenario2",
                "target_class": None,
                "imbalance_ratio": None,
                "strategy": strat,
                "seed": int(seed),
                "num_positives": None,
                "num_negatives": None,
                "step_history": res.get("step_history", []),
                "exposure_diagnostics": res.get("exposure_diagnostics", {}),
                "final_metrics": res.get("final_metrics", {}),
                "val_metrics": res.get("val_metrics", {}),
            }

            rec_file = out_path / f"{run_id}.json"
            with open(rec_file, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, default=str)

            all_results.append(record)

    return all_results
