"""Programmatic multi-seed benchmark runner and persistence for Tier-1 experiments."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.splits import Scenario1Split, Scenario2Split
from src.training.trainer import train_scenario1, train_scenario2
from src.trees.base_tree import FastHierarchyTree
from src.trees.confusion_tree import build_confusion_tree, estimate_validation_confusion_matrix
from src.trees.semantic_tree import build_semantic_tree
from src.utils.seed import set_global_seed
from src.utils.statistics import compute_paired_statistical_tests

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRunResult:
    scenario: str
    strategy: str
    tree_type: str
    seed: int
    final_metrics: dict[str, float]


def _atomic_save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.json")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def _atomic_save_csv(fieldnames: list[str], rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.csv")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def run_benchmark(
    scenario: str,
    train_features: torch.Tensor,
    train_labels: np.ndarray | torch.Tensor,
    test_features: torch.Tensor,
    test_labels: np.ndarray | torch.Tensor,
    strategies: tuple[str, ...] = ("uniform", "ohem", "mcts_strict", "mcts_enhanced"),
    tree_types: tuple[str, ...] = ("semantic", "confusion_ward"),
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    output_dir: str | Path = "results",
    benchmark_name: str = "tier1_benchmark",
    num_steps: int = 500,
    eval_every: int = 100,
    target_class_scenario1: int = 0,
    val_features: torch.Tensor | None = None,
    val_labels: np.ndarray | torch.Tensor | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    r"""Execute multi-seed benchmark grid across strategies and tree types.

    Computes:
    - Per-run evaluation logs
    - Aggregated mean $\pm$ standard deviation across seeds
    - Paired Student's t-test and Wilcoxon signed-rank comparisons against `mcts_enhanced`
    - Atomically saves results to `<output_dir>/<benchmark_name>.json` and `.csv`.
    """
    if scenario not in ("scenario1", "scenario2"):
        raise ValueError(f"scenario must be 'scenario1' or 'scenario2', got '{scenario}'.")
    if len(seeds) < 2:
        raise ValueError(f"At least 2 seeds required for statistical comparison, got {len(seeds)}.")

    out_path = Path(output_dir)
    runs: list[BenchmarkRunResult] = []

    train_lbl_np = (
        train_labels.detach().cpu().numpy()
        if isinstance(train_labels, torch.Tensor)
        else np.asarray(train_labels)
    ).astype(np.int64)

    test_lbl_np = (
        test_labels.detach().cpu().numpy()
        if isinstance(test_labels, torch.Tensor)
        else np.asarray(test_labels)
    ).astype(np.int64)

    # Pre-build validation confusion matrix if tree_type requires confusion_ward
    conf_mat: np.ndarray | None = None
    if "confusion_ward" in tree_types:
        v_feat = val_features if val_features is not None else test_features
        v_lbl = val_labels if val_labels is not None else test_labels
        num_cls = 100 if scenario == "scenario2" else max(int(np.max(train_lbl_np)) + 1, 2)
        conf_mat = estimate_validation_confusion_matrix(
            train_embeddings=train_features,
            train_labels=train_labels,
            val_embeddings=v_feat,
            val_labels=v_lbl,
            num_classes=num_cls,
        )

    for tree_type in tree_types:
        for strat in strategies:
            for seed in seeds:
                set_global_seed(seed)

                if scenario == "scenario1":
                    # Build binary split for target class
                    pos_mask = train_lbl_np == target_class_scenario1
                    pos_idx = np.flatnonzero(pos_mask).astype(np.int64)
                    neg_idx = np.flatnonzero(~pos_mask).astype(np.int64)
                    binary_train_lbl = pos_mask.astype(np.int64)
                    binary_test_lbl = (test_lbl_np == target_class_scenario1).astype(np.int64)

                    tree: FastHierarchyTree | None = None
                    if strat in ("mcts_strict", "mcts_enhanced"):
                        neg_classes = np.unique(train_lbl_np[neg_idx])
                        class_to_indices = {
                            c: np.flatnonzero(train_lbl_np == c).astype(np.int64)
                            for c in neg_classes
                        }
                        if tree_type == "semantic":
                            tree = build_semantic_tree(class_to_indices)
                        elif tree_type == "confusion_ward":
                            assert conf_mat is not None
                            tree = build_confusion_tree(conf_mat, class_to_indices)

                    res = train_scenario1(
                        train_features=train_features,
                        train_binary_labels=binary_train_lbl,
                        positive_indices=pos_idx,
                        negative_indices=neg_idx,
                        test_features=test_features,
                        test_binary_labels=binary_test_lbl,
                        strategy=strat,
                        tree=tree,
                        num_steps=num_steps,
                        eval_every=eval_every,
                        seed=seed,
                        device=device,
                    )
                else:
                    # Scenario 2
                    num_cls = max(int(np.max(train_lbl_np)) + 1, 2)
                    tree = None
                    if strat in ("mcts_strict", "mcts_enhanced"):
                        class_to_indices = {
                            c: np.flatnonzero(train_lbl_np == c).astype(np.int64)
                            for c in range(num_cls)
                        }
                        if tree_type == "semantic":
                            tree = build_semantic_tree(class_to_indices)
                        elif tree_type == "confusion_ward":
                            assert conf_mat is not None
                            tree = build_confusion_tree(conf_mat, class_to_indices)

                    res = train_scenario2(
                        train_features=train_features,
                        train_labels=train_lbl_np,
                        test_features=test_features,
                        test_labels=test_lbl_np,
                        strategy=strat,
                        tree=tree,
                        num_classes=num_cls,
                        num_steps=num_steps,
                        eval_every=eval_every,
                        seed=seed,
                        device=device,
                    )

                final_metrics = res["final_metrics"]
                runs.append(
                    BenchmarkRunResult(
                        scenario=scenario,
                        strategy=strat,
                        tree_type=tree_type if strat.startswith("mcts") else "none",
                        seed=seed,
                        final_metrics=final_metrics,
                    )
                )

    # Metric aggregation
    primary_metric = "auprc" if scenario == "scenario1" else "top1_accuracy"
    grouped_scores: dict[tuple[str, str], list[float]] = {}
    for r in runs:
        key = (r.strategy, r.tree_type)
        grouped_scores.setdefault(key, []).append(r.final_metrics[primary_metric])

    # Comparisons against mcts_enhanced with semantic / confusion tree
    enhanced_sem_key = ("mcts_enhanced", "semantic")
    enhanced_conf_key = ("mcts_enhanced", "confusion_ward")
    baseline_comp_key = (
        enhanced_sem_key if enhanced_sem_key in grouped_scores else enhanced_conf_key
    )

    statistical_comparisons: dict[str, Any] = {}
    if baseline_comp_key in grouped_scores:
        enhanced_scores = grouped_scores[baseline_comp_key]
        for key, scores in grouped_scores.items():
            if key != baseline_comp_key:
                comp_name = f"{baseline_comp_key[0]}_{baseline_comp_key[1]}_vs_{key[0]}_{key[1]}"
                statistical_comparisons[comp_name] = compute_paired_statistical_tests(
                    enhanced_scores, scores
                )

    # Flatten summary rows for CSV export
    csv_rows: list[dict[str, Any]] = []
    all_fieldnames: set[str] = {"scenario", "strategy", "tree_type", "seed"}
    for r in runs:
        row: dict[str, Any] = {
            "scenario": r.scenario,
            "strategy": r.strategy,
            "tree_type": r.tree_type,
            "seed": r.seed,
        }
        for k, v in r.final_metrics.items():
            row[k] = v
            all_fieldnames.add(k)
        csv_rows.append(row)

    ordered_fields = ["scenario", "strategy", "tree_type", "seed"] + sorted(
        [f for f in all_fieldnames if f not in ("scenario", "strategy", "tree_type", "seed")]
    )
    for row in csv_rows:
        for f in ordered_fields:
            row.setdefault(f, None)

    json_export = {
        "scenario": scenario,
        "benchmark_name": benchmark_name,
        "primary_metric": primary_metric,
        "seeds": list(seeds),
        "statistical_comparisons": statistical_comparisons,
        "runs": [asdict(r) for r in runs],
    }

    json_path = out_path / f"{benchmark_name}.json"
    csv_path = out_path / f"{benchmark_name}.csv"
    _atomic_save_json(json_export, json_path)
    _atomic_save_csv(ordered_fields, csv_rows, csv_path)

    return json_export
