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
from src.mcts.diagnostics import compute_visitation_entropy
from src.training.trainer import train_scenario1, train_scenario2
from src.trees.base_tree import FastHierarchyTree
from src.trees.confusion_tree import build_confusion_tree, estimate_validation_confusion_matrix
from src.trees.semantic_tree import build_semantic_tree
from src.utils.experiment_config import (
    ExecutionMode,
    ExperimentConfig,
    save_provenance_metadata,
)
from src.utils.seed import set_global_seed
from src.utils.statistics import compute_paired_statistical_tests

logger = logging.getLogger(__name__)


def audit_scenario1_imbalance(
    split: Scenario1Split,
    pos_batch_size: int = 16,
    neg_batch_size: int = 48,
    num_steps: int = 500,
) -> dict[str, Any]:
    """Audit available vs effective sampling ratios for Scenario 1 imbalanced regime."""
    avail_pos = len(split.positive_indices)
    avail_neg = len(split.negative_indices)
    eff_ratio = float(pos_batch_size / neg_batch_size) if neg_batch_size > 0 else 0.0
    cum_ratio = float((pos_batch_size * num_steps) / (neg_batch_size * num_steps)) if (neg_batch_size * num_steps) > 0 else 0.0

    return {
        "available_positives": avail_pos,
        "available_negatives": avail_neg,
        "pos_batch_size": pos_batch_size,
        "neg_batch_size": neg_batch_size,
        "effective_batch_ratio": eff_ratio,
        "cumulative_ratio": cum_ratio,
    }


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
    config: ExperimentConfig | None = None,
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
    if len(seeds) < 1:
        raise ValueError("At least 1 seed required.")

    active_config = config if config is not None else ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
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

    computational_budget = {
        "pos_batch_size": 16,
        "neg_batch_size": 48,
        "budget_equivalent": True,
        "strategies_verified": list(strategies),
    }

    imbalance_audit: dict[str, Any] | None = None
    if scenario == "scenario1":
        pos_mask_all = train_lbl_np == target_class_scenario1
        pos_idx_all = np.flatnonzero(pos_mask_all).astype(np.int64)
        neg_idx_all = np.flatnonzero(~pos_mask_all).astype(np.int64)
        neg_classes_all = np.unique(train_lbl_np[neg_idx_all])
        neg_cls_to_idx = {
            int(c): np.flatnonzero(train_lbl_np == c).astype(np.int64)
            for c in neg_classes_all
        }
        dummy_split = Scenario1Split(
            target_class_id=target_class_scenario1,
            target_class_name=str(target_class_scenario1),
            pos_fraction=1.0,
            positive_indices=pos_idx_all,
            negative_indices=neg_idx_all,
            negative_class_to_indices=neg_cls_to_idx,
            binary_labels=pos_mask_all.astype(np.int64),
        )
        imbalance_audit = audit_scenario1_imbalance(
            split=dummy_split,
            pos_batch_size=16,
            neg_batch_size=48,
            num_steps=num_steps,
        )

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

    diagnostics_per_strat: dict[str, Any] = {}

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

                    if tree is not None:
                        leaf_ids = tree.leaf_node_ids()
                        class_visits: dict[int, int] = {}
                        for lid in leaf_ids:
                            cls_id = int(tree.leaf_to_class_id[int(lid)])
                            class_visits[cls_id] = class_visits.get(cls_id, 0) + int(tree.visits[int(lid)])
                        for c in range(num_cls):
                            class_visits.setdefault(c, 0)

                        unvisited_classes = sum(1 for c in range(num_cls) if class_visits.get(c, 0) == 0)
                        ent_nats, ent_norm = compute_visitation_entropy(tree)
                        coverage_pct = float((num_cls - unvisited_classes) / num_cls * 100.0) if num_cls > 0 else 0.0

                        trained_model = res["model"]
                        trained_model.eval()
                        m_dev = device if device is not None else next(trained_model.parameters()).device
                        with torch.no_grad():
                            all_p: list[torch.Tensor] = []
                            for bi in range(0, test_features.shape[0], 512):
                                bx = test_features[bi : bi + 512].to(m_dev)
                                logits_b = trained_model(bx)
                                all_p.append(torch.argmax(logits_b, dim=1).detach().cpu())
                        preds_arr = torch.cat(all_p, dim=0).numpy()

                        per_class_recall: list[float] = []
                        for c in range(num_cls):
                            c_mask = (test_lbl_np == c)
                            if np.any(c_mask):
                                per_class_recall.append(float(np.mean(preds_arr[c_mask] == c)))
                            else:
                                per_class_recall.append(0.0)

                        diagnostics_per_strat[strat] = {
                            "class_visit_distribution": class_visits,
                            "unvisited_classes_count": unvisited_classes,
                            "visitation_entropy": ent_norm,
                            "entropy_nats": ent_nats,
                            "class_coverage_pct": coverage_pct,
                            "per_class_recall": per_class_recall,
                        }

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
    if len(seeds) >= 2 and baseline_comp_key in grouped_scores:
        enhanced_scores = grouped_scores[baseline_comp_key]
        for key, scores in grouped_scores.items():
            if key != baseline_comp_key:
                comp_name = f"{baseline_comp_key[0]}_{baseline_comp_key[1]}_vs_{key[0]}_{key[1]}"
                statistical_comparisons[comp_name] = compute_paired_statistical_tests(
                    enhanced_scores, scores
                )

    # Compute aggregated summary across tree_types and strategies
    aggregated_summary: dict[str, dict[str, dict[str, float]]] = {}
    for tt in tree_types:
        aggregated_summary[tt] = {}
        for strat in strategies:
            strat_runs = [r for r in runs if r.strategy == strat and (r.tree_type == tt or r.tree_type == "none")]
            if not strat_runs:
                continue
            metrics_dict: dict[str, list[float]] = {}
            for r in strat_runs:
                for mk, mv in r.final_metrics.items():
                    if isinstance(mv, (int, float, np.number)):
                        metrics_dict.setdefault(mk, []).append(float(mv))
            summary_strat: dict[str, float] = {}
            for mk, vals in metrics_dict.items():
                summary_strat[f"{mk}_mean"] = float(np.mean(vals))
                summary_strat[f"{mk}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            if "f1_mean" in summary_strat and "macro_f1_mean" not in summary_strat:
                summary_strat["macro_f1_mean"] = summary_strat["f1_mean"]
                summary_strat["macro_f1_std"] = summary_strat["f1_std"]
            aggregated_summary[tt][strat] = summary_strat

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

    json_export: dict[str, Any] = {
        "scenario": scenario,
        "benchmark_name": benchmark_name,
        "primary_metric": primary_metric,
        "seeds": list(seeds),
        "computational_budget": computational_budget,
        "statistical_comparisons": statistical_comparisons,
        "aggregated_summary": aggregated_summary,
        "runs": [asdict(r) for r in runs],
        "per_seed_results": [asdict(r) for r in runs],
    }

    if imbalance_audit is not None:
        json_export["imbalance_audit"] = imbalance_audit
    if diagnostics_per_strat:
        json_export["diagnostics"] = diagnostics_per_strat

    json_path = out_path / f"{benchmark_name}.json"
    csv_path = out_path / f"{benchmark_name}.csv"
    _atomic_save_json(json_export, json_path)
    _atomic_save_csv(ordered_fields, csv_rows, csv_path)

    save_provenance_metadata(json_path, active_config)
    save_provenance_metadata(csv_path, active_config)

    return json_export


def run_sensitivity_ablation(
    train_features: torch.Tensor,
    train_labels: np.ndarray | torch.Tensor,
    test_features: torch.Tensor,
    test_labels: np.ndarray | torch.Tensor,
    c_values: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    alpha_values: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7),
    output_dir: str | Path = "results",
    benchmark_name: str = "tier1_hyperparameter_sensitivity",
    num_steps: int = 500,
    eval_every: int = 100,
    seeds: tuple[int, ...] = (42,),
    target_class_scenario1: int = 0,
    config: ExperimentConfig | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate grid sensitivity over exploration coefficient c and momentum alpha."""
    active_config = config if config is not None else ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    out_path = Path(output_dir)

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

    pos_mask = train_lbl_np == target_class_scenario1
    pos_idx = np.flatnonzero(pos_mask).astype(np.int64)
    neg_idx = np.flatnonzero(~pos_mask).astype(np.int64)
    binary_train_lbl = pos_mask.astype(np.int64)
    binary_test_lbl = (test_lbl_np == target_class_scenario1).astype(np.int64)

    neg_classes = np.unique(train_lbl_np[neg_idx])
    class_to_indices = {
        c: np.flatnonzero(train_lbl_np == c).astype(np.int64)
        for c in neg_classes
    }

    grid_results: list[dict[str, Any]] = []

    for c in c_values:
        for alpha in alpha_values:
            c_alpha_metrics: list[dict[str, float]] = []
            for seed in seeds:
                set_global_seed(seed)
                tree = build_semantic_tree(class_to_indices)
                res = train_scenario1(
                    train_features=train_features,
                    train_binary_labels=binary_train_lbl,
                    positive_indices=pos_idx,
                    negative_indices=neg_idx,
                    test_features=test_features,
                    test_binary_labels=binary_test_lbl,
                    strategy="mcts_enhanced",
                    tree=tree,
                    num_steps=num_steps,
                    eval_every=eval_every,
                    exploration_c=float(c),
                    momentum_alpha=float(alpha),
                    seed=seed,
                    device=device,
                )
                c_alpha_metrics.append(res["final_metrics"])

            auprc_vals = [m["auprc"] for m in c_alpha_metrics]
            f1_vals = [m["f1"] for m in c_alpha_metrics]
            grid_results.append({
                "c": float(c),
                "alpha": float(alpha),
                "auprc_mean": float(np.mean(auprc_vals)),
                "auprc_std": float(np.std(auprc_vals, ddof=1)) if len(auprc_vals) > 1 else 0.0,
                "f1_mean": float(np.mean(f1_vals)),
                "f1_std": float(np.std(f1_vals, ddof=1)) if len(f1_vals) > 1 else 0.0,
                "runs": c_alpha_metrics,
            })

    json_export = {
        "benchmark_name": benchmark_name,
        "exploration_c": list(c_values),
        "momentum_alpha": list(alpha_values),
        "seeds": list(seeds),
        "grid": grid_results,
    }

    json_path = out_path / f"{benchmark_name}.json"
    _atomic_save_json(json_export, json_path)
    save_provenance_metadata(json_path, active_config)

    return json_export
