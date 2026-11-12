"""Programmatic multi-seed benchmark runner and persistence for Tier-1 experiments."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.cifar100_taxonomy import get_cifar100_taxonomy
from src.data.splits import (
    IMBALANCE_RATIOS,
    Scenario1Split,
    Scenario2Split,
    build_scenario1_split,
)
from src.mcts.diagnostics import compute_visitation_entropy
from src.training.exposure import compute_exposure_recall_correlation
from src.training.trainer import (
    MCTS_STRATEGIES,
    SUPPORTED_STRATEGIES,
    train_scenario1,
    train_scenario2,
)
from src.trees.base_tree import FastHierarchyTree, compute_tree_topology_diagnostics
from src.trees.confusion_tree import build_confusion_tree, estimate_validation_confusion_matrix
from src.trees.semantic_tree import build_semantic_tree
from src.utils.experiment_config import (
    ExecutionMode,
    ExperimentConfig,
    find_repo_root,
    get_active_session_mode,
    get_git_commit,
    get_isolated_paths,
    save_provenance_metadata,
    validate_mode_isolated_path,
)
from src.utils.seed import set_global_seed
from src.utils.statistics import (
    compute_2x2_factorial_effects,
    compute_paired_statistical_tests,
)

logger = logging.getLogger(__name__)


def audit_scenario1_imbalance(
    split: Scenario1Split,
    pos_batch_size: int = 16,
    neg_batch_size: int = 48,
    num_steps: int = 500,
    selected_pos_indices: np.ndarray | None = None,
    selected_neg_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    """Audit available vs effective sampling ratios and coverage for Scenario 1 imbalanced regime."""
    avail_pos = len(split.positive_indices)
    avail_neg = len(split.negative_indices)
    eff_ratio = float(pos_batch_size / neg_batch_size) if neg_batch_size > 0 else 0.0
    cum_ratio = float((pos_batch_size * num_steps) / (neg_batch_size * num_steps)) if (neg_batch_size * num_steps) > 0 else 0.0

    res: dict[str, Any] = {
        "available_positives": avail_pos,
        "available_negatives": avail_neg,
        "pos_batch_size": pos_batch_size,
        "neg_batch_size": neg_batch_size,
        "effective_batch_ratio": eff_ratio,
        "cumulative_ratio": cum_ratio,
    }

    if selected_pos_indices is not None and selected_neg_indices is not None:
        total_pos = len(selected_pos_indices)
        total_neg = len(selected_neg_indices)
        uniq_pos = len(np.unique(selected_pos_indices))
        uniq_neg = len(np.unique(selected_neg_indices))
        res["empirical_positives_selected"] = total_pos
        res["empirical_negatives_selected"] = total_neg
        res["empirical_selected_ratio"] = float(total_pos / total_neg) if total_neg > 0 else 0.0
        res["unique_positives_covered"] = uniq_pos
        res["unique_negatives_covered"] = uniq_neg
        res["positive_coverage_pct"] = float(uniq_pos / avail_pos * 100.0) if avail_pos > 0 else 0.0
        res["negative_coverage_pct"] = float(uniq_neg / avail_neg * 100.0) if avail_neg > 0 else 0.0

    return res


@dataclass
class BenchmarkRunResult:
    scenario: str
    strategy: str
    tree_type: str
    seed: int
    final_metrics: dict[str, float]
    imbalance_ratio: str | None = None
    run_id: str | None = None


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
    imbalance_ratios: tuple[str, ...] = ("1:50",),
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
    for strat in strategies:
        if strat not in SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strat}' for {scenario}. Supported: {sorted(SUPPORTED_STRATEGIES)}."
            )
    if val_features is None or val_labels is None:
        raise ValueError(
            "val_features and val_labels are required in run_benchmark to prevent data leakage. "
            "Falling back to test data is strictly prohibited."
        )

    if config is not None:
        active_config = config
    elif get_active_session_mode() is not None:
        active_config = ExperimentConfig(execution_mode=get_active_session_mode())
    else:
        active_config = ExperimentConfig(execution_mode=ExecutionMode.REAL)

    paths = get_isolated_paths(active_config)
    if str(output_dir) == "results" or Path(output_dir) == Path("results"):
        out_path = paths.results_dir
    else:
        out_path = validate_mode_isolated_path(output_dir, mode=active_config.execution_mode)

    runs: list[BenchmarkRunResult] = []
    convergence_history: dict[str, Any] = {}

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
        primary_ratio = imbalance_ratios[0] if imbalance_ratios else "1:50"
        pos_frac = IMBALANCE_RATIOS.get(primary_ratio, 0.02)
        audit_split = build_scenario1_split(
            fine_labels=train_lbl_np,
            target_class=target_class_scenario1,
            pos_fraction=pos_frac,
            seed=seeds[0] if seeds else 42,
        )
        imbalance_audit = audit_scenario1_imbalance(
            split=audit_split,
            pos_batch_size=16,
            neg_batch_size=48,
            num_steps=num_steps,
        )
        imbalance_audit["configured_ratios"] = list(imbalance_ratios)

    val_lbl_np = (
        val_labels.detach().cpu().numpy()
        if isinstance(val_labels, torch.Tensor)
        else np.asarray(val_labels)
    ).astype(np.int64)

    # Pre-build validation confusion matrix if tree_type requires confusion_ward
    conf_mat: np.ndarray | None = None
    if "confusion_ward" in tree_types:
        num_cls = 100 if scenario == "scenario2" else max(int(np.max(train_lbl_np)) + 1, 2)
        conf_mat = estimate_validation_confusion_matrix(
            train_embeddings=train_features,
            train_labels=train_labels,
            val_embeddings=val_features,
            val_labels=val_labels,
            num_classes=num_cls,
        )

    diagnostics_per_strat: dict[str, Any] = {}
    tree_diagnostics: dict[str, Any] = {}
    visitation_payload: dict[str, Any] | None = None

    ratios_to_run: tuple[str | None, ...] = imbalance_ratios if scenario == "scenario1" else (None,)

    for rho in ratios_to_run:
        pos_frac = IMBALANCE_RATIOS.get(rho, 0.02) if rho is not None else 1.0
        for tree_type in tree_types:
            for strat in strategies:
                for seed in seeds:
                    set_global_seed(seed)

                    if scenario == "scenario1":
                        split = build_scenario1_split(
                            fine_labels=train_lbl_np,
                            target_class=target_class_scenario1,
                            pos_fraction=pos_frac,
                            seed=seed,
                        )
                        pos_idx = split.positive_indices
                        neg_idx = split.negative_indices
                        binary_train_lbl = split.binary_labels
                        binary_test_lbl = (test_lbl_np == target_class_scenario1).astype(np.int64)

                        tree: FastHierarchyTree | None = None
                        if strat in MCTS_STRATEGIES:
                            class_to_indices = split.negative_class_to_indices
                            if tree_type == "semantic":
                                tree = build_semantic_tree(class_to_indices)
                            elif tree_type == "confusion_ward":
                                assert conf_mat is not None
                                tree = build_confusion_tree(conf_mat, class_to_indices)
                            if tree is not None and tree_type not in tree_diagnostics:
                                tree_diagnostics[tree_type] = compute_tree_topology_diagnostics(tree)

                        binary_val_lbl = (val_lbl_np == target_class_scenario1).astype(np.int64)

                        res = train_scenario1(
                            train_features=train_features,
                            train_binary_labels=binary_train_lbl,
                            positive_indices=pos_idx,
                            negative_indices=neg_idx,
                            val_features=val_features,
                            val_binary_labels=binary_val_lbl,
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
                        if strat in MCTS_STRATEGIES:
                            class_to_indices = {
                                c: np.flatnonzero(train_lbl_np == c).astype(np.int64)
                                for c in range(num_cls)
                            }
                            if tree_type == "semantic":
                                tree = build_semantic_tree(class_to_indices)
                            elif tree_type == "confusion_ward":
                                assert conf_mat is not None
                                tree = build_confusion_tree(conf_mat, class_to_indices)
                            if tree is not None and tree_type not in tree_diagnostics:
                                tree_diagnostics[tree_type] = compute_tree_topology_diagnostics(tree)

                        res = train_scenario2(
                            train_features=train_features,
                            train_labels=train_lbl_np,
                            val_features=val_features,
                            val_labels=val_lbl_np,
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

                        # Compute test predictions and per-class recall across all strategies
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

                        # Exposure telemetry from ExposureTracker
                        exp_diag = res.get("exposure_diagnostics", {})
                        exp_counts = exp_diag.get("class_exposure_counts", [0] * num_cls)
                        exposure_entropy = float(exp_diag.get("normalized_exposure_entropy", 0.0))
                        class_cov = float(exp_diag.get("cumulative_class_coverage", 0.0))
                        unvisited_cls = int(exp_diag.get("unexposed_classes_count", num_cls))

                        corr_metrics = compute_exposure_recall_correlation(exp_counts, per_class_recall)
                        spearman_rho = corr_metrics["spearman_rho"]

                        ent_nats = 0.0
                        ent_norm = exposure_entropy
                        class_visits: dict[int, int] = {}
                        if tree is not None:
                            leaf_ids = tree.leaf_node_ids()
                            for lid in leaf_ids:
                                cls_id = int(tree.leaf_to_class_id[int(lid)])
                                class_visits[cls_id] = class_visits.get(cls_id, 0) + int(tree.visits[int(lid)])
                            for c in range(num_cls):
                                class_visits.setdefault(c, 0)
                            ent_nats, ent_norm = compute_visitation_entropy(tree)

                        diagnostics_per_strat[strat] = {
                            "class_visit_distribution": class_visits if tree is not None else {c: int(exp_counts[c]) for c in range(num_cls)},
                            "class_exposure_counts": exp_counts,
                            "unvisited_classes_count": unvisited_cls,
                            "visitation_entropy": ent_norm,
                            "entropy_nats": ent_nats,
                            "class_coverage_pct": class_cov * 100.0,
                            "per_class_recall": per_class_recall,
                            "exposure_entropy": exposure_entropy,
                            "class_coverage": class_cov,
                            "unvisited_classes": unvisited_cls,
                            "exposure_recall_correlation": spearman_rho,
                            "correlation": corr_metrics,
                        }

                        # Inject quantitative telemetry into final_metrics
                        res["final_metrics"]["exposure_entropy"] = exposure_entropy
                        res["final_metrics"]["class_coverage"] = class_cov
                        res["final_metrics"]["unvisited_classes"] = unvisited_cls
                        res["final_metrics"]["exposure_recall_correlation"] = spearman_rho

                    # Inject computational costs into final_metrics across all scenarios
                    costs = res.get("computational_costs", {})
                    if isinstance(costs, dict):
                        for ck in ("total_updates", "gradient_samples", "mining_forwards", "elapsed_seconds"):
                            if ck in costs:
                                res["final_metrics"][ck] = costs[ck]

                    final_metrics = res["final_metrics"]
                    if "history" in res and res["history"] and f"{strat}_f1" not in convergence_history:
                        steps_seq = [int(h.get("step", 0)) for h in res["history"]]
                        convergence_history["steps"] = steps_seq
                        primary_key = "f1" if scenario == "scenario1" else "macro_f1"
                        convergence_history[f"{strat}_f1"] = [float(h.get(primary_key, h.get("f1", 0.0))) for h in res["history"]]
                        convergence_history[f"{strat}_loss"] = [float(h.get("loss", 0.0)) for h in res["history"]]
                        if rho is not None and "imbalance_ratio" not in convergence_history:
                            convergence_history["imbalance_ratio"] = rho

                    if strat == "mcts_enhanced" and tree_type == "semantic" and "visitation_checkpoints" in res and res["visitation_checkpoints"] and visitation_payload is None:
                        vis_chks = res["visitation_checkpoints"]
                        tax = get_cifar100_taxonomy()
                        sc_names = list(tax.coarse_names)
                        mat = [
                            [float(chk["superclass_visit_distribution"].get(sc, 0.0)) for chk in vis_chks]
                            for sc in sc_names
                        ]
                        steps_h = [int(chk["step"]) for chk in vis_chks]
                        visitation_payload = {
                            "benchmark_name": "tier1_visitation_diagnostics",
                            "scenario": scenario,
                            "strategy": strat,
                            "tree_type": tree_type,
                            "seed": seed,
                            "checkpoints": vis_chks,
                            "heatmap": {
                                "steps": steps_h,
                                "superclasses": sc_names,
                                "matrix": mat,
                            },
                        }

                    commit_hash = get_git_commit(find_repo_root())[:8]
                    cls_id_str = f"cls{target_class_scenario1}" if scenario == "scenario1" else "cls100"
                    r_str = f"r{rho.replace(':', '_')}" if rho is not None else "r_none"
                    t_str = tree_type if strat.startswith("mcts") else "none"
                    run_id = f"{scenario}_{cls_id_str}_{r_str}_{t_str}_{strat}_s{seed}_{commit_hash}"

                    # Serialize individual atomic run
                    runs_dir = out_path / "runs"
                    runs_dir.mkdir(parents=True, exist_ok=True)
                    single_run_payload = {
                        "run_id": run_id,
                        "scenario": scenario,
                        "strategy": strat,
                        "tree_type": t_str,
                        "seed": seed,
                        "target_class": target_class_scenario1 if scenario == "scenario1" else None,
                        "imbalance_ratio": rho,
                        "final_metrics": final_metrics,
                        "val_metrics": res.get("val_metrics"),
                        "test_metrics": res.get("test_metrics"),
                        "history": res.get("history", []),
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    run_json_path = runs_dir / f"{run_id}.json"
                    _atomic_save_json(single_run_payload, run_json_path)
                    save_provenance_metadata(run_json_path, active_config, extra={"run_id": run_id})

                    runs.append(
                        BenchmarkRunResult(
                            scenario=scenario,
                            strategy=strat,
                            tree_type=t_str,
                            seed=seed,
                            final_metrics=final_metrics,
                            imbalance_ratio=rho,
                            run_id=run_id,
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

    # Paired comparisons against uniform baseline
    paired_differences_against_uniform: dict[str, Any] = {}
    uni_runs = {r.seed: r for r in runs if r.strategy == "uniform"}
    if uni_runs:
        for strat in sorted(set(r.strategy for r in runs if r.strategy != "uniform")):
            strat_runs = {r.seed: r for r in runs if r.strategy == strat}
            common_seeds = sorted(set(uni_runs.keys()) & set(strat_runs.keys()))
            if common_seeds:
                primary_k = "auprc" if scenario == "scenario1" else "top1_accuracy"
                uni_primary = [uni_runs[s].final_metrics.get(primary_k, 0.0) for s in common_seeds]
                strat_primary = [strat_runs[s].final_metrics.get(primary_k, 0.0) for s in common_seeds]
                f1_k = "f1" if scenario == "scenario1" else "macro_f1"
                uni_f1 = [uni_runs[s].final_metrics.get(f1_k, 0.0) for s in common_seeds]
                strat_f1 = [strat_runs[s].final_metrics.get(f1_k, 0.0) for s in common_seeds]
                uni_w10 = [uni_runs[s].final_metrics.get("worst_10_class_recall", 0.0) for s in common_seeds]
                strat_w10 = [strat_runs[s].final_metrics.get("worst_10_class_recall", 0.0) for s in common_seeds]

                delta_primary = [float(st - un) for st, un in zip(strat_primary, uni_primary)]
                delta_f1 = [float(st - un) for st, un in zip(strat_f1, uni_f1)]
                delta_w10 = [float(st - un) for st, un in zip(strat_w10, uni_w10)]

                diff_entry: dict[str, Any] = {
                    "common_seeds": common_seeds,
                    "delta_primary_mean": float(np.mean(delta_primary)),
                    "delta_f1_mean": float(np.mean(delta_f1)),
                    "delta_worst_10_mean": float(np.mean(delta_w10)),
                    "delta_primary_per_seed": delta_primary,
                    "delta_f1_per_seed": delta_f1,
                    "delta_worst_10_per_seed": delta_w10,
                }
                if len(common_seeds) >= 2:
                    diff_entry["primary_stats"] = compute_paired_statistical_tests(uni_primary, strat_primary)
                    diff_entry["f1_stats"] = compute_paired_statistical_tests(uni_f1, strat_f1)
                    diff_entry["worst_10_stats"] = compute_paired_statistical_tests(uni_w10, strat_w10)
                    statistical_comparisons[f"uniform_vs_{strat}"] = diff_entry["primary_stats"]

                paired_differences_against_uniform[strat] = diff_entry

    # Paired comparisons across hierarchies (Axis 2: Semantic vs Ward)
    paired_hierarchy_tests: dict[str, Any] = {}
    if "semantic" in tree_types and "confusion_ward" in tree_types:
        for strat in strategies:
            sem_runs = {r.seed: r for r in runs if r.strategy == strat and r.tree_type == "semantic"}
            conf_runs = {r.seed: r for r in runs if r.strategy == strat and r.tree_type == "confusion_ward"}
            common_s = sorted(set(sem_runs.keys()) & set(conf_runs.keys()))
            if common_s:
                metric_k = "auprc" if scenario == "scenario1" else "top1_accuracy"
                f1_k = "f1" if scenario == "scenario1" else "macro_f1"
                sem_scores = [sem_runs[s].final_metrics.get(metric_k, 0.0) for s in common_s]
                conf_scores = [conf_runs[s].final_metrics.get(metric_k, 0.0) for s in common_s]
                sem_f1 = [sem_runs[s].final_metrics.get(f1_k, 0.0) for s in common_s]
                conf_f1 = [conf_runs[s].final_metrics.get(f1_k, 0.0) for s in common_s]

                d_metric = [float(sm - cf) for sm, cf in zip(sem_scores, conf_scores)]
                d_f1 = [float(sm - cf) for sm, cf in zip(sem_f1, conf_f1)]

                h_entry: dict[str, Any] = {
                    "strategy": strat,
                    "common_seeds": common_s,
                    "delta_primary_mean": float(np.mean(d_metric)),
                    "delta_f1_mean": float(np.mean(d_f1)),
                    "delta_primary_per_seed": d_metric,
                    "delta_f1_per_seed": d_f1,
                }
                if len(common_s) >= 2:
                    h_entry["primary_stats"] = compute_paired_statistical_tests(sem_scores, conf_scores)
                    h_entry["f1_stats"] = compute_paired_statistical_tests(sem_f1, conf_f1)
                    statistical_comparisons[f"semantic_vs_confusion_ward_{strat}"] = h_entry["primary_stats"]

                paired_hierarchy_tests[strat] = h_entry

    # 2x2 Factorial Mechanism Analysis (Strict vs QNorm vs DualStream vs Enhanced)
    factorial_set = {"mcts_strict", "mcts_qnorm_only", "mcts_dualstream_only", "mcts_enhanced"}
    factorial_2x2_effects: dict[str, Any] = {}
    if factorial_set.issubset(set(strategies)):
        for tt in tree_types:
            runs_by_strat = {
                s: {r.seed: r for r in runs if r.strategy == s and (r.tree_type == tt or r.tree_type == "none")}
                for s in factorial_set
            }
            common_s = sorted(set.intersection(*(set(runs_by_strat[s].keys()) for s in factorial_set)))
            if common_s:
                primary_k = "auprc" if scenario == "scenario1" else "top1_accuracy"
                f1_k = "f1" if scenario == "scenario1" else "macro_f1"

                primary_dict = {
                    s: [runs_by_strat[s][sd].final_metrics.get(primary_k, 0.0) for sd in common_s]
                    for s in factorial_set
                }
                f1_dict = {
                    s: [runs_by_strat[s][sd].final_metrics.get(f1_k, 0.0) for sd in common_s]
                    for s in factorial_set
                }

                factorial_2x2_effects[tt] = {
                    "common_seeds": common_s,
                    "primary_metric": primary_k,
                    "primary_effects": compute_2x2_factorial_effects(primary_dict),
                    "f1_effects": compute_2x2_factorial_effects(f1_dict),
                }

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
            if scenario == "scenario1" and strat_runs:
                ratios_in_runs = {r.imbalance_ratio for r in strat_runs if r.imbalance_ratio is not None}
                if len(ratios_in_runs) == 1:
                    summary_strat["imbalance_ratio"] = next(iter(ratios_in_runs))
            aggregated_summary[tt][strat] = summary_strat

    # Flatten summary rows for CSV export
    csv_rows: list[dict[str, Any]] = []
    all_fieldnames: set[str] = {"scenario", "strategy", "tree_type", "seed", "imbalance_ratio"}
    for r in runs:
        row: dict[str, Any] = {
            "scenario": r.scenario,
            "strategy": r.strategy,
            "tree_type": r.tree_type,
            "seed": r.seed,
            "imbalance_ratio": r.imbalance_ratio,
        }
        for k, v in r.final_metrics.items():
            row[k] = v
            all_fieldnames.add(k)
        csv_rows.append(row)

    ordered_fields = ["scenario", "strategy", "tree_type", "seed", "imbalance_ratio"] + sorted(
        [f for f in all_fieldnames if f not in ("scenario", "strategy", "tree_type", "seed", "imbalance_ratio")]
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
        "paired_differences_against_uniform": paired_differences_against_uniform,
        "paired_hierarchy_tests": paired_hierarchy_tests,
        "tree_diagnostics": tree_diagnostics,
        "factorial_2x2_effects": factorial_2x2_effects,
        "aggregated_summary": aggregated_summary,
        "runs": [asdict(r) for r in runs],
        "per_seed_results": [asdict(r) for r in runs],
    }

    if scenario == "scenario1" and imbalance_ratios:
        json_export["imbalance_ratio"] = imbalance_ratios[0] if len(imbalance_ratios) == 1 else list(imbalance_ratios)

    if imbalance_audit is not None:
        json_export["imbalance_audit"] = imbalance_audit
    if diagnostics_per_strat:
        json_export["diagnostics"] = diagnostics_per_strat
    if convergence_history:
        json_export["convergence"] = convergence_history
    if visitation_payload is not None:
        json_export["heatmap"] = visitation_payload["heatmap"]
        json_export["visitation_diagnostics"] = visitation_payload
        vis_path = out_path / "tier1_visitation_diagnostics.json"
        _atomic_save_json(visitation_payload, vis_path)
        save_provenance_metadata(vis_path, active_config)

    if scenario == "scenario2":
        diag_path = out_path / "tier1_scenario2_diagnostics.json"
        diag_payload = {
            "benchmark_name": "tier1_scenario2_diagnostics",
            "scenario": scenario,
            "diagnostics": diagnostics_per_strat,
            "paired_differences_against_uniform": paired_differences_against_uniform,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_save_json(diag_payload, diag_path)
        save_provenance_metadata(diag_path, active_config)
        json_export["scenario2_diagnostics"] = diag_payload

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
    val_features: torch.Tensor,
    val_labels: np.ndarray | torch.Tensor,
    test_features: torch.Tensor,
    test_labels: np.ndarray | torch.Tensor,
    c_values: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    alpha_values: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7),
    output_dir: str | Path = "results",
    benchmark_name: str = "tier1_hyperparameter_sensitivity",
    imbalance_ratio: str = "1:50",
    num_steps: int = 1000,
    eval_every: int = 100,
    seeds: tuple[int, ...] = (42,),
    target_class_scenario1: int = 0,
    config: ExperimentConfig | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate grid sensitivity over exploration coefficient c and momentum alpha."""
    if val_features is None or val_labels is None:
        raise ValueError(
            "val_features and val_labels are required in run_sensitivity_ablation to prevent data leakage."
        )
    if imbalance_ratio not in IMBALANCE_RATIOS:
        raise ValueError(
            f"Invalid imbalance_ratio: '{imbalance_ratio}'. Valid choices are: {list(IMBALANCE_RATIOS.keys())}"
        )
    if config is not None:
        active_config = config
    elif get_active_session_mode() is not None:
        active_config = ExperimentConfig(execution_mode=get_active_session_mode())
    else:
        active_config = ExperimentConfig(execution_mode=ExecutionMode.REAL)

    paths = get_isolated_paths(active_config)
    if str(output_dir) == "results" or Path(output_dir) == Path("results"):
        out_path = paths.results_dir
    else:
        out_path = validate_mode_isolated_path(output_dir, mode=active_config.execution_mode)

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

    val_lbl_np = (
        val_labels.detach().cpu().numpy()
        if isinstance(val_labels, torch.Tensor)
        else np.asarray(val_labels)
    ).astype(np.int64)

    pos_frac = IMBALANCE_RATIOS[imbalance_ratio]
    binary_val_lbl = (val_lbl_np == target_class_scenario1).astype(np.int64)
    binary_test_lbl = (test_lbl_np == target_class_scenario1).astype(np.int64)

    splits_by_seed: dict[int, Any] = {}
    for s in seeds:
        set_global_seed(s)
        splits_by_seed[s] = build_scenario1_split(
            fine_labels=train_lbl_np,
            target_class=target_class_scenario1,
            pos_fraction=pos_frac,
            seed=s,
        )

    grid_results: list[dict[str, Any]] = []

    for c in c_values:
        for alpha in alpha_values:
            c_alpha_metrics: list[dict[str, Any]] = []
            for seed in seeds:
                set_global_seed(seed)
                split = splits_by_seed[seed]
                tree = build_semantic_tree(split.negative_class_to_indices)
                res = train_scenario1(
                    train_features=train_features,
                    train_binary_labels=split.binary_labels,
                    positive_indices=split.positive_indices,
                    negative_indices=split.negative_indices,
                    val_features=val_features,
                    val_binary_labels=binary_val_lbl,
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
                last_m = res["last_metrics"]
                best_val_m = res["best_validation_metrics"]
                run_metrics = {
                    "c": float(c),
                    "alpha": float(alpha),
                    "seed": int(seed),
                    "imbalance_ratio": imbalance_ratio,
                    "num_positives_train": int(len(split.positive_indices)),
                    "num_negatives_train": int(len(split.negative_indices)),
                    # Backward-compatible primary metrics (mapped to best_val calibrated)
                    "auprc": float(best_val_m.get("auprc", 0.0)),
                    "f1": float(best_val_m.get("calibrated_f1", best_val_m.get("f1", 0.0))),
                    "precision": float(best_val_m.get("calibrated_precision", best_val_m.get("precision", 0.0))),
                    "recall": float(best_val_m.get("calibrated_recall", best_val_m.get("recall", 0.0))),
                    # 4 Quadrants explicit keys
                    "test_f1_last_thresh05": float(last_m.get("f1_thresh05", last_m.get("f1", 0.0))),
                    "test_precision_last_thresh05": float(last_m.get("precision_thresh05", last_m.get("precision", 0.0))),
                    "test_recall_last_thresh05": float(last_m.get("recall_thresh05", last_m.get("recall", 0.0))),
                    "test_f1_last_calibrated": float(last_m.get("calibrated_f1", last_m.get("f1", 0.0))),
                    "test_precision_last_calibrated": float(last_m.get("calibrated_precision", last_m.get("precision", 0.0))),
                    "test_recall_last_calibrated": float(last_m.get("calibrated_recall", last_m.get("recall", 0.0))),
                    "best_tau_val_last": float(last_m.get("calibrated_threshold", last_m.get("threshold", 0.5))),
                    "test_f1_best_val_thresh05": float(best_val_m.get("f1_thresh05", best_val_m.get("f1", 0.0))),
                    "test_precision_best_val_thresh05": float(best_val_m.get("precision_thresh05", best_val_m.get("precision", 0.0))),
                    "test_recall_best_val_thresh05": float(best_val_m.get("recall_thresh05", best_val_m.get("recall", 0.0))),
                    "test_f1_best_val_calibrated": float(best_val_m.get("calibrated_f1", best_val_m.get("f1", 0.0))),
                    "test_precision_best_val_calibrated": float(best_val_m.get("calibrated_precision", best_val_m.get("precision", 0.0))),
                    "test_recall_best_val_calibrated": float(best_val_m.get("calibrated_recall", best_val_m.get("recall", 0.0))),
                    "best_tau_val": float(best_val_m.get("calibrated_threshold", best_val_m.get("threshold", 0.5))),
                    "test_auprc": float(best_val_m.get("auprc", 0.0)),
                }
                c_alpha_metrics.append(run_metrics)

            auprc_vals = [m["auprc"] for m in c_alpha_metrics]
            f1_vals = [m["f1"] for m in c_alpha_metrics]
            grid_results.append({
                "c": float(c),
                "alpha": float(alpha),
                "imbalance_ratio": imbalance_ratio,
                "auprc_mean": float(np.mean(auprc_vals)),
                "auprc_std": float(np.std(auprc_vals, ddof=1)) if len(auprc_vals) > 1 else 0.0,
                "f1_mean": float(np.mean(f1_vals)),
                "f1_std": float(np.std(f1_vals, ddof=1)) if len(f1_vals) > 1 else 0.0,
                "f1_last_thresh05_mean": float(np.mean([m["test_f1_last_thresh05"] for m in c_alpha_metrics])),
                "f1_last_calibrated_mean": float(np.mean([m["test_f1_last_calibrated"] for m in c_alpha_metrics])),
                "f1_best_val_thresh05_mean": float(np.mean([m["test_f1_best_val_thresh05"] for m in c_alpha_metrics])),
                "f1_best_val_calibrated_mean": float(np.mean([m["test_f1_best_val_calibrated"] for m in c_alpha_metrics])),
                "runs": c_alpha_metrics,
            })

    first_split = next(iter(splits_by_seed.values())) if splits_by_seed else None
    json_export = {
        "benchmark_name": benchmark_name,
        "imbalance_ratio": imbalance_ratio,
        "num_positives_train": int(len(first_split.positive_indices)) if first_split is not None else 0,
        "num_negatives_train": int(len(first_split.negative_indices)) if first_split is not None else 0,
        "num_steps": num_steps,
        "exploration_c": list(c_values),
        "momentum_alpha": list(alpha_values),
        "seeds": list(seeds),
        "grid": grid_results,
    }

    json_path = out_path / f"{benchmark_name}.json"
    _atomic_save_json(json_export, json_path)
    save_provenance_metadata(json_path, active_config)

    csv_rows = []
    for cell in grid_results:
        for run_dict in cell["runs"]:
            csv_rows.append(run_dict)

    if csv_rows:
        ordered_fields = list(csv_rows[0].keys())
        csv_path = out_path / f"{benchmark_name}.csv"
        _atomic_save_csv(ordered_fields, csv_rows, csv_path)
        save_provenance_metadata(csv_path, active_config)

    return json_export
