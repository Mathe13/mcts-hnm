"""Scientific Campaign Runner and Frozen Protocol Matrix Orchestrator (Phase D1)."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.extract_embeddings import extract_or_generate_embeddings, load_embedding_cache
from src.data.splits import (
    IMBALANCE_RATIOS,
    build_nested_imbalanced_splits,
    build_scenario1_split,
    build_stratified_imbalanced_split,
)
from src.training.trainer import (
    MCTS_STRATEGIES,
    SUPPORTED_STRATEGIES,
    train_scenario1,
    train_scenario2,
)
from src.trees.semantic_tree import build_semantic_tree
from src.utils.experiment_config import (
    ExecutionMode,
    ExperimentConfig,
    get_active_session_mode,
    get_isolated_paths,
    save_provenance_metadata,
    validate_mode_isolated_path,
)
from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert NumPy scalars/arrays and PyTorch tensors to JSON-serializable types."""
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(obj, np.ndarray):
        return [_sanitize_for_json(v) for v in obj.tolist()]
    if isinstance(obj, torch.Tensor):
        return _sanitize_for_json(obj.detach().cpu().numpy())
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in obj]
    return str(obj)


def _atomic_save_json(data: Any, path: Path) -> None:
    """Save data to path atomically via a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = _sanitize_for_json(data)
    tmp_path = path.with_suffix(".tmp.json")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2)
    tmp_path.replace(path)


def run_scientific_campaign(
    scenarios: tuple[str, ...] = ("scenario1", "scenario2"),
    scenario1_ratios: tuple[str, ...] = ("1:50", "1:100", "1:500"),
    scenario1_methods: tuple[str, ...] = ("uniform", "ohem", "mcts_strict", "mcts_enhanced"),
    scenario2_methods: tuple[str, ...] = (
        "uniform",
        "uniform_grouped",
        "mcts_strict",
        "mcts_qnorm_only",
        "mcts_dualstream_only",
        "mcts_enhanced",
    ),
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    target_class_scenario1: int = 0,
    target_positive_count: int = 80,
    config: ExperimentConfig | None = None,
    train_features: torch.Tensor | None = None,
    train_labels: np.ndarray | torch.Tensor | None = None,
    val_features: torch.Tensor | None = None,
    val_labels: np.ndarray | torch.Tensor | None = None,
    test_features: torch.Tensor | None = None,
    test_labels: np.ndarray | torch.Tensor | None = None,
    output_dir: str | Path | None = None,
    num_steps_scenario1: int | None = None,
    num_steps_scenario2: int | None = None,
    eval_every: int | None = None,
    device: torch.device | None = None,
    campaign_id: str | None = None,
) -> dict[str, Any]:
    r"""Execute frozen scientific campaign matrix across Scenarios 1 and 2 with paired seeds.

    Guarantees:
    - Fixed frozen matrix: Scenarios, methods, ratios, and seeds.
    - Scenario 1 shares identical positive instances across all ratios per seed.
    - Both `last` and `best_validation` metrics are computed and persisted.
    - Computational costs (updates, gradient samples, mining forwards, elapsed time) are tracked.
    - Atomic JSON and `.meta.json` sidecar provenance per run.
    - Consolidated `campaign_manifest.json` indexing all executed runs.
    """
    if config is not None:
        active_config = config
    elif get_active_session_mode() is not None:
        active_config = ExperimentConfig(execution_mode=get_active_session_mode())
    else:
        active_config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)

    paths = get_isolated_paths(active_config)

    # Resolve output directory
    if output_dir is not None:
        campaign_dir = Path(output_dir)
    else:
        campaign_dir = paths.results_dir / "campaign"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    cid = campaign_id if campaign_id is not None else f"campaign_{active_config.execution_mode.value}"

    # Load data if not provided
    if train_features is None or train_labels is None or val_features is None or val_labels is None:
        cache_path = paths.embeddings_dir / "cifar100_resnet18.pt"
        if not cache_path.is_file():
            if active_config.execution_mode == ExecutionMode.REAL:
                raise FileNotFoundError(
                    f"Real CIFAR-100 embeddings not found at '{cache_path}'. "
                    "Run feature extraction in REAL mode first."
                )
            extract_or_generate_embeddings(
                output_path=cache_path,
                config=active_config,
                smoke_samples_per_class=30,
                smoke_feature_dim=64,
            )

        bundle = load_embedding_cache(cache_path)
        train_features = bundle["train"]["embeddings"]
        train_labels = bundle["train"]["fine_labels"]
        val_features = bundle["val"]["embeddings"]
        val_labels = bundle["val"]["fine_labels"]
        test_features = bundle["test"]["embeddings"]
        test_labels = bundle["test"]["fine_labels"]

    train_lbl_np = (
        train_labels.detach().cpu().numpy()
        if isinstance(train_labels, torch.Tensor)
        else np.asarray(train_labels)
    ).astype(np.int64)

    val_lbl_np = (
        val_labels.detach().cpu().numpy()
        if isinstance(val_labels, torch.Tensor)
        else np.asarray(val_labels)
    ).astype(np.int64)

    test_lbl_np = (
        test_labels.detach().cpu().numpy()
        if isinstance(test_labels, torch.Tensor)
        else np.asarray(test_labels)
    ).astype(np.int64) if test_features is not None and test_labels is not None else None

    # Determine step counts based on mode
    is_real = active_config.execution_mode == ExecutionMode.REAL
    steps_s1 = num_steps_scenario1 if num_steps_scenario1 is not None else (5000 if is_real else 10)
    steps_s2 = num_steps_scenario2 if num_steps_scenario2 is not None else (5000 if is_real else 10)
    eval_step = eval_every if eval_every is not None else (100 if is_real else 5)

    runs_registry: dict[str, dict[str, Any]] = {}
    total_expected = 0
    completed_count = 0
    failed_count = 0

    # 1. Scenario 1 execution
    if "scenario1" in scenarios:
        int_ratios = tuple(int(r.split(":")[1]) for r in scenario1_ratios)
        all_target_indices = np.flatnonzero(train_lbl_np == target_class_scenario1)
        actual_pos_count = (
            target_positive_count
            if is_real
            else min(target_positive_count, max(2, len(all_target_indices)))
        )

        for seed in seeds:
            set_global_seed(seed)
            # Build nested splits sharing identical positive instances across all ratios
            split_family: dict[int, Any] = {}
            try:
                split_family = build_nested_imbalanced_splits(
                    fine_labels=train_lbl_np,
                    target_class=target_class_scenario1,
                    target_positive_count=actual_pos_count,
                    ratios=int_ratios,
                    seed=seed,
                )
            except Exception:
                for ratio_str, ratio_int in zip(scenario1_ratios, int_ratios):
                    try:
                        split_family[ratio_int] = build_stratified_imbalanced_split(
                            fine_labels=train_lbl_np,
                            target_class=target_class_scenario1,
                            target_positive_count=actual_pos_count,
                            target_neg_pos_ratio=ratio_int,
                            seed=seed,
                        )
                    except Exception:
                        pos_frac = IMBALANCE_RATIOS.get(ratio_str, 1.0 / max(1, ratio_int))
                        split_family[ratio_int] = build_scenario1_split(
                            fine_labels=train_lbl_np,
                            target_class=target_class_scenario1,
                            pos_fraction=pos_frac,
                            seed=seed,
                        )

            for ratio_str, ratio_int in zip(scenario1_ratios, int_ratios):
                split = split_family.get(ratio_int)
                if split is None:
                    continue

                tree = build_semantic_tree(split.negative_class_to_indices)
                val_binary_lbl = (val_lbl_np == target_class_scenario1).astype(np.int64)
                test_binary_lbl = (test_lbl_np == target_class_scenario1).astype(np.int64) if test_lbl_np is not None else None

                for method in scenario1_methods:
                    total_expected += 1
                    run_id = f"s1_{ratio_str.replace(':', '_')}_{method}_seed{seed}"
                    run_file = campaign_dir / f"{run_id}.json"

                    try:
                        res = train_scenario1(
                            train_features=train_features,
                            train_binary_labels=split.binary_labels,
                            positive_indices=split.positive_indices,
                            negative_indices=split.negative_indices,
                            val_features=val_features,
                            val_binary_labels=val_binary_lbl,
                            test_features=test_features,
                            test_binary_labels=test_binary_lbl,
                            strategy=method,
                            tree=tree,
                            num_steps=steps_s1,
                            eval_every=eval_step,
                            seed=seed,
                            device=device,
                        )

                        # Check for non-finite metrics
                        last_m = res.get("last_metrics", {})
                        best_m = res.get("best_validation_metrics", {})
                        has_nan = any(
                            np.isnan(v) or np.isinf(v)
                            for m in (last_m, best_m)
                            for v in m.values()
                            if isinstance(v, (int, float, np.number))
                        )
                        if has_nan:
                            status = "failed"
                            err = "NaN/Inf detected in evaluation metrics"
                            failed_count += 1
                        else:
                            status = "completed"
                            err = None
                            completed_count += 1

                        run_record = {
                            "run_id": run_id,
                            "scenario": "scenario1",
                            "imbalance_ratio": ratio_str,
                            "strategy": method,
                            "seed": seed,
                            "status": status,
                            "error": err,
                            "last_metrics": last_m,
                            "best_validation_metrics": best_m,
                            "computational_costs": res.get("computational_costs"),
                            "exposure_diagnostics": res.get("exposure_diagnostics"),
                            "final_num_leaves": res.get("final_num_leaves"),
                        }
                    except Exception as exc:
                        logger.exception("Error executing run %s", run_id)
                        status = "failed"
                        err = str(exc)
                        failed_count += 1
                        run_record = {
                            "run_id": run_id,
                            "scenario": "scenario1",
                            "imbalance_ratio": ratio_str,
                            "strategy": method,
                            "seed": seed,
                            "status": status,
                            "error": err,
                            "last_metrics": None,
                            "best_validation_metrics": None,
                            "computational_costs": None,
                            "exposure_diagnostics": None,
                        }

                    _atomic_save_json(run_record, run_file)
                    save_provenance_metadata(run_file, active_config, extra={"run_id": run_id, "seed": seed})
                    runs_registry[run_id] = {
                        "run_id": run_id,
                        "scenario": "scenario1",
                        "imbalance_ratio": ratio_str,
                        "strategy": method,
                        "seed": seed,
                        "status": status,
                        "run_file": str(run_file),
                        "last_metrics": run_record["last_metrics"],
                        "best_validation_metrics": run_record["best_validation_metrics"],
                    }

    # 2. Scenario 2 execution
    if "scenario2" in scenarios:
        num_classes = max(int(np.max(train_lbl_np)) + 1, 2)
        class_to_indices = {
            c: np.flatnonzero(train_lbl_np == c).astype(np.int64) for c in range(num_classes)
        }
        tree = build_semantic_tree(class_to_indices)

        for seed in seeds:
            set_global_seed(seed)
            for method in scenario2_methods:
                total_expected += 1
                run_id = f"s2_{method}_seed{seed}"
                run_file = campaign_dir / f"{run_id}.json"

                try:
                    res = train_scenario2(
                        train_features=train_features,
                        train_labels=train_lbl_np,
                        val_features=val_features,
                        val_labels=val_lbl_np,
                        test_features=test_features,
                        test_labels=test_lbl_np,
                        strategy=method,
                        tree=tree,
                        num_classes=num_classes,
                        num_steps=steps_s2,
                        eval_every=eval_step,
                        seed=seed,
                        device=device,
                    )

                    last_m = res.get("last_metrics", {})
                    best_m = res.get("best_validation_metrics", {})
                    has_nan = any(
                        np.isnan(v) or np.isinf(v)
                        for m in (last_m, best_m)
                        for v in m.values()
                        if isinstance(v, (int, float, np.number))
                    )
                    if has_nan:
                        status = "failed"
                        err = "NaN/Inf detected in evaluation metrics"
                        failed_count += 1
                    else:
                        status = "completed"
                        err = None
                        completed_count += 1

                    run_record = {
                        "run_id": run_id,
                        "scenario": "scenario2",
                        "strategy": method,
                        "seed": seed,
                        "status": status,
                        "error": err,
                        "last_metrics": last_m,
                        "best_validation_metrics": best_m,
                        "computational_costs": res.get("computational_costs"),
                        "exposure_diagnostics": res.get("exposure_diagnostics"),
                        "final_num_leaves": res.get("final_num_leaves"),
                    }
                except Exception as exc:
                    logger.exception("Error executing run %s", run_id)
                    status = "failed"
                    err = str(exc)
                    failed_count += 1
                    run_record = {
                        "run_id": run_id,
                        "scenario": "scenario2",
                        "strategy": method,
                        "seed": seed,
                        "status": status,
                        "error": err,
                        "last_metrics": None,
                        "best_validation_metrics": None,
                        "computational_costs": None,
                        "exposure_diagnostics": None,
                    }

                _atomic_save_json(run_record, run_file)
                save_provenance_metadata(run_file, active_config, extra={"run_id": run_id, "seed": seed})
                runs_registry[run_id] = {
                    "run_id": run_id,
                    "scenario": "scenario2",
                    "strategy": method,
                    "seed": seed,
                    "status": status,
                    "run_file": str(run_file),
                    "last_metrics": run_record["last_metrics"],
                    "best_validation_metrics": run_record["best_validation_metrics"],
                }

    # Consolidated manifest
    manifest_data = {
        "campaign_id": cid,
        "mode": active_config.execution_mode.value,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_expected_runs": total_expected,
        "completed_runs": completed_count,
        "failed_runs": failed_count,
        "seeds": list(seeds),
        "scenarios": list(scenarios),
        "scenario1_ratios": list(scenario1_ratios),
        "scenario1_methods": list(scenario1_methods),
        "scenario2_methods": list(scenario2_methods),
        "runs": runs_registry,
    }

    manifest_file = campaign_dir / "campaign_manifest.json"
    _atomic_save_json(manifest_data, manifest_file)
    save_provenance_metadata(manifest_file, active_config)

    manifest_data["manifest_path"] = str(manifest_file)
    manifest_data["campaign_dir"] = str(campaign_dir)
    return manifest_data
