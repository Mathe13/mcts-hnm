"""Paper artifacts and publication figure/table generation module.

Implements automated export of publication-grade figures (PDF/PNG),
LaTeX booktabs tables (.tex), paper metadata loading (paper_metadata.yaml),
and manuscript compilation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from src.utils.experiment_config import (
    ExecutionMode,
    ExperimentConfig,
    find_repo_root,
    get_active_session_mode,
    get_isolated_paths,
    validate_provenance_metadata,
)

logger = logging.getLogger(__name__)

DEFAULT_PAPER_METADATA: dict[str, Any] = {
    "title": "MCTS-HNM: Monte Carlo Tree Search for Hierarchical Hard Negative Mining in Class-Imbalanced Learning",
    "authors": [
        {
            "name": "",
            "affiliation": "PPGC - Universidade Federal do Rio Grande do Sul (UFRGS)",
            "email": "",
        },
        {
            "name": "Co-Author Placeholder",
            "affiliation": "PPGC - Universidade Federal do Rio Grande do Sul (UFRGS)",
            "email": "coauthor@inf.ufrgs.br",
        },
    ],
    "institution": (
        "Postgraduate Program in Computer Science (PPGC)\n"
        "Universidade Federal do Rio Grande do Sul (UFRGS)\n"
        "Porto Alegre, RS, Brazil"
    ),
    "abstract": (
        "Deep image classification under severe class imbalance and hierarchical label structures "
        "presents significant challenges for standard training regimes. In this work, we investigate "
        "Monte Carlo Tree Search (MCTS) as an active reinforcement learning agent for hierarchical "
        "hard negative mining (MCTS-HNM)."
    ),
    "keywords": [
        "Monte Carlo Tree Search",
        "Hard Negative Mining",
        "Hierarchical Classification",
        "Class Imbalance",
        "Curriculum Learning",
    ],
}


def _simple_yaml_parse(text: str) -> dict[str, Any]:
    """Lightweight fallback YAML parser for basic paper_metadata dictionaries."""
    data: dict[str, Any] = {}
    current_key = None
    list_items = []
    in_list = False

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" in line and not stripped.startswith("-"):
            if in_list and current_key:
                data[current_key] = list_items
                in_list = False
                list_items = []
            key, val = line.split(":", 1)
            current_key = key.strip()
            val = val.strip().strip('"').strip("'")
            if val:
                data[current_key] = val
        elif stripped.startswith("-") and current_key:
            in_list = True
            list_items.append(stripped.lstrip("-").strip().strip('"').strip("'"))
    if in_list and current_key:
        data[current_key] = list_items

    return data if data else DEFAULT_PAPER_METADATA.copy()


def load_paper_metadata(path: str | Path = "paper_metadata.yaml") -> dict[str, Any]:
    """Load paper metadata from a YAML file, falling back to defaults if missing or invalid.

    Args:
        path: Path to paper_metadata.yaml.

    Returns:
        Dictionary containing title, authors, institution, and keywords.
    """
    file_path = Path(path)
    if not file_path.exists():
        logger.warning(
            "Paper metadata file %s not found. Using default PPGC UFRGS metadata.",
            file_path,
        )
        return DEFAULT_PAPER_METADATA.copy()

    try:
        try:
            import yaml  # type: ignore

            with open(file_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict) and "title" in loaded:
                return loaded
        except Exception as e:
            logger.debug("PyYAML load failed: %s; using fallback parser", e)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        parsed = _simple_yaml_parse(text)
        if isinstance(parsed, dict) and "title" in parsed:
            return parsed
    except Exception as exc:
        logger.warning(
            "Failed to parse %s: %s. Falling back to default metadata.", file_path, exc
        )

    return DEFAULT_PAPER_METADATA.copy()


def _compute_sha256(file_path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    if not file_path.is_file():
        return ""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _get_git_commit(cwd: Optional[Path] = None) -> str:
    """Retrieve current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def load_real_benchmark_results(
    results_dir: str | Path = "results",
    expected_mode: ExecutionMode = ExecutionMode.REAL,
) -> dict[str, Any]:
    """Load, validate provenance, and normalize real benchmark results from results_dir.

    Args:
        results_dir: Directory containing benchmark result files.
        expected_mode: Expected execution mode (defaults to ExecutionMode.REAL).

    Returns:
        Structured dictionary matching Scenario 1, Scenario 2, and ablation schemas.

    Raises:
        FileNotFoundError: If no valid real benchmark result files exist in results_dir.
        ValueError: If loaded artifacts fail provenance validation (e.g. synthetic data in REAL mode).
    """
    import numpy as np

    res_path = Path(results_dir)

    def _validate_meta(json_f: Path) -> None:
        meta_f = json_f.parent / f"{json_f.name}.meta.json"
        validate_provenance_metadata(meta_f, expected_mode=expected_mode)

    tier1_file = res_path / "tier1_benchmark_summary.json"
    if tier1_file.exists():
        _validate_meta(tier1_file)
        try:
            with open(tier1_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and ("scenario1" in data or "scenario2" in data):
                logger.info("Loaded real benchmark results from %s", tier1_file)
                if expected_mode == ExecutionMode.SMOKE:
                    data["is_smoke"] = True
                return data
        except Exception as e:
            logger.warning("Failed to load %s: %s", tier1_file, e)

    # Check for individual benchmark suite outputs
    axis1_file = res_path / "tier1_axis1_samplers.json"
    scen2_file = res_path / "tier1_scenario2_curriculum.json"
    ablation_file = res_path / "tier1_axis2_tree_ablation.json"
    sensitivity_file = res_path / "tier1_hyperparameter_sensitivity.json"

    found_any = False
    merged: dict[str, Any] = {
        "scenario1": {},
        "scenario2": {},
        "ablation": {},
        "convergence": {},
        "heatmap": {},
        "sensitivity": {},
    }

    if axis1_file.exists():
        found_any = True
        _validate_meta(axis1_file)
        try:
            with open(axis1_file, "r", encoding="utf-8") as f:
                d1 = json.load(f)
            if "scenario1" in d1 and isinstance(d1["scenario1"], dict) and any("f1" in str(v) for v in d1["scenario1"].values()):
                merged["scenario1"] = d1["scenario1"]
            elif "runs" in d1:
                scen1_norm: dict[str, Any] = {}
                for r in d1.get("runs", []):
                    strat = r.get("strategy")
                    metrics = r.get("final_metrics", {})
                    if not strat or not metrics:
                        continue
                    scen1_norm.setdefault(strat, {}).setdefault("10", {"f1": [], "auprc": [], "fpr95": []})
                    scen1_norm[strat]["10"]["f1"].append(float(metrics.get("f1", 0.0)))
                    scen1_norm[strat]["10"]["auprc"].append(float(metrics.get("auprc", 0.0)))
                    scen1_norm[strat]["10"]["fpr95"].append(float(metrics.get("fpr_at_95_recall", metrics.get("fpr95", 0.0))))

                for s, r_dict in scen1_norm.items():
                    for rho, m_dict in r_dict.items():
                        for mk in ("f1", "auprc", "fpr95"):
                            vals = m_dict[mk]
                            m_dict[mk] = {
                                "mean": float(np.mean(vals)) if vals else 0.0,
                                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                                "values": vals,
                            }
                merged["scenario1"] = scen1_norm
            elif "aggregated_summary" in d1:
                scen1_norm = {}
                for tt, strat_dict in d1["aggregated_summary"].items():
                    for strat, s_metrics in strat_dict.items():
                        scen1_norm.setdefault(strat, {})["10"] = {
                            "f1": {"mean": s_metrics.get("f1_mean", 0.0), "std": s_metrics.get("f1_std", 0.0), "values": [s_metrics.get("f1_mean", 0.0)]},
                            "auprc": {"mean": s_metrics.get("auprc_mean", 0.0), "std": s_metrics.get("auprc_std", 0.0), "values": [s_metrics.get("auprc_mean", 0.0)]},
                            "fpr95": {"mean": s_metrics.get("fpr_at_95_recall_mean", s_metrics.get("fpr95_mean", 0.0)), "std": s_metrics.get("fpr_at_95_recall_std", 0.0), "values": [s_metrics.get("fpr_at_95_recall_mean", 0.0)]},
                        }
                merged["scenario1"] = scen1_norm

            if "convergence" in d1:
                merged["convergence"] = d1["convergence"]
            if "heatmap" in d1:
                merged["heatmap"] = d1["heatmap"]
        except Exception as e:
            logger.warning("Error parsing %s: %s", axis1_file, e)

    if scen2_file.exists():
        found_any = True
        _validate_meta(scen2_file)
        try:
            with open(scen2_file, "r", encoding="utf-8") as f:
                d2 = json.load(f)
            if "scenario2" in d2 and isinstance(d2["scenario2"], dict) and any("top1_accuracy" in str(v) for v in d2["scenario2"].values()):
                merged["scenario2"] = d2["scenario2"]
            elif "runs" in d2:
                scen2_norm: dict[str, Any] = {}
                for r in d2.get("runs", []):
                    strat = r.get("strategy")
                    metrics = r.get("final_metrics", {})
                    if not strat or not metrics:
                        continue
                    scen2_norm.setdefault(strat, {"top1_accuracy": [], "macro_f1": [], "worst_10_recall": []})
                    scen2_norm[strat]["top1_accuracy"].append(float(metrics.get("top1_accuracy", 0.0)))
                    scen2_norm[strat]["macro_f1"].append(float(metrics.get("macro_f1", metrics.get("f1", 0.0))))
                    scen2_norm[strat]["worst_10_recall"].append(float(metrics.get("worst_10_class_recall", metrics.get("worst_10_recall", 0.0))))

                for s, m_dict in scen2_norm.items():
                    for mk in ("top1_accuracy", "macro_f1", "worst_10_recall"):
                        vals = m_dict[mk]
                        m_dict[mk] = {
                            "mean": float(np.mean(vals)) if vals else 0.0,
                            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                            "values": vals,
                        }
                merged["scenario2"] = scen2_norm
            elif "aggregated_summary" in d2:
                scen2_norm = {}
                for tt, strat_dict in d2["aggregated_summary"].items():
                    for strat, s_metrics in strat_dict.items():
                        scen2_norm[strat] = {
                            "top1_accuracy": {"mean": s_metrics.get("top1_accuracy_mean", 0.0), "std": s_metrics.get("top1_accuracy_std", 0.0), "values": [s_metrics.get("top1_accuracy_mean", 0.0)]},
                            "macro_f1": {"mean": s_metrics.get("macro_f1_mean", s_metrics.get("f1_mean", 0.0)), "std": s_metrics.get("macro_f1_std", 0.0), "values": [s_metrics.get("macro_f1_mean", 0.0)]},
                            "worst_10_recall": {"mean": s_metrics.get("worst_10_class_recall_mean", 0.0), "std": s_metrics.get("worst_10_class_recall_std", 0.0), "values": [s_metrics.get("worst_10_class_recall_mean", 0.0)]},
                        }
                merged["scenario2"] = scen2_norm
        except Exception as e:
            logger.warning("Error parsing %s: %s", scen2_file, e)

    if ablation_file.exists():
        found_any = True
        _validate_meta(ablation_file)
        try:
            with open(ablation_file, "r", encoding="utf-8") as f:
                da = json.load(f)
            if "ablation" in da:
                merged["ablation"] = da["ablation"]
            elif "runs" in da:
                ab_norm: dict[str, Any] = {}
                for r in da.get("runs", []):
                    tt = r.get("tree_type")
                    metrics = r.get("final_metrics", {})
                    key = f"{tt}_tree" if not str(tt).endswith("_tree") else str(tt)
                    ab_norm.setdefault(key, {"f1": [], "auprc": []})
                    ab_norm[key]["f1"].append(float(metrics.get("f1", 0.0)))
                    ab_norm[key]["auprc"].append(float(metrics.get("auprc", 0.0)))
                merged["ablation"] = {
                    k: {"f1": float(np.mean(v["f1"])), "auprc": float(np.mean(v["auprc"]))}
                    for k, v in ab_norm.items()
                }
            elif "aggregated_summary" in da:
                ab_norm = {}
                for tt, s_dict in da["aggregated_summary"].items():
                    for s, m in s_dict.items():
                        key = f"{tt}_tree" if not str(tt).endswith("_tree") else str(tt)
                        ab_norm[key] = {"f1": m.get("f1_mean", 0.0), "auprc": m.get("auprc_mean", 0.0)}
                merged["ablation"] = ab_norm
        except Exception as e:
            logger.warning("Error parsing %s: %s", ablation_file, e)

    if sensitivity_file.exists():
        found_any = True
        _validate_meta(sensitivity_file)
        try:
            with open(sensitivity_file, "r", encoding="utf-8") as f:
                ds = json.load(f)
            if "sensitivity" in ds:
                merged["sensitivity"] = ds["sensitivity"]
            elif "grid" in ds:
                grid = ds["grid"]
                c_vals = sorted(list({float(g["c"]) for g in grid}))
                alpha_vals = sorted(list({float(g["alpha"]) for g in grid}))
                def_alpha = 0.3 if 0.3 in alpha_vals else (alpha_vals[len(alpha_vals) // 2] if alpha_vals else 0.3)
                c_f1 = [float(next((g["f1_mean"] for g in grid if float(g["c"]) == c and float(g["alpha"]) == def_alpha), 0.0)) for c in c_vals]
                def_c = 1.0 if 1.0 in c_vals else (c_vals[len(c_vals) // 2] if c_vals else 1.0)
                alpha_f1 = [float(next((g["f1_mean"] for g in grid if float(g["c"]) == def_c and float(g["alpha"]) == a), 0.0)) for a in alpha_vals]
                merged["sensitivity"] = {
                    "c_values": c_vals,
                    "c_f1": c_f1,
                    "alpha_values": alpha_vals,
                    "alpha_f1": alpha_f1,
                }
        except Exception as e:
            logger.warning("Error parsing %s: %s", sensitivity_file, e)

    if found_any and (merged["scenario1"] or merged["scenario2"]):
        if expected_mode == ExecutionMode.SMOKE:
            merged["is_smoke"] = True
        return merged

    raise FileNotFoundError(
        f"Real benchmark results not found in {res_path}. "
        "Automatic synthetic fallback is prohibited in REAL mode. "
        "Ensure tier1_benchmark_summary.json or component benchmark results are produced first."
    )


def load_or_synthesize_benchmark_results(
    results_dir: str | Path = "results",
    seed: int = 42,
    config: Optional[ExperimentConfig] = None,
) -> dict[str, Any]:
    """Load benchmark results from JSON files in results_dir, or synthesize deterministic 5-seed data.

    In ExecutionMode.REAL, raises FileNotFoundError if real benchmark results are missing.
    In ExecutionMode.SMOKE, returns deterministic synthetic distributions flagged with is_smoke=True.

    Args:
        results_dir: Directory containing benchmark JSON summaries.
        seed: Random seed for deterministic synthetic fallback.
        config: Optional ExperimentConfig specifying execution mode.

    Returns:
        Structured dictionary matching Scenario 1, Scenario 2, and ablation schemas.
    """
    mode = config.execution_mode if config is not None else get_active_session_mode()

    if mode == ExecutionMode.REAL:
        return load_real_benchmark_results(results_dir, expected_mode=ExecutionMode.REAL)

    if mode == ExecutionMode.SMOKE:
        try:
            return load_real_benchmark_results(results_dir, expected_mode=ExecutionMode.SMOKE)
        except Exception:
            pass
    else:
        # Default strictly to REAL when unconfigured; fail with FileNotFoundError if results are missing.
        # NEVER infer SMOKE from absent files.
        return load_real_benchmark_results(results_dir, expected_mode=ExecutionMode.REAL)

    logger.warning("SMOKE mode: Generating deterministic synthetic benchmark distributions.")

    # Deterministic synthesis of 5-seed benchmark distributions
    import numpy as np

    rng = np.random.default_rng(seed)

    strategies = ["uniform", "ohem", "mcts_strict", "mcts_enhanced"]
    imbalance_ratios = [10, 50, 100]

    # Scenario 1 base means (f1, auprc, fpr95)
    base_metrics = {
        "uniform": {
            10: (0.421, 0.405, 0.352),
            50: (0.283, 0.252, 0.521),
            100: (0.184, 0.151, 0.684),
        },
        "ohem": {
            10: (0.485, 0.462, 0.301),
            50: (0.342, 0.315, 0.463),
            100: (0.226, 0.194, 0.612),
        },
        "mcts_strict": {
            10: (0.512, 0.501, 0.264),
            50: (0.384, 0.362, 0.415),
            100: (0.261, 0.235, 0.551),
        },
        "mcts_enhanced": {
            10: (0.564, 0.552, 0.213),
            50: (0.431, 0.415, 0.342),
            100: (0.312, 0.286, 0.473),
        },
    }

    scenario1_results: dict[str, Any] = {}
    for strat in strategies:
        scenario1_results[strat] = {}
        for rho in imbalance_ratios:
            m_f1, m_auprc, m_fpr = base_metrics[strat][rho]
            f1_seeds = (m_f1 + rng.normal(0, 0.012, 5)).clip(0, 1).tolist()
            auprc_seeds = (m_auprc + rng.normal(0, 0.015, 5)).clip(0, 1).tolist()
            fpr_seeds = (m_fpr + rng.normal(0, 0.018, 5)).clip(0, 1).tolist()
            prec_seeds = [round(f * 1.05, 4) for f in f1_seeds]
            rec_seeds = [round(f * 0.95, 4) for f in f1_seeds]

            scenario1_results[strat][str(rho)] = {
                "f1": {"mean": float(np.mean(f1_seeds)), "std": float(np.std(f1_seeds)), "values": f1_seeds},
                "auprc": {"mean": float(np.mean(auprc_seeds)), "std": float(np.std(auprc_seeds)), "values": auprc_seeds},
                "fpr95": {"mean": float(np.mean(fpr_seeds)), "std": float(np.std(fpr_seeds)), "values": fpr_seeds},
                "precision": {"mean": float(np.mean(prec_seeds)), "std": float(np.std(prec_seeds)), "values": prec_seeds},
                "recall": {"mean": float(np.mean(rec_seeds)), "std": float(np.std(rec_seeds)), "values": rec_seeds},
            }

    # Scenario 2 (100-way multi-class)
    scen2_base = {
        "uniform": (0.642, 0.628, 0.382),
        "ohem": (0.665, 0.651, 0.415),
        "mcts_strict": (0.678, 0.669, 0.442),
        "mcts_enhanced": (0.704, 0.698, 0.495),
    }
    scenario2_results: dict[str, Any] = {}
    for strat in strategies:
        acc, f1_m, rec10 = scen2_base[strat]
        acc_s = (acc + rng.normal(0, 0.008, 5)).clip(0, 1).tolist()
        f1_s = (f1_m + rng.normal(0, 0.009, 5)).clip(0, 1).tolist()
        rec10_s = (rec10 + rng.normal(0, 0.014, 5)).clip(0, 1).tolist()
        scenario2_results[strat] = {
            "top1_accuracy": {"mean": float(np.mean(acc_s)), "std": float(np.std(acc_s)), "values": acc_s},
            "macro_f1": {"mean": float(np.mean(f1_s)), "std": float(np.std(f1_s)), "values": f1_s},
            "worst_10_recall": {"mean": float(np.mean(rec10_s)), "std": float(np.std(rec10_s)), "values": rec10_s},
        }

    # Ablation study: Tree geometry, Q-norm, Dual stream, Leaf expansion
    ablation_results = {
        "semantic_tree": {"f1": 0.548, "std": 0.011, "auprc": 0.536, "std_auprc": 0.013},
        "confusion_ward_tree": {"f1": 0.564, "std": 0.009, "auprc": 0.552, "std_auprc": 0.012},
        "q_norm_off": {"f1": 0.512, "std": 0.015, "auprc": 0.501, "std_auprc": 0.016},
        "q_norm_on": {"f1": 0.564, "std": 0.009, "auprc": 0.552, "std_auprc": 0.012},
        "dual_stream_0.0": {"f1": 0.508, "std": 0.018, "auprc": 0.498, "std_auprc": 0.019},
        "dual_stream_0.5": {"f1": 0.564, "std": 0.009, "auprc": 0.552, "std_auprc": 0.012},
        "dual_stream_1.0": {"f1": 0.421, "std": 0.014, "auprc": 0.405, "std_auprc": 0.015},
        "dynamic_expansion_off": {"f1": 0.564, "std": 0.009, "auprc": 0.552, "std_auprc": 0.012},
        "dynamic_expansion_on": {"f1": 0.581, "std": 0.008, "auprc": 0.572, "std_auprc": 0.010},
    }

    # Convergence trajectories (10 eval checkpoints)
    steps = [0, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 10000]
    convergence = {"steps": steps}
    for strat in strategies:
        final_f1 = scenario1_results[strat]["10"]["f1"]["mean"]
        final_loss = 0.12 if "enhanced" in strat else (0.16 if "strict" in strat else (0.21 if "ohem" in strat else 0.28))
        f1_curve = [round(float(final_f1 * (1.0 - np.exp(-0.0006 * s)) + rng.normal(0, 0.008)), 4) for s in steps]
        f1_curve[0] = 0.05
        loss_curve = [round(float(final_loss + (0.75 - final_loss) * np.exp(-0.0005 * s) + rng.normal(0, 0.006)), 4) for s in steps]
        convergence[f"{strat}_f1"] = f1_curve
        convergence[f"{strat}_loss"] = loss_curve

    # Superclass visitation distribution (20 superclasses across 10 steps)
    num_superclasses = 20
    heatmap_matrix = np.full((num_superclasses, len(steps)), 1.0 / num_superclasses)
    for idx, s in enumerate(steps):
        factor = float(min(1.0, s / 5000.0))
        dist = np.ones(num_superclasses) * (1.0 - 0.7 * factor)
        dist[1] += 3.5 * factor
        dist[4] += 2.8 * factor
        dist[7] += 2.1 * factor
        dist[12] += 1.8 * factor
        dist = dist / dist.sum()
        heatmap_matrix[:, idx] = dist

    # Hyperparameter grids
    c_values = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    c_f1 = [0.521, 0.548, 0.564, 0.558, 0.541, 0.518]
    alpha_values = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
    alpha_f1 = [0.529, 0.547, 0.564, 0.556, 0.538, 0.512]

    return {
        "is_smoke": True,
        "scenario1": scenario1_results,
        "scenario2": scenario2_results,
        "ablation": ablation_results,
        "convergence": convergence,
        "heatmap": {
            "steps": steps,
            "superclasses": [f"SC-{i:02d}" for i in range(num_superclasses)],
            "matrix": heatmap_matrix.tolist(),
        },
        "sensitivity": {
            "c_values": c_values,
            "c_f1": c_f1,
            "alpha_values": alpha_values,
            "alpha_f1": alpha_f1,
        },
    }


def validate_artifact_completeness(
    data: dict[str, Any],
    artifact_type: str,
    strict_strategies: bool = False,
) -> None:
    """Validate completeness of benchmark results prior to artifact generation.

    Args:
        data: Benchmark results dictionary.
        artifact_type: Type of artifact to validate ('scenario1_table', 'scenario2_table',
            'ablation_table', 'convergence_plot', 'visitation_heatmap', 'sensitivity_plot').
        strict_strategies: Whether all canonical strategies are strictly required.

    Raises:
        ValueError: If required data keys, strategies, or metrics are missing or incomplete.
    """
    if not isinstance(data, dict):
        raise ValueError("Benchmark data must be a dictionary.")

    is_smoke = bool(data.get("is_smoke", False))
    require_all_strats = strict_strategies or (not is_smoke and not data.get("partial_allowed", False))

    if artifact_type == "scenario1_table":
        scen1 = data.get("scenario1")
        if not scen1 or not isinstance(scen1, dict):
            raise ValueError("Scenario 1 results dictionary is empty or missing.")
        canonical = ["uniform", "ohem", "mcts_strict", "mcts_enhanced"]
        if require_all_strats:
            missing = [s for s in canonical if s not in scen1]
            if missing:
                raise ValueError(f"Scenario 1 results missing required strategies: {missing}")
        for strat, r_dict in scen1.items():
            if not isinstance(r_dict, dict) or not r_dict:
                raise ValueError(f"Scenario 1 strategy '{strat}' has no imbalance ratio data.")
            for rho, m_dict in r_dict.items():
                for mk in ("f1", "auprc", "fpr95"):
                    if mk not in m_dict or "mean" not in m_dict[mk] or "std" not in m_dict[mk]:
                        raise ValueError(f"Scenario 1 strategy '{strat}' missing metric '{mk}' at rho={rho}")

    elif artifact_type == "scenario2_table":
        scen2 = data.get("scenario2")
        if not scen2 or not isinstance(scen2, dict):
            raise ValueError("Scenario 2 results dictionary is empty or missing.")
        if require_all_strats:
            if "uniform" not in scen2 or "mcts_enhanced" not in scen2:
                raise ValueError("Scenario 2 results missing required strategies ('uniform', 'mcts_enhanced').")
        for strat, m_dict in scen2.items():
            for mk in ("top1_accuracy", "macro_f1", "worst_10_recall"):
                if mk not in m_dict or "mean" not in m_dict[mk] or "std" not in m_dict[mk]:
                    raise ValueError(f"Scenario 2 strategy '{strat}' missing metric '{mk}'")

    elif artifact_type == "ablation_table":
        ablation = data.get("ablation")
        if not ablation or not isinstance(ablation, dict):
            raise ValueError("Ablation results dictionary is empty or missing.")
        for k, v in ablation.items():
            if not isinstance(v, dict) or "f1" not in v or "auprc" not in v:
                raise ValueError(f"Ablation configuration '{k}' missing required 'f1' or 'auprc' metric: {v}")

    elif artifact_type == "convergence_plot":
        conv = data.get("convergence")
        if not is_smoke and (not conv or not isinstance(conv, dict) or "steps" not in conv or len(conv["steps"]) < 2):
            raise ValueError("Convergence curve metrics missing or steps array insufficient for REAL publication figure.")

    elif artifact_type == "visitation_heatmap":
        heat = data.get("heatmap")
        if not is_smoke and (not heat or not isinstance(heat, dict) or "matrix" not in heat or not heat["matrix"]):
            raise ValueError("Tree visitation heatmap matrix missing or empty for REAL publication figure.")

    elif artifact_type == "sensitivity_plot":
        sens = data.get("sensitivity")
        if not is_smoke and (not sens or not isinstance(sens, dict) or not sens.get("c_values") or not sens.get("alpha_values")):
            raise ValueError("Hyperparameter sensitivity metrics ('c_values', 'alpha_values') missing for REAL publication figure.")

    else:
        raise ValueError(f"Unknown artifact type: '{artifact_type}'")


def _ensure_dir(path: Path) -> Path:
    """Ensure directory exists and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_convergence_plot(
    data: dict[str, Any],
    output_dir: str | Path = "paper/figures",
    watermark: Optional[str] = None,
) -> tuple[Path, Path]:
    """Generate convergence curves figure (PDF vector and PNG raster).

    Args:
        data: Benchmark results dictionary containing 'convergence'.
        output_dir: Directory where figures will be saved.
        watermark: Optional watermark banner text for smoke test outputs.

    Returns:
        Tuple of (pdf_path, png_path).
    """
    validate_artifact_completeness(data, "convergence_plot")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = _ensure_dir(Path(output_dir))
    pdf_path = out / "fig_convergence_curves.pdf"
    png_path = out / "fig_convergence_curves.png"

    conv = data.get("convergence", {})
    steps = conv.get("steps", [0, 1000, 2000, 5000, 10000])

    palette = {
        "uniform": ("#7f7f7f", "--", "Uniform"),
        "ohem": ("#ff7f0e", "-.", "OHEM"),
        "mcts_strict": ("#1f77b4", ":", "Strict GiantHunter"),
        "mcts_enhanced": ("#2ca02c", "-", "Enhanced MCTS-HNM (Ours)"),
    }

    if not data.get("is_smoke") and not any(f"{k}_f1" in conv for k in palette):
        raise ValueError("Convergence curve metrics missing for REAL publication figure.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)

    # Subplot 1: Test Minority F1-Score
    for key, (color, style, label) in palette.items():
        f1_key = f"{key}_f1"
        if f1_key in conv:
            vals = conv[f1_key]
            ax1.plot(steps, vals, label=label, color=color, linestyle=style, linewidth=2.0)

    ax1.set_title(r"(a) Test Minority F1-Score ($\rho=1:10$)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Training Steps", fontsize=10)
    ax1.set_ylabel("F1-Score", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_ylim(0.0, 0.65)
    ax1.legend(loc="lower right", frameon=True, fontsize=9)

    # Subplot 2: Loss Convergence
    for key, (color, style, label) in palette.items():
        loss_key = f"{key}_loss"
        if loss_key in conv:
            vals = conv[loss_key]
            ax2.plot(steps, vals, label=label, color=color, linestyle=style, linewidth=2.0)

    ax2.set_title("(b) Binary Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Training Steps", fontsize=10)
    ax2.set_ylabel("Loss", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_ylim(0.0, 0.8)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)

    if watermark is None and data.get("is_smoke"):
        watermark = "SMOKE TEST — NOT FOR PUBLICATION"

    if watermark:
        fig.suptitle(f"[{watermark}]", fontsize=11, color="crimson", fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info("Saved convergence plots to %s and %s", pdf_path, png_path)
    return pdf_path, png_path


def generate_tree_visitation_heatmap(
    data: dict[str, Any],
    output_dir: str | Path = "paper/figures",
    watermark: Optional[str] = None,
) -> tuple[Path, Path]:
    """Generate tree visitation heatmap figure showing exploration to exploitation.

    Args:
        data: Benchmark results dictionary containing 'heatmap'.
        output_dir: Directory where figures will be saved.
        watermark: Optional watermark banner text for smoke test outputs.

    Returns:
        Tuple of (pdf_path, png_path).
    """
    validate_artifact_completeness(data, "visitation_heatmap")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out = _ensure_dir(Path(output_dir))
    pdf_path = out / "fig_tree_visitation_heatmap.pdf"
    png_path = out / "fig_tree_visitation_heatmap.png"

    heat = data.get("heatmap", {})
    if not data.get("is_smoke") and ("matrix" not in heat or not heat["matrix"]):
        raise ValueError("Tree visitation heatmap matrix missing for REAL publication figure.")
    raw_matrix = heat.get("matrix")
    if raw_matrix is None or (isinstance(raw_matrix, (list, np.ndarray)) and len(raw_matrix) == 0):
        matrix = np.full((20, 10), 1.0 / 20.0)
    else:
        matrix = np.array(raw_matrix)
    if matrix.ndim < 2 or matrix.size == 0:
        matrix = np.full((20, 10), 1.0 / 20.0)
    steps = heat.get("steps", list(range(matrix.shape[1])))
    superclasses = heat.get("superclasses", [f"SC-{i:02d}" for i in range(matrix.shape[0])])

    # Show top 10 most visited superclasses for readability
    top_indices = np.argsort(matrix[:, -1])[::-1][:10]
    sub_matrix = matrix[top_indices, :]
    sub_names = [superclasses[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    im = ax.imshow(sub_matrix, cmap="YlGnBu", aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], fontsize=9)
    ax.set_yticks(range(len(sub_names)))
    ax.set_yticklabels(sub_names, fontsize=9)

    ax.set_xlabel("Training Step", fontsize=10)
    ax.set_ylabel("CIFAR-100 Superclass (Sorted by Final Visitation)", fontsize=10)
    ax.set_title("MCTS Superclass Visitation Density Transition ($p_k$)", fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Visitation Probability $p_k$", fontsize=9)

    if watermark is None and data.get("is_smoke"):
        watermark = "SMOKE TEST — NOT FOR PUBLICATION"

    if watermark:
        fig.suptitle(f"[{watermark}]", fontsize=11, color="crimson", fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info("Saved heatmap plots to %s and %s", pdf_path, png_path)
    return pdf_path, png_path


def generate_hyperparameter_sensitivity_plot(
    data: dict[str, Any],
    output_dir: str | Path = "paper/figures",
    watermark: Optional[str] = None,
) -> tuple[Path, Path]:
    """Generate hyperparameter sensitivity plot for exploration constant c and momentum alpha.

    Args:
        data: Benchmark results dictionary containing 'sensitivity'.
        output_dir: Directory where figures will be saved.
        watermark: Optional watermark banner text for smoke test outputs.

    Returns:
        Tuple of (pdf_path, png_path).
    """
    validate_artifact_completeness(data, "sensitivity_plot")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = _ensure_dir(Path(output_dir))
    pdf_path = out / "fig_hyperparameter_sensitivity.pdf"
    png_path = out / "fig_hyperparameter_sensitivity.png"

    sens = data.get("sensitivity", {})
    if not data.get("is_smoke") and (not sens or not sens.get("c_values")):
        raise ValueError("Hyperparameter sensitivity metrics missing for REAL publication figure.")
    c_vals = sens.get("c_values", [0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
    c_f1 = sens.get("c_f1", [0.52, 0.55, 0.56, 0.55, 0.54, 0.51])
    alpha_vals = sens.get("alpha_values", [0.05, 0.1, 0.3, 0.5, 0.7, 0.9])
    alpha_f1 = sens.get("alpha_f1", [0.53, 0.55, 0.56, 0.55, 0.53, 0.51])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0), dpi=300)

    ax1.plot(c_vals, c_f1, marker="o", color="#1f77b4", linewidth=2.0, markersize=6)
    ax1.axvline(1.0, color="gray", linestyle="--", alpha=0.7, label="Default ($c=1.0$)")
    ax1.set_title("(a) Exploration Constant $c$", fontsize=11, fontweight="bold")
    ax1.set_xlabel("UCB1 Exploration Factor ($c$)", fontsize=10)
    ax1.set_ylabel("Minority F1-Score", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=9)

    ax2.plot(alpha_vals, alpha_f1, marker="s", color="#2ca02c", linewidth=2.0, markersize=6)
    ax2.axvline(0.3, color="gray", linestyle="--", alpha=0.7, label=r"Default ($\alpha=0.3$)")
    ax2.set_title(r"(b) Momentum Factor $\alpha$", fontsize=11, fontweight="bold")
    ax2.set_xlabel(r"Leaf Momentum Update Rate ($\alpha$)", fontsize=10)
    ax2.set_ylabel("Minority F1-Score", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=9)

    if watermark is None and data.get("is_smoke"):
        watermark = "SMOKE TEST — NOT FOR PUBLICATION"

    if watermark:
        fig.suptitle(f"[{watermark}]", fontsize=11, color="crimson", fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info("Saved hyperparameter sensitivity plots to %s and %s", pdf_path, png_path)
    return pdf_path, png_path


def generate_scenario1_table(
    data: dict[str, Any],
    output_dir: str | Path = "paper/tables",
) -> Path:
    """Generate LaTeX booktabs table for Scenario 1 (Imbalanced Binary One-vs-Rest).

    Args:
        data: Benchmark results dictionary containing 'scenario1'.
        output_dir: Directory where the table will be saved.

    Returns:
        Path to the saved .tex file.
    """
    validate_artifact_completeness(data, "scenario1_table")
    from src.utils.statistics import compute_paired_statistical_tests

    out = _ensure_dir(Path(output_dir))
    tex_path = out / "tab_scenario1_imbalanced.tex"

    scen1 = data.get("scenario1", {})
    if not scen1:
        raise ValueError("Scenario 1 results dictionary is empty or missing.")

    cand_strategies = [
        ("uniform", "Uniform Random"),
        ("ohem", "OHEM (Shrivastava et al.)"),
        ("mcts_strict", "Strict GiantHunter (Qu et al.)"),
        ("mcts_enhanced", "\\textbf{Enhanced MCTS-HNM (Ours)}"),
    ]
    strategies = [
        (k, label) for k, label in cand_strategies if k in scen1
    ]
    if not strategies:
        raise ValueError("Scenario 1 results dictionary has no recognized strategies.")

    cand_ratios = [("10", "1:10"), ("50", "1:50"), ("100", "1:100")]
    imbalance_ratios = [
        (rk, rl) for rk, rl in cand_ratios
        if any(rk in scen1.get(sk, {}) for sk, _ in strategies)
    ]
    if not imbalance_ratios:
        imbalance_ratios = [("10", "1:10")]

    lines = [
        r"% Auto-generated by src/utils/paper_artifacts.py",
        r"% SMOKE TEST — NOT VALID FOR SCIENTIFIC PUBLICATION" if data.get("is_smoke") else "",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Scenario 1: Imbalanced Binary One-vs-Rest evaluation on CIFAR-100 across 5 random seeds ($N=5$). "
        r"Reporting mean $\pm$ standard deviation for F1-Score, AUPRC, and False Positive Rate at 95\% recall (FPR@95). "
        r"$p$-values computed via paired Student's $t$-test against Enhanced MCTS-HNM.}",
        r"\label{tab:scenario1_imbalanced}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Imbalance ($\rho$) & Negative Mining Strategy & Minority F1-Score (\%) & AUPRC & FPR@95 ($\downarrow$) & $p$-value ($t$-test) \\",
        r"\midrule",
    ]
    lines = [l for l in lines if l]

    for rho_key, rho_label in imbalance_ratios:
        # Reference baseline: enhanced
        enhanced_entry = scen1.get("mcts_enhanced", {}).get(rho_key)
        enhanced_vals = (
            enhanced_entry["f1"].get("values")
            if enhanced_entry and "f1" in enhanced_entry and "values" in enhanced_entry["f1"]
            else None
        )

        for strat_key, strat_label in strategies:
            entry = scen1.get(strat_key, {}).get(rho_key)
            if not entry:
                raise ValueError(f"Scenario 1 results missing entry for strategy='{strat_key}' at rho={rho_key}")
            for m_key in ("f1", "auprc", "fpr95"):
                if m_key not in entry or "mean" not in entry[m_key] or "std" not in entry[m_key]:
                    raise ValueError(f"Scenario 1 strategy '{strat_key}' missing metric '{m_key}' at rho={rho_key}")
            f1_mean = float(entry["f1"]["mean"]) * 100.0
            f1_std = float(entry["f1"]["std"]) * 100.0
            auprc_mean = float(entry["auprc"]["mean"])
            auprc_std = float(entry["auprc"]["std"])
            fpr_mean = float(entry["fpr95"]["mean"])
            fpr_std = float(entry["fpr95"]["std"])

            strat_vals = entry["f1"].get("values")
            if not strat_vals:
                raise ValueError(f"Scenario 1 strategy '{strat_key}' missing seed values for rho={rho_key}")
            if strat_key == "mcts_enhanced":
                p_val_str = "--- (Reference)"
            elif enhanced_vals and len(strat_vals) >= 2 and len(enhanced_vals) >= 2 and len(strat_vals) == len(enhanced_vals):
                stat_res = compute_paired_statistical_tests(strat_vals, enhanced_vals)
                p_val = stat_res["t_pvalue"]
                p_val_str = f"$p < 0.001$" if p_val < 0.001 else f"$p = {p_val:.3f}$"
            else:
                p_val_str = "N/A"

            lines.append(
                f"  {rho_label} & {strat_label} & {f1_mean:.1f} $\\pm$ {f1_std:.1f} & "
                f"{auprc_mean:.3f} $\\pm$ {auprc_std:.3f} & {fpr_mean:.3f} $\\pm$ {fpr_std:.3f} & {p_val_str} \\\\"
            )
        if rho_key != "100":
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Saved Scenario 1 table to %s", tex_path)
    return tex_path


def generate_scenario2_table(
    data: dict[str, Any],
    output_dir: str | Path = "paper/tables",
) -> Path:
    """Generate LaTeX booktabs table for Scenario 2 (100-way Multi-Class Curriculum).

    Args:
        data: Benchmark results dictionary containing 'scenario2'.
        output_dir: Directory where the table will be saved.

    Returns:
        Path to the saved .tex file.
    """
    validate_artifact_completeness(data, "scenario2_table")
    from src.utils.statistics import compute_paired_statistical_tests

    out = _ensure_dir(Path(output_dir))
    tex_path = out / "tab_scenario2_multiclass.tex"

    scen2 = data.get("scenario2", {})
    if not scen2:
        raise ValueError("Scenario 2 results dictionary is empty or missing.")

    cand_strategies = [
        ("uniform", "Uniform Random"),
        ("ohem", "OHEM (Shrivastava et al.)"),
        ("mcts_strict", "Strict GiantHunter (Qu et al.)"),
        ("mcts_enhanced", "\\textbf{Enhanced MCTS-HNM (Ours)}"),
    ]
    strategies = [
        (k, label) for k, label in cand_strategies if k in scen2
    ]
    if not strategies:
        raise ValueError("Scenario 2 results dictionary has no recognized strategies.")

    enhanced_entry = scen2.get("mcts_enhanced")
    enhanced_acc_vals = (
        enhanced_entry["top1_accuracy"].get("values")
        if enhanced_entry and "top1_accuracy" in enhanced_entry
        else None
    )

    lines = [
        r"% Auto-generated by src/utils/paper_artifacts.py",
        r"% SMOKE TEST — NOT VALID FOR SCIENTIFIC PUBLICATION" if data.get("is_smoke") else "",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Scenario 2: 100-way multi-class curriculum classification on CIFAR-100 across 5 random seeds ($N=5$). "
        r"Evaluating Top-1 Accuracy, Macro-Averaged F1-Score, and Worst-10-Class Recall.}",
        r"\label{tab:scenario2_multiclass}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Strategy & Top-1 Accuracy (\%) & Macro-F1 (\%) & Worst-10 Recall (\%) & $p$-value ($t$-test) \\",
        r"\midrule",
    ]
    lines = [l for l in lines if l]

    for strat_key, strat_label in strategies:
        entry = scen2.get(strat_key)
        if not entry:
            raise ValueError(f"Scenario 2 results missing entry for strategy='{strat_key}'")
        for m_key in ("top1_accuracy", "macro_f1", "worst_10_recall"):
            if m_key not in entry or "mean" not in entry[m_key] or "std" not in entry[m_key]:
                raise ValueError(f"Scenario 2 strategy '{strat_key}' missing metric '{m_key}'")
        acc_mean = float(entry["top1_accuracy"]["mean"]) * 100.0
        acc_std = float(entry["top1_accuracy"]["std"]) * 100.0
        f1_mean = float(entry["macro_f1"]["mean"]) * 100.0
        f1_std = float(entry["macro_f1"]["std"]) * 100.0
        rec_mean = float(entry["worst_10_recall"]["mean"]) * 100.0
        rec_std = float(entry["worst_10_recall"]["std"]) * 100.0

        strat_acc_vals = entry["top1_accuracy"].get("values")
        if not strat_acc_vals:
            raise ValueError(f"Scenario 2 strategy '{strat_key}' missing top1_accuracy seed values")
        if strat_key == "mcts_enhanced":
            p_val_str = "--- (Ref)"
        elif enhanced_acc_vals and len(strat_acc_vals) >= 2 and len(enhanced_acc_vals) >= 2 and len(strat_acc_vals) == len(enhanced_acc_vals):
            stat_res = compute_paired_statistical_tests(strat_acc_vals, enhanced_acc_vals)
            p_val = stat_res["t_pvalue"]
            p_val_str = f"$p < 0.001$" if p_val < 0.001 else f"$p = {p_val:.3f}$"
        else:
            p_val_str = "N/A"

        lines.append(
            f"  {strat_label} & {acc_mean:.1f} $\\pm$ {acc_std:.1f} & "
            f"{f1_mean:.1f} $\\pm$ {f1_std:.1f} & {rec_mean:.1f} $\\pm$ {rec_std:.1f} & {p_val_str} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Saved Scenario 2 table to %s", tex_path)
    return tex_path


def generate_ablation_table(
    data: dict[str, Any],
    output_dir: str | Path = "paper/tables",
) -> Path:
    """Generate LaTeX booktabs table for ablation studies.

    Args:
        data: Benchmark results dictionary containing 'ablation'.
        output_dir: Directory where the table will be saved.

    Returns:
        Path to the saved .tex file.
    """
    validate_artifact_completeness(data, "ablation_table")
    out = _ensure_dir(Path(output_dir))
    tex_path = out / "tab_ablation_study.tex"

    ablation = data.get("ablation", {})
    if not ablation:
        raise ValueError("Ablation results dictionary is empty or missing.")

    all_configs = [
        ("Hierarchy Geometry", "semantic_tree", "Native Semantic Tree (20 superclasses)"),
        ("Hierarchy Geometry", "confusion_ward_tree", "Ward Confusion Tree (Empirical)"),
        ("UCB1 Normalization", "q_norm_off", "Raw Unnormalized Q (Strict GiantHunter)"),
        ("UCB1 Normalization", "q_norm_on", "Min-Max Sibling Q-Normalization"),
        ("Dual-Stream Batch", "dual_stream_0.0", r"100\% MCTS Negatives ($\beta=0.0$)"),
        ("Dual-Stream Batch", "dual_stream_0.5", r"50/50 Dual-Stream Hybrid ($\beta=0.5$)"),
        ("Dual-Stream Batch", "dual_stream_1.0", r"100\% Uniform Random ($\beta=1.0$)"),
        ("Leaf Granularity", "dynamic_expansion_off", "Fixed 100 Class Leaves (P1 MVP)"),
        ("Leaf Granularity", "dynamic_expansion_on", "Dynamic Entropy Sub-Clustering (P2)"),
    ]

    rows = []
    for axis, cfg_key, cfg_label in all_configs:
        if cfg_key in ablation:
            cfg_data = ablation[cfg_key]
            if not isinstance(cfg_data, dict) or "f1" not in cfg_data or "auprc" not in cfg_data:
                raise ValueError(
                    f"Missing required ablation metrics ('f1', 'auprc') for '{cfg_key}': {cfg_data}"
                )
            f1_val = float(cfg_data["f1"])
            auprc_val = float(cfg_data["auprc"])
            rows.append((axis, cfg_label, f1_val, auprc_val))

    if not rows:
        raise ValueError("Ablation results dictionary has no recognized configurations.")

    baseline_f1 = float(ablation["q_norm_off"]["f1"]) if "q_norm_off" in ablation and "f1" in ablation["q_norm_off"] else rows[0][2]

    lines = [
        r"% Auto-generated by src/utils/paper_artifacts.py",
        r"% SMOKE TEST — NOT VALID FOR SCIENTIFIC PUBLICATION" if data.get("is_smoke") else "",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Ablation study on CIFAR-100 ($\rho=1:10$) isolating tree hierarchy structure, $Q$-normalization, "
        r"Dual-Stream sampling ratio, and dynamic prediction-entropy leaf sub-clustering.}",
        r"\label{tab:ablation_study}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Ablation Axis & Configuration & F1-Score & AUPRC & $\Delta$ F1 vs. Strict \\",
        r"\midrule",
    ]
    lines = [l for l in lines if l]

    last_axis = ""
    for axis, config, f1, auprc in rows:
        axis_label = axis if axis != last_axis else ""
        last_axis = axis
        delta = (f1 - baseline_f1) * 100.0
        delta_str = f"+{delta:.1f}\\%" if delta >= 0 else f"{delta:.1f}\\%"
        lines.append(f"  {axis_label} & {config} & {f1:.3f} & {auprc:.3f} & {delta_str} \\\\")
        if axis in ["Hierarchy Geometry", "UCB1 Normalization", "Dual-Stream Batch"] and config == rows[rows.index((axis, config, f1, auprc))][1]:
            # Add small separator if next row changes axis
            idx = rows.index((axis, config, f1, auprc))
            if idx + 1 < len(rows) and rows[idx + 1][0] != axis:
                lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Saved ablation table to %s", tex_path)
    return tex_path


def generate_artifact_manifest(
    output_dir: str | Path,
    artifacts: list[Path | str],
    source_files: list[Path | str],
    mode: ExecutionMode | str,
    git_commit: Optional[str] = None,
) -> Path:
    """Generate artifact_manifest.json with SHA-256 hashes and provenance metadata.

    Args:
        output_dir: Directory where the manifest will be written.
        artifacts: List of generated artifact file paths.
        source_files: List of input data file paths.
        mode: ExecutionMode (REAL or SMOKE).
        git_commit: Optional git commit hash.

    Returns:
        Path to the generated artifact_manifest.json.
    """
    out_dir = _ensure_dir(Path(output_dir))
    manifest_path = out_dir / "artifact_manifest.json"

    mode_str = mode.value if isinstance(mode, ExecutionMode) else str(mode).lower()
    commit_str = git_commit or _get_git_commit(out_dir)

    art_entries = []
    for art in artifacts:
        p = Path(art)
        if p.exists():
            art_entries.append({
                "file": str(p),
                "name": p.name,
                "sha256": _compute_sha256(p),
                "bytes": p.stat().st_size if p.is_file() else 0,
            })

    src_entries = []
    for src in source_files:
        p = Path(src)
        if p.exists():
            src_entries.append({
                "file": str(p),
                "name": p.name,
                "sha256": _compute_sha256(p),
                "bytes": p.stat().st_size if p.is_file() else 0,
            })

    manifest_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution_mode": mode_str,
        "git_commit": commit_str,
        "artifacts": art_entries,
        "source_files": src_entries,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info("Saved artifact manifest to %s (%d artifacts)", manifest_path, len(art_entries))
    return manifest_path


def generate_all_paper_artifacts(
    results_dir: str | Path = "results",
    paper_dir: str | Path = "paper",
    metadata_path: str | Path = "paper_metadata.yaml",
    config: Optional[ExperimentConfig] = None,
) -> dict[str, Any]:
    """High-level pipeline: generate all figures and tables into isolated paper directories.

    Args:
        results_dir: Path to directory containing experiment JSON files.
        paper_dir: Root paper directory.
        metadata_path: Path to paper_metadata.yaml.
        config: Optional ExperimentConfig for mode-segregated paths and execution rules.

    Returns:
        Dictionary mapping 'figures', 'tables', and 'manifest' to generated file Paths.
    """
    if config is not None:
        mode = config.execution_mode
        paths = get_isolated_paths(config)
        is_default_paper = (
            str(paper_dir) == "paper"
            or Path(paper_dir).resolve() == (paths.repo_root / "paper").resolve()
        )
        if is_default_paper:
            fig_dir = paths.paper_figures_dir
            tab_dir = paths.paper_tables_dir
            manifest_dir = fig_dir.parent
        else:
            p_dir = _ensure_dir(Path(paper_dir))
            fig_dir = _ensure_dir(p_dir / "figures")
            tab_dir = _ensure_dir(p_dir / "tables")
            manifest_dir = p_dir
        res_dir = paths.results_dir if str(results_dir) == "results" else Path(results_dir)
    else:
        active = get_active_session_mode()
        mode = active if active is not None else ExecutionMode.REAL
        p_dir = _ensure_dir(Path(paper_dir))
        fig_dir = _ensure_dir(p_dir / "figures")
        tab_dir = _ensure_dir(p_dir / "tables")
        manifest_dir = p_dir
        res_dir = Path(results_dir)

    meta = load_paper_metadata(metadata_path)
    data = load_or_synthesize_benchmark_results(results_dir=res_dir, config=config)

    watermark = "SMOKE TEST — NOT FOR PUBLICATION" if (mode == ExecutionMode.SMOKE or data.get("is_smoke")) else None

    # 1. Figures
    c_pdf, c_png = generate_convergence_plot(data, output_dir=fig_dir, watermark=watermark)
    h_pdf, h_png = generate_tree_visitation_heatmap(data, output_dir=fig_dir, watermark=watermark)
    s_pdf, s_png = generate_hyperparameter_sensitivity_plot(data, output_dir=fig_dir, watermark=watermark)

    # 2. Tables
    t_scen1 = generate_scenario1_table(data, output_dir=tab_dir)
    t_scen2 = generate_scenario2_table(data, output_dir=tab_dir)
    t_ablation = generate_ablation_table(data, output_dir=tab_dir)

    all_artifacts = [c_pdf, c_png, h_pdf, h_png, s_pdf, s_png, t_scen1, t_scen2, t_ablation]

    # 3. Provenance Source Files
    source_candidates = [
        res_dir / "tier1_benchmark_summary.json",
        res_dir / "tier1_axis1_samplers.json",
        res_dir / "tier1_scenario2_curriculum.json",
        res_dir / "tier1_axis2_tree_ablation.json",
        res_dir / "tier1_hyperparameter_sensitivity.json",
        Path(metadata_path),
    ]
    source_files = [p for p in source_candidates if p.exists()]

    # 4. Manifest
    manifest_path = generate_artifact_manifest(
        output_dir=manifest_dir,
        artifacts=all_artifacts,
        source_files=source_files,
        mode=mode,
    )

    return {
        "figures": [c_pdf, c_png, h_pdf, h_png, s_pdf, s_png],
        "tables": [t_scen1, t_scen2, t_ablation],
        "manifest": manifest_path,
    }


def compile_paper(
    paper_dir: str | Path = "paper",
    engine: str = "pdflatex",
    runs: int = 2,
) -> Path:
    """Compile LaTeX manuscript (main.tex) into PDF via pdflatex and bibtex.

    Args:
        paper_dir: Directory containing main.tex and PRIMEarxiv.sty.
        engine: LaTeX compiler command (default: 'pdflatex').
        runs: Number of compilation passes (default: 2).

    Returns:
        Path to the compiled PDF file.

    Raises:
        RuntimeError: If compiler fails or PDF is not produced.
        FileNotFoundError: If main.tex or compiler binary is missing.
    """
    import shutil
    import subprocess

    p_dir = Path(paper_dir).resolve()
    main_tex = p_dir / "main.tex"
    main_pdf = p_dir / "main.pdf"

    if not main_tex.exists():
        raise FileNotFoundError(f"LaTeX document not found at {main_tex}")

    compiler_bin = shutil.which(engine)
    if not compiler_bin:
        raise RuntimeError(f"LaTeX compiler '{engine}' not found on PATH.")

    bibtex_bin = shutil.which("bibtex")

    for pass_idx in range(runs):
        cmd = [compiler_bin, "-interaction=nonstopmode", "main.tex"]
        proc = subprocess.run(
            cmd,
            cwd=str(p_dir),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 and pass_idx == runs - 1 and not main_pdf.exists():
            raise RuntimeError(
                f"LaTeX compilation failed with exit code {proc.returncode}:\n"
                f"{proc.stdout[-2000:]}\n{proc.stderr}"
            )

        # Run bibtex on first pass if bibliography exists
        if pass_idx == 0 and bibtex_bin and (p_dir / "references.bib").exists():
            subprocess.run(
                [bibtex_bin, "main"],
                cwd=str(p_dir),
                capture_output=True,
                text=True,
            )

    if not main_pdf.exists() or main_pdf.stat().st_size == 0:
        raise RuntimeError(f"Compilation did not produce a valid PDF at {main_pdf}")

    logger.info("Successfully compiled %s (%d bytes)", main_pdf, main_pdf.stat().st_size)
    return main_pdf
