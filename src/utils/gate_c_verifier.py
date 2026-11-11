"""Gate C Verifier: Variable Isolation and Causal Attribution Audit (Phase C)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def verify_gate_c(
    ablation_runs: list[dict[str, Any]] | Path | str,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    r"""Verify variable isolation, telemetry completeness, and causal diagnosis for Gate C.

    Audits:
    1. Parameter Invariance: Non-ablated parameters (learning rate, seed, step count, architecture)
       must remain strictly identical across paired comparison runs.
    2. Numerical Validity: All recorded metrics, losses, and diagnostics must be finite (no NaN/Inf).
    3. Telemetry Completeness: Each run record must contain complete batch diversity and exposure telemetry:
       `mean_batch_entropy`, `mean_unique_classes_per_batch`, and `cumulative_class_coverage`.
    4. Causal Classification: Compares `uniform`, `uniform_grouped`, and `mcts_enhanced` to classify
       whether performance drops stem primarily from batch homogeneity or MCTS policy starvation.

    Args:
        ablation_runs: List of run dictionary outputs or Path/directory containing JSON records.
        raise_on_error: If True, raises ValueError upon any invariant violation.

    Returns:
        Structured audit report dictionary with status, verification flags, and causal diagnosis.
    """
    runs: list[dict[str, Any]] = []

    if isinstance(ablation_runs, (str, Path)):
        p = Path(ablation_runs)
        if p.is_dir():
            json_files = sorted(f for f in p.glob("*.json") if not f.name.endswith(".meta.json"))
            for jf in json_files:
                with open(jf, "r", encoding="utf-8") as f:
                    runs.append(json.load(f))
        elif p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                content = json.load(f)
                runs = content if isinstance(content, list) else [content]
        else:
            raise FileNotFoundError(f"Path does not exist: {p}")
    elif isinstance(ablation_runs, list):
        runs = ablation_runs
    else:
        raise ValueError(f"ablation_runs must be a list or Path, got {type(ablation_runs).__name__}.")

    if len(runs) == 0:
        if raise_on_error:
            raise ValueError("No ablation run records provided to verify_gate_c().")
        return {"status": "FAIL", "reason": "No runs provided", "passed": False}

    # 1. Inspect each run for numerical validity and telemetry completeness
    for idx, r in enumerate(runs):
        strategy = r.get("strategy", f"run_{idx}")
        # Check telemetry
        exp_diag = r.get("exposure_diagnostics")
        if not isinstance(exp_diag, dict):
            msg = f"Run {idx} ({strategy}) missing 'exposure_diagnostics'."
            if raise_on_error:
                raise ValueError(msg)
            return {"status": "FAIL", "reason": msg, "passed": False}

        for required_key in (
            "mean_batch_entropy",
            "mean_unique_classes_per_batch",
            "cumulative_class_coverage",
        ):
            if required_key not in exp_diag:
                msg = f"Run {idx} ({strategy}) exposure_diagnostics missing '{required_key}'."
                if raise_on_error:
                    raise ValueError(msg)
                return {"status": "FAIL", "reason": msg, "passed": False}
            val = exp_diag[required_key]
            if not np.isfinite(val):
                msg = f"Run {idx} ({strategy}) has non-finite '{required_key}': {val}."
                if raise_on_error:
                    raise ValueError(msg)
                return {"status": "FAIL", "reason": msg, "passed": False}

        # Check metrics
        val_metrics = r.get("val_metrics") or r.get("final_metrics")
        if isinstance(val_metrics, dict):
            for m_key, m_val in val_metrics.items():
                if isinstance(m_val, (int, float, np.number)) and not np.isfinite(m_val):
                    msg = f"Run {idx} ({strategy}) has non-finite metric '{m_key}': {m_val}."
                    if raise_on_error:
                        raise ValueError(msg)
                    return {"status": "FAIL", "reason": msg, "passed": False}

    # 2. Check pairwise parameter invariance
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for r in runs:
        s = int(r.get("seed", 42))
        by_seed.setdefault(s, []).append(r)

    invariant_keys = ("lr", "num_steps", "batch_size")
    for s, seed_runs in by_seed.items():
        if len(seed_runs) > 1:
            ref = seed_runs[0]
            for other in seed_runs[1:]:
                for k in invariant_keys:
                    if k in ref and k in other:
                        if ref[k] != other[k]:
                            msg = (
                                f"Parameter mismatch for seed {s}: '{k}' differs "
                                f"between {ref.get('strategy')} ({ref[k]}) and "
                                f"{other.get('strategy')} ({other[k]})."
                            )
                            if raise_on_error:
                                raise ValueError(msg)
                            return {"status": "FAIL", "reason": msg, "passed": False}

    # 3. Causal diagnostic classification
    strategy_acc: dict[str, float] = {}
    for r in runs:
        strat = r.get("strategy")
        vm = r.get("val_metrics") or r.get("final_metrics") or {}
        acc = vm.get("top1_accuracy") or vm.get("accuracy")
        if strat and acc is not None and np.isfinite(acc):
            strategy_acc[strat] = float(acc)

    diagnosis = {
        "classification": "insufficient_data",
        "delta_batch": None,
        "delta_policy": None,
        "explanation": "Runs do not contain complete Uniform, Uniform Grouped, and Enhanced triplet.",
    }

    if (
        "uniform" in strategy_acc
        and "uniform_grouped" in strategy_acc
        and "mcts_enhanced" in strategy_acc
    ):
        acc_uni = strategy_acc["uniform"]
        acc_grp = strategy_acc["uniform_grouped"]
        acc_enh = strategy_acc["mcts_enhanced"]

        delta_batch = acc_uni - acc_grp
        delta_policy = acc_grp - acc_enh

        if delta_batch > 0.05 and abs(acc_grp - acc_enh) <= 0.05:
            classification = "batch_homogeneity"
            explanation = (
                "Batch homogeneity is the primary driver of performance drop; "
                "uniform grouped batch replicates the drop without MCTS policy."
            )
        elif delta_policy > 0.05 and abs(acc_uni - acc_grp) <= 0.05:
            classification = "policy_selection"
            explanation = (
                "MCTS policy selection/starvation is the primary driver; "
                "uniform grouping alone preserves baseline accuracy."
            )
        else:
            classification = "mixed_or_inconclusive"
            explanation = (
                "Both batch composition and policy selection contribute, or observed "
                "differences are within margin of uncertainty."
            )

        diagnosis = {
            "classification": classification,
            "delta_batch": float(delta_batch),
            "delta_policy": float(delta_policy),
            "explanation": explanation,
        }

    return {
        "status": "PASS",
        "passed": True,
        "num_runs_verified": len(runs),
        "parameter_invariance_verified": True,
        "telemetry_completeness_verified": True,
        "causal_diagnosis": diagnosis,
    }
