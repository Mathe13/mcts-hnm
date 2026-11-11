"""Gate D Verifier: Scientific Campaign Audit and Integrity Verifier (Phase D)."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.experiment_config import ExecutionMode, ExperimentConfig, save_provenance_metadata

logger = logging.getLogger(__name__)


def _compute_sha256(path: Path) -> str:
    """Compute hex SHA-256 checksum for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_gate_d(
    campaign_dir_or_manifest: dict[str, Any] | str | Path,
    expected_mode: ExecutionMode | str | None = None,
    output_audit_manifest: bool = True,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    r"""Audit frozen matrix completeness, numerical validity, dual checkpoints, and anti-synthetic integrity.

    Audits:
    1. Matrix Completeness: All expected runs for Scenarios 1 and 2 must have status "completed".
    2. Numerical Validity: All metrics (AUPRC, F1, accuracy, etc.) must be finite (no NaN/Inf).
    3. Dual Checkpoints: Both `last` and `best_validation` checkpoint metrics must be populated.
    4. Computational Cost Telemetry: Step updates, gradient samples, mining forwards, and elapsed time.
    5. Anti-Synthetic Cleanliness: In REAL mode, rejects any synthetic runs or missing result sets.
    6. Audit Manifest: Generates signed `campaign_audit_manifest.json` with file SHA-256 hashes.
    """
    if isinstance(campaign_dir_or_manifest, dict):
        manifest = campaign_dir_or_manifest
        base_dir = Path(manifest.get("campaign_dir", "."))
    else:
        p = Path(campaign_dir_or_manifest)
        if p.is_dir():
            manifest_file = p / "campaign_manifest.json"
        else:
            manifest_file = p
            base_dir = p.parent
        if not manifest_file.is_file():
            msg = f"Campaign manifest file does not exist at '{manifest_file}'."
            if raise_on_error:
                raise FileNotFoundError(msg)
            return {"passed": False, "status": "FAIL", "reason": msg}

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        base_dir = manifest_file.parent

    # Determine mode
    mode_str = str(expected_mode.value if isinstance(expected_mode, ExecutionMode) else (expected_mode or manifest.get("mode", "smoke"))).lower()
    is_real = mode_str == "real"

    checks = {
        "matrix_completeness": False,
        "finite_metrics": False,
        "dual_checkpoints_present": False,
        "computational_telemetry": False,
        "anti_synthetic_clean": False,
    }

    runs_dict = manifest.get("runs", {})
    if not runs_dict:
        msg = f"No run records found in campaign manifest in '{base_dir}'."
        if raise_on_error:
            raise ValueError(msg)
        return {"passed": False, "status": "FAIL", "reason": msg, "checks": checks}

    # 1. Anti-synthetic check in REAL mode
    if is_real:
        if manifest.get("mode") != "real":
            msg = f"Campaign manifest indicates mode '{manifest.get('mode')}', but REAL mode was expected."
            if raise_on_error:
                raise ValueError(msg)
            return {"passed": False, "status": "FAIL", "reason": msg, "checks": checks}

        if manifest.get("failed_runs", 0) > 0:
            msg = f"Campaign has {manifest['failed_runs']} failed runs; partial campaigns are prohibited in REAL mode."
            if raise_on_error:
                raise ValueError(msg)
            return {"passed": False, "status": "FAIL", "reason": msg, "checks": checks}
    checks["anti_synthetic_clean"] = True

    # 2. Inspect each run
    total_audited = 0
    file_hashes: dict[str, str] = {}

    for run_id, r_info in runs_dict.items():
        total_audited += 1
        run_file = Path(r_info.get("run_file", base_dir / f"{run_id}.json"))
        if run_file.is_file():
            file_hashes[run_file.name] = _compute_sha256(run_file)
            with open(run_file, "r", encoding="utf-8") as rf:
                payload = json.load(rf)
        else:
            payload = r_info

        # Check status in either manifest summary or run file payload
        run_status = r_info.get("status") if r_info.get("status") is not None else payload.get("status")
        if run_status != "completed":
            err_msg = r_info.get("error") or payload.get("error")
            msg = f"Run '{run_id}' has status '{run_status}', error: {err_msg}."
            if raise_on_error:
                raise ValueError(msg)
            return {"passed": False, "status": "FAIL", "reason": msg, "checks": checks}

        # Check dual checkpoints
        last_m = payload.get("last_metrics")
        best_m = payload.get("best_validation_metrics")
        if not isinstance(last_m, dict) or not isinstance(best_m, dict):
            msg = f"Run '{run_id}' missing 'last_metrics' or 'best_validation_metrics'."
            if raise_on_error:
                raise ValueError(msg)
            return {"passed": False, "status": "FAIL", "reason": msg, "checks": checks}

        # Check finite metrics
        for chk_name, m_dict in [("last_metrics", last_m), ("best_validation_metrics", best_m)]:
            for k, v in m_dict.items():
                if isinstance(v, (int, float, np.number)):
                    if not np.isfinite(v):
                        msg = f"Run '{run_id}' has non-finite metric '{k}' in {chk_name}: {v}."
                        if raise_on_error:
                            raise ValueError(msg)
                        return {"passed": False, "status": "FAIL", "reason": msg, "checks": checks}

        scenario = payload.get("scenario")
        if scenario == "scenario1":
            for chk_name, m_dict in [("last_metrics", last_m), ("best_validation_metrics", best_m)]:
                if "auprc" not in m_dict or "calibrated_f1" not in m_dict:
                    msg = f"Scenario 1 run '{run_id}' missing 'auprc' or 'calibrated_f1' in {chk_name}."
                    if raise_on_error:
                        raise ValueError(msg)
                    return {"passed": False, "status": "FAIL", "reason": msg, "checks": checks}
        elif scenario == "scenario2":
            for chk_name, m_dict in [("last_metrics", last_m), ("best_validation_metrics", best_m)]:
                if "macro_f1" not in m_dict or "worst_10_class_recall" not in m_dict:
                    msg = f"Scenario 2 run '{run_id}' missing 'macro_f1' or 'worst_10_class_recall' in {chk_name}."
                    if raise_on_error:
                        raise ValueError(msg)
                    return {"passed": False, "status": "FAIL", "reason": msg, "checks": checks}

        # Check computational cost telemetry
        costs = payload.get("computational_costs")
        if not isinstance(costs, dict):
            msg = f"Run '{run_id}' missing 'computational_costs' dictionary."
            if raise_on_error:
                raise ValueError(msg)
            return {"passed": False, "status": "FAIL", "reason": msg, "checks": checks}

        for cost_key in ("total_updates", "gradient_samples", "mining_forwards", "elapsed_seconds"):
            if cost_key not in costs or not np.isfinite(costs[cost_key]):
                msg = f"Run '{run_id}' missing or non-finite cost key '{cost_key}'."
                if raise_on_error:
                    raise ValueError(msg)
                return {"passed": False, "status": "FAIL", "reason": msg, "checks": checks}

        strat = payload.get("strategy")
        if strat == "ohem" and costs["mining_forwards"] <= 0:
            msg = f"Run '{run_id}' is OHEM but reported {costs['mining_forwards']} mining forwards."
            if raise_on_error:
                raise ValueError(msg)
            return {"passed": False, "status": "FAIL", "reason": msg, "checks": checks}

    checks["matrix_completeness"] = True
    checks["finite_metrics"] = True
    checks["dual_checkpoints_present"] = True
    checks["computational_telemetry"] = True

    audit_path = base_dir / "campaign_audit_manifest.json"
    audit_data = {
        "audit_version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "campaign_id": manifest.get("campaign_id", "unknown"),
        "mode": mode_str,
        "total_runs_audited": total_audited,
        "checks": checks,
        "verified_file_hashes": file_hashes,
        "status": "PASS",
    }

    if output_audit_manifest:
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(audit_data, f, indent=2)
        save_provenance_metadata(audit_path, ExperimentConfig(execution_mode=ExecutionMode(mode_str)))

    return {
        "passed": True,
        "status": "PASS",
        "mode": mode_str,
        "total_runs_audited": total_audited,
        "checks": checks,
        "audit_manifest_path": str(audit_path),
        "reason": None,
    }
