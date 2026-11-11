"""Tests for Phase D: Gate D Scientific Audit and Manifest Verifier."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.utils.experiment_config import ExecutionMode
from src.utils.gate_d_verifier import verify_gate_d


def _make_dummy_campaign_files(tmp_path: Path, mode: str = "smoke"):
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    r1_file = campaign_dir / "s1_1_50_uniform_seed42.json"
    r1_payload = {
        "run_id": "s1_1_50_uniform_seed42",
        "scenario": "scenario1",
        "imbalance_ratio": "1:50",
        "strategy": "uniform",
        "seed": 42,
        "status": "completed",
        "last_metrics": {"auprc": 0.12, "calibrated_f1": 0.25, "precision": 0.20, "recall": 0.35},
        "best_validation_metrics": {"auprc": 0.15, "calibrated_f1": 0.28, "precision": 0.22, "recall": 0.38},
        "computational_costs": {"total_updates": 100, "gradient_samples": 6400, "mining_forwards": 0, "elapsed_seconds": 12.0},
    }
    with open(r1_file, "w", encoding="utf-8") as f:
        json.dump(r1_payload, f)

    r2_file = campaign_dir / "s1_1_50_ohem_seed42.json"
    r2_payload = {
        "run_id": "s1_1_50_ohem_seed42",
        "scenario": "scenario1",
        "imbalance_ratio": "1:50",
        "strategy": "ohem",
        "seed": 42,
        "status": "completed",
        "last_metrics": {"auprc": 0.14, "calibrated_f1": 0.26, "precision": 0.21, "recall": 0.36},
        "best_validation_metrics": {"auprc": 0.16, "calibrated_f1": 0.30, "precision": 0.24, "recall": 0.40},
        "computational_costs": {"total_updates": 100, "gradient_samples": 6400, "mining_forwards": 19200, "elapsed_seconds": 18.0},
    }
    with open(r2_file, "w", encoding="utf-8") as f:
        json.dump(r2_payload, f)

    r3_file = campaign_dir / "s2_uniform_seed42.json"
    r3_payload = {
        "run_id": "s2_uniform_seed42",
        "scenario": "scenario2",
        "strategy": "uniform",
        "seed": 42,
        "status": "completed",
        "last_metrics": {"top1_accuracy": 0.50, "macro_f1": 0.48, "balanced_accuracy": 0.49, "worst_10_class_recall": 0.25},
        "best_validation_metrics": {"top1_accuracy": 0.52, "macro_f1": 0.50, "balanced_accuracy": 0.51, "worst_10_class_recall": 0.28},
        "computational_costs": {"total_updates": 100, "gradient_samples": 6400, "mining_forwards": 0, "elapsed_seconds": 10.0},
    }
    with open(r3_file, "w", encoding="utf-8") as f:
        json.dump(r3_payload, f)

    manifest_file = campaign_dir / "campaign_manifest.json"
    manifest_data = {
        "campaign_id": f"campaign_{mode}",
        "mode": mode,
        "total_expected_runs": 3,
        "completed_runs": 3,
        "failed_runs": 0,
        "runs": {
            "s1_1_50_uniform_seed42": {"run_file": str(r1_file)},
            "s1_1_50_ohem_seed42": {"run_file": str(r2_file)},
            "s2_uniform_seed42": {"run_file": str(r3_file)},
        },
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    return campaign_dir


def test_verify_gate_d_passes_complete_campaign(tmp_path: Path) -> None:
    campaign_dir = _make_dummy_campaign_files(tmp_path, mode="smoke")
    res = verify_gate_d(campaign_dir, expected_mode=ExecutionMode.SMOKE)

    assert res["passed"] is True
    assert res["status"] == "PASS"
    assert res["total_runs_audited"] == 3
    assert res["checks"]["matrix_completeness"] is True
    assert res["checks"]["finite_metrics"] is True
    assert res["checks"]["dual_checkpoints_present"] is True
    assert res["checks"]["computational_telemetry"] is True


def test_verify_gate_d_rejects_incomplete_or_synthetic_campaign(tmp_path: Path) -> None:
    # 1. Missing manifest file
    with pytest.raises(FileNotFoundError, match="Campaign manifest file does not exist"):
        verify_gate_d(tmp_path / "empty_dir")

    # 2. Mode mismatch: manifest has smoke, but REAL mode is expected
    smoke_campaign_dir = _make_dummy_campaign_files(tmp_path, mode="smoke")
    with pytest.raises(ValueError, match="REAL mode was expected"):
        verify_gate_d(smoke_campaign_dir, expected_mode=ExecutionMode.REAL)

    # 3. Failed run in manifest
    manifest_file = smoke_campaign_dir / "campaign_manifest.json"
    with open(manifest_file, "r") as f:
        data = json.load(f)
    data["runs"]["s1_1_50_uniform_seed42"]["status"] = "failed"
    data["runs"]["s1_1_50_uniform_seed42"]["error"] = "Out of memory"
    with open(manifest_file, "w") as f:
        json.dump(data, f)

    with pytest.raises(ValueError, match="has status 'failed'"):
        verify_gate_d(manifest_file, expected_mode=ExecutionMode.SMOKE)


def test_verify_gate_d_generates_signed_audit_manifest(tmp_path: Path) -> None:
    campaign_dir = _make_dummy_campaign_files(tmp_path, mode="smoke")
    res = verify_gate_d(campaign_dir, expected_mode=ExecutionMode.SMOKE, output_audit_manifest=True)

    audit_file = campaign_dir / "campaign_audit_manifest.json"
    assert audit_file.is_file()
    assert (campaign_dir / "campaign_audit_manifest.json.meta.json").is_file()

    with open(audit_file, "r", encoding="utf-8") as f:
        audit_manifest = json.load(f)

    assert audit_manifest["status"] == "PASS"
    assert audit_manifest["total_runs_audited"] == 3
    assert len(audit_manifest["verified_file_hashes"]) >= 3
