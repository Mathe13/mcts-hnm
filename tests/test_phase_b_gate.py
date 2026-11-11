"""Tests for Phase B diagnostic pilot runner and Gate B verification (B4, Gate B)."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.training.pilot_runner import run_diagnostic_pilot


def test_pilot_runner_smoke_execution(tmp_path: Path):
    """Verify that run_diagnostic_pilot runs short diagnostic runs across Scenarios 1 and 2 (PILOT-01, AC 1)."""
    # 2 seeds, 1 step, ratios 50 and 100 for fast test execution
    records = run_diagnostic_pilot(
        output_dir=tmp_path / "pilot_runs",
        seeds=(42, 43),
        num_steps=2,
        eval_every=2,
        ratios=(50, 100),
        num_positives=10,
        execution_mode="smoke",
    )

    # Scenario 1: 2 seeds * 2 ratios * 2 strats = 8 runs
    # Scenario 2: 2 seeds * 2 strats = 4 runs
    # Total = 12 runs
    assert len(records) == 12

    for rec in records:
        assert "run_id" in rec
        assert "step_history" in rec
        assert "exposure_diagnostics" in rec
        assert len(rec["step_history"]) >= 1


def test_pilot_output_isolation(tmp_path: Path):
    """Verify that pilot writes isolated JSON records to target dir without touching legacy (PILOT-01, AC 1)."""
    out_dir = tmp_path / "custom_pilot"
    run_diagnostic_pilot(
        output_dir=out_dir,
        seeds=(42,),
        num_steps=1,
        eval_every=1,
        ratios=(50,),
        num_positives=5,
    )

    assert out_dir.exists()
    json_files = list(out_dir.glob("*.json"))
    # 1 seed * 1 ratio * 2 strats (scen1) + 1 seed * 2 strats (scen2) = 4 files
    assert len(json_files) == 4

    # Ensure no file was written to results/legacy
    legacy_files = list(Path("results/legacy").glob("pilot*.json"))
    assert len(legacy_files) == 0


def test_verify_gate_b_passes(tmp_path: Path):
    """Verify that verify_gate_b audits pilot records and returns exit code 0 (GATE-02, AC 2)."""
    from src.utils.gate_b_verifier import verify_gate_b

    records = run_diagnostic_pilot(
        output_dir=tmp_path / "valid_pilot",
        seeds=(42,),
        num_steps=2,
        eval_every=2,
        ratios=(50, 100),
        num_positives=10,
    )

    exit_code = verify_gate_b(run_records=records, verbose=True)
    assert exit_code == 0


def test_verify_gate_b_rejects_nan_and_invariants():
    """Verify that verify_gate_b returns exit code 1 on NaN values or broken invariants (GATE-02, AC 3)."""
    from src.utils.gate_b_verifier import verify_gate_b

    # Case 1: NaN in step_history
    corrupted_records = [{
        "run_id": "test_corrupt",
        "scenario": "scenario1",
        "imbalance_ratio": 50,
        "num_positives": 10,
        "num_negatives": 500,
        "step_history": [{"train_loss": float("nan"), "val_loss": 0.5, "step": 1}],
        "exposure_diagnostics": {"normalized_exposure_entropy": 0.5},
        "val_metrics": {"val_loss": 0.5},
    }]
    assert verify_gate_b(run_records=corrupted_records, verbose=False) == 1

    # Case 2: Ratio mismatch (10 positives * ratio 50 != 400 negatives)
    mismatched_records = [{
        "run_id": "test_mismatch",
        "scenario": "scenario1",
        "imbalance_ratio": 50,
        "num_positives": 10,
        "num_negatives": 400,
        "step_history": [{"train_loss": 0.4, "val_loss": 0.5, "step": 1}],
        "exposure_diagnostics": {"normalized_exposure_entropy": 0.5},
        "val_metrics": {"val_loss": 0.5},
    }]
    assert verify_gate_b(run_records=mismatched_records, verbose=False) == 1
