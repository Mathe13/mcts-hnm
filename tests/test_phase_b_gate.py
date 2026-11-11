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
