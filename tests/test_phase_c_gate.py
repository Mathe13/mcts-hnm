"""Tests for Phase C Gate C Verifier and causal classification (Gate C)."""

from __future__ import annotations

import pytest

from src.utils.gate_c_verifier import verify_gate_c


def make_dummy_run(
    strategy: str,
    seed: int = 42,
    lr: float = 1e-3,
    num_steps: int = 100,
    batch_size: int = 64,
    accuracy: float = 0.50,
    mean_batch_entropy: float = 2.5,
    mean_unique_classes: float = 20.0,
    coverage: float = 0.80,
) -> dict[str, object]:
    """Create a mock run dictionary conforming to Phase C requirements."""
    return {
        "strategy": strategy,
        "seed": seed,
        "lr": lr,
        "num_steps": num_steps,
        "batch_size": batch_size,
        "val_metrics": {
            "loss": 1.25,
            "accuracy": accuracy,
            "top1_accuracy": accuracy,
            "macro_f1": accuracy * 0.9,
        },
        "exposure_diagnostics": {
            "mean_batch_entropy": mean_batch_entropy,
            "mean_unique_classes_per_batch": mean_unique_classes,
            "cumulative_class_coverage": coverage,
            "normalized_exposure_entropy": 0.85,
        },
    }


def test_verify_gate_c_passes_valid_ablations() -> None:
    """Verify verify_gate_c passes valid paired runs and classifies batch homogeneity."""
    # Scenario: Uniform has 60% acc, Uniform Grouped drops to 45%, Enhanced drops to 44%
    # -> Batch homogeneity is diagnosed as primary driver
    runs = [
        make_dummy_run("uniform", accuracy=0.60, mean_batch_entropy=3.5, mean_unique_classes=40.0),
        make_dummy_run("uniform_grouped", accuracy=0.45, mean_batch_entropy=1.8, mean_unique_classes=15.0),
        make_dummy_run("mcts_enhanced", accuracy=0.44, mean_batch_entropy=1.7, mean_unique_classes=14.0),
    ]

    report = verify_gate_c(runs)
    assert report["status"] == "PASS"
    assert report["passed"] is True
    assert report["num_runs_verified"] == 3
    assert report["parameter_invariance_verified"] is True
    assert report["telemetry_completeness_verified"] is True
    assert report["causal_diagnosis"]["classification"] == "batch_homogeneity"
    assert pytest.approx(report["causal_diagnosis"]["delta_batch"], rel=1e-5) == 0.15


def test_verify_gate_c_rejects_divergent_parameters() -> None:
    """Verify verify_gate_c rejects runs with divergent fixed parameters or missing telemetry."""
    # 1. Divergent learning rate between paired runs
    run1 = make_dummy_run("mcts_strict", seed=42, lr=1e-3)
    run2 = make_dummy_run("mcts_enhanced", seed=42, lr=5e-3)
    with pytest.raises(ValueError, match="Parameter mismatch"):
        verify_gate_c([run1, run2], raise_on_error=True)

    # 2. Missing exposure telemetry
    bad_telemetry = make_dummy_run("uniform", seed=42)
    del bad_telemetry["exposure_diagnostics"]["mean_batch_entropy"]  # type: ignore
    with pytest.raises(ValueError, match="missing 'mean_batch_entropy'"):
        verify_gate_c([bad_telemetry], raise_on_error=True)

    # 3. Non-finite metric
    nan_metric = make_dummy_run("uniform", seed=42)
    nan_metric["val_metrics"]["loss"] = float("nan")  # type: ignore
    with pytest.raises(ValueError, match="non-finite metric"):
        verify_gate_c([nan_metric], raise_on_error=True)
