"""Tests for Phase A audit documentation and run manifest generation (A1)."""

from __future__ import annotations

from pathlib import Path


def test_audit_documentation_present_and_complete():
    """Verify that docs/experimental_audit.md exists and contains required audit sections."""
    audit_file = Path("docs/experimental_audit.md")
    assert audit_file.exists(), "docs/experimental_audit.md must exist."
    assert audit_file.stat().st_size > 0, "docs/experimental_audit.md must not be empty."

    content = audit_file.read_text(encoding="utf-8")
    required_sections = [
        "Experimental Audit and Historical Inventory",
        "Directory Layout and Immutability Boundary",
        "Embedding and Tree Artifact Hashes",
        "Experimental Provenance Requirements",
        "Immutability Invariants",
    ]
    for section in required_sections:
        assert section.lower() in content.lower(), f"Missing required section '{section}' in audit doc."


def test_generate_run_manifest():
    """Verify that generate_run_manifest produces all required provenance metadata."""
    from src.utils.experiment_config import ExecutionMode, ExperimentConfig, generate_run_manifest

    cfg = ExperimentConfig(execution_mode=ExecutionMode.SMOKE, seed=42)
    manifest = generate_run_manifest(
        run_id="scen1_cls0_r50_semantic_mcts_enhanced_s42_test1234",
        config=cfg,
        seed=42,
    )

    assert manifest["run_id"] == "scen1_cls0_r50_semantic_mcts_enhanced_s42_test1234"
    assert "git_commit" in manifest and len(manifest["git_commit"]) > 0
    assert "timestamp_utc" in manifest
    assert "hardware" in manifest
    assert "platform" in manifest["hardware"]
    assert "dependencies" in manifest
    assert "torch" in manifest["dependencies"]
    assert "numpy" in manifest["dependencies"]
    assert "resolved_config" in manifest
    assert manifest["resolved_config"]["execution_mode"] == "smoke"


def test_legacy_immutability():
    """Verify that write operations targeting results/legacy raise PermissionError."""
    import pytest
    from src.utils.experiment_config import (
        ExecutionMode,
        ExperimentConfig,
        assert_path_mutable,
        save_provenance_metadata,
    )

    legacy_path = Path("results/legacy/test_artifact.json")
    with pytest.raises(PermissionError, match="immutable"):
        assert_path_mutable(legacy_path)

    cfg = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    with pytest.raises(PermissionError, match="immutable"):
        save_provenance_metadata(legacy_path, cfg)

