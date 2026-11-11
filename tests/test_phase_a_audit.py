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
