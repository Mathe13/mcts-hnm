"""End-to-end integration test verifying Gate A audit and integrity checks."""

from __future__ import annotations

from src.utils.gate_a_verifier import verify_gate_a


def test_verify_gate_a():
    """Verify that Gate A audit returns 0, confirming all Phase A invariants."""
    exit_code = verify_gate_a(verbose=True)
    assert exit_code == 0, "Gate A verification failed! Check log output for failed invariant."
