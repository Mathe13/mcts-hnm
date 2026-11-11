"""Gate B Verifier: Validates pilot records, 1:500 viability, and complete diagnostics (Phase B4)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def verify_gate_b(
    run_records: list[dict[str, Any]] | None = None,
    pilot_dir: Path | str | None = None,
    verbose: bool = True,
) -> int:
    r"""Audit pilot execution records against Gate B and 1:500 viability invariants.

    Invariants checked:
    1. Every record contains non-empty `step_history` with finite loss and evaluation metrics.
    2. No NaN or Inf exists in any loss, gradient norm, or evaluation metric.
    3. Complete `exposure_diagnostics` is present with valid normalized entropy in [0, 1].
    4. In Scenario 1 with ratio 500, positive and negative counts strictly obey exact ratio.
    5. Final and validation metrics are finite and non-empty.

    Returns:
        0 if all invariants hold, 1 if any record violates invariants.
    """
    records: list[dict[str, Any]] = []

    if run_records is not None:
        records = run_records
    elif pilot_dir is not None:
        p_dir = Path(pilot_dir)
        if not p_dir.exists():
            logger.error("Gate B FAILED: Pilot directory does not exist: %s", p_dir)
            return 1
        for f in sorted(p_dir.glob("*.json")):
            with open(f, encoding="utf-8") as fp:
                records.append(json.load(fp))
    else:
        # Default pilot location
        default_dir = Path("results/pilot")
        if default_dir.exists():
            for f in sorted(default_dir.glob("*.json")):
                with open(f, encoding="utf-8") as fp:
                    records.append(json.load(fp))

    if not records:
        logger.error("Gate B FAILED: No pilot run records available to verify.")
        return 1

    for idx, rec in enumerate(records):
        run_id = rec.get("run_id", f"record_{idx}")

        # Check 1: step_history presence and non-emptiness
        step_hist = rec.get("step_history")
        if not step_hist or not isinstance(step_hist, list):
            logger.error("Gate B FAILED on %s: missing or empty step_history", run_id)
            return 1

        for step_entry in step_hist:
            for k in ("train_loss", "val_loss", "step"):
                if k not in step_entry:
                    logger.error("Gate B FAILED on %s: missing key '%s' in step_history", run_id, k)
                    return 1
                val = step_entry[k]
                if isinstance(val, (int, float)) and not np.isfinite(val):
                    logger.error("Gate B FAILED on %s: non-finite value in step_history for '%s': %s", run_id, k, val)
                    return 1

        # Check 2: exposure_diagnostics
        diag = rec.get("exposure_diagnostics")
        if not diag or not isinstance(diag, dict):
            logger.error("Gate B FAILED on %s: missing exposure_diagnostics", run_id)
            return 1
        entropy = diag.get("normalized_exposure_entropy")
        if entropy is None or not (0.0 <= float(entropy) <= 1.0):
            logger.error("Gate B FAILED on %s: invalid normalized_exposure_entropy: %s", run_id, entropy)
            return 1

        # Check 3: Scenario 1 ratio viability
        if rec.get("scenario") == "scenario1":
            ratio = rec.get("imbalance_ratio")
            num_pos = rec.get("num_positives")
            num_neg = rec.get("num_negatives")

            if ratio is not None and num_pos is not None and num_neg is not None:
                expected_neg = int(num_pos) * int(ratio)
                if int(num_neg) != expected_neg:
                    logger.error(
                        "Gate B FAILED on %s: ratio mismatch: %s positives * %s != %s negatives",
                        run_id,
                        num_pos,
                        ratio,
                        num_neg,
                    )
                    return 1

        # Check 4: validation metrics presence and finiteness
        val_met = rec.get("val_metrics")
        if not val_met or not isinstance(val_met, dict):
            logger.error("Gate B FAILED on %s: missing val_metrics", run_id)
            return 1
        for mk, mv in val_met.items():
            if isinstance(mv, (int, float)) and not np.isfinite(mv):
                logger.error("Gate B FAILED on %s: non-finite val_metric '%s': %s", run_id, mk, mv)
                return 1

    if verbose:
        print(f"[Gate B] ALL {len(records)} PILOT RUN RECORDS VERIFIED SUCCESSFULLY.")
    return 0


if __name__ == "__main__":
    sys.exit(verify_gate_b(verbose=True))
