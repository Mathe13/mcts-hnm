# Publication integrity fix (2026-11)

This patch prevents the REAL publication workflow from outputting synthetic
or fixed-value figures that contradict benchmark measurements.

- FIG-04 (`fig_ablation_2x2`) is constructed from the measured four-strategy
  ablation at a shared observed ratio, or omitted.
- FIG-05 (`fig_interaction_ratio_methods`) is constructed only when at least
  two imbalance settings were actually evaluated per strategy. A single
  measured 1:50 configuration does **not** imply 1:10 or 1:100 results.
- FIG-06 (`fig_precision_recall_curves`) is omitted in REAL until per-seed
  prediction scores / measured PR points are exported. AUPRC alone cannot
  reconstruct a PR curve.
- FIG-07 (`fig_evaluation_4quadrants_comparison`) is omitted in REAL until
  four evaluated test-quadrant metrics per seed are persisted.
- Other auxiliary plots require measured per-run telemetry and are recorded
  as skipped when their inputs are missing.
- Obsolete figures are moved into `paper/figures/_unverified_legacy/` during
  REAL generation; they are never silently deleted.
- The LaTeX manuscript is rewritten conservatively to remove unverified
  superiority statements and to use optional figure includes.
- SMOKE synthetic figures remain available only under explicit SMOKE mode.
- Notebook 05 stored outputs are cleared; re-execute it for authoritative
  figures, tables and a new artifact manifest.

Review `git diff` and new figures prior to publication. This patch does not
implement new per-seed PR score exports or generate missing experiments.
