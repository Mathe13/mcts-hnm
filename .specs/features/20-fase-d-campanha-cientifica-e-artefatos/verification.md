# 20-fase-d-campanha-cientifica-e-artefatos verification

**Verdict**: PASS
**Proofs**: `proofs.md` at fcf6dd1
**Verifier**: independent sub-agent (author != verifier)
**Date**: 2026-11-11

## Scope and Acceptance Criteria Audit

- **CAMP-01 (D1 - Frozen Protocol Campaign Matrix)**:
  - Scenario 1 executed across 4 methods (Uniform, OHEM, Strict, Enhanced) and 3 ratios (1:50, 1:100, 1:500) with 5 paired seeds and identical positive instances. (AC 1)
  - Scenario 2 executed across 6 ablation methods (Uniform, Uniform Grouped, Strict, QNorm-only, DualStream-only, Enhanced) with 5 paired seeds. (AC 2)
  - Atomic JSON and `.meta.json` sidecar provenance per run with git commit SHA, execution mode, and timestamps. (AC 3)
  - Graceful abort with diagnostic provenance logging on unhandled exceptions or NaN metrics. (AC 4)
- **METR-01 (D2 - Primary Metrics & Computational Efficiency Telemetry)**:
  - Scenario 1 computes AUPRC, precision, recall, and calibrated minority F1 at $\tau_{\text{val}}$. (AC 1)
  - Scenario 2 computes top-1 accuracy, macro-F1, balanced accuracy, Worst-10 recall, and normalized exposure entropy. (AC 2)
  - Computational cost telemetry tracked: step updates, sample gradient counts, additional mining forward passes, and elapsed wall-clock seconds. (AC 3)
  - Both `last` and `best_validation` checkpoint metrics maintained distinctly. (AC 4)
- **STAT-01 (D3 - Paired Inferential Statistics & Interaction Effects)**:
  - Paired method differences across identical seeds computed with mean, standard deviation, and Cohen's d effect size. (AC 1)
  - Method-by-ratio interaction statistics (difference of differences) computed preserving seed pairing. (AC 2)
  - Statistical summary tables report confidence intervals, sample size $n$, and individual seed values. (AC 3)
- **ARTF-01 (D4 - Publication Artifacts, Figures & LaTeX Tables)**:
  - Generates validation curves with variability bands, exposure distribution heatmaps, and $2 \times 2$ QNorm $\times$ DualStream ablation charts. (AC 1)
  - Publication-ready LaTeX tables generated displaying both checkpoints, AUPRC, and costs without placeholder defaults. (AC 2)
  - Rejects missing results or synthetic smoke artifacts during REAL generation with `FileNotFoundError` / `ValueError`. (AC 3)
- **GATE-04 (Gate D - Scientific Audit & Integrity Verifier)**:
  - `verify_gate_d()` audits presence and completeness of all planned scenarios, ratios, methods, and paired seeds. (AC 1)
  - Rejects incomplete runs, non-finite metrics, or synthetic data in REAL mode. (AC 2)
  - Generates signed `campaign_audit_manifest.json` with SHA-256 hashes certifying reproducibility. (AC 3)

## Checks

| Check | Test | Holds |
| --- | --- | --- |
| C1 | `tests/test_phase_d_campaign.py` `test_campaign_runner_scenario1_execution` | yes |
| C2 | `tests/test_phase_d_campaign.py` `test_campaign_runner_scenario2_execution` | yes |
| C3 | `tests/test_phase_d_campaign.py` `test_campaign_runner_provenance_and_manifest` | yes |
| C4 | `tests/test_phase_d_campaign.py` `test_campaign_runner_handles_failure_gracefully` | yes |
| C5 | `tests/test_phase_d_metrics.py` `test_scenario1_computes_auprc_and_calibrated_f1` | yes |
| C6 | `tests/test_phase_d_metrics.py` `test_scenario2_computes_macro_f1_and_worst10` | yes |
| C7 | `tests/test_phase_d_metrics.py` `test_computational_cost_telemetry_recording` | yes |
| C8 | `tests/test_phase_d_metrics.py` `test_dual_checkpoint_metrics_distinction` | yes |
| C9 | `tests/test_phase_d_statistics.py` `test_paired_method_differences_and_cohens_d` | yes |
| C10 | `tests/test_phase_d_statistics.py` `test_method_by_ratio_interaction_statistics` | yes |
| C11 | `tests/test_phase_d_statistics.py` `test_statistical_summary_confidence_intervals` | yes |
| C12 | `tests/test_phase_d_artifacts.py` `test_figure_generation_validation_and_ablation` | yes |
| C13 | `tests/test_phase_d_artifacts.py` `test_latex_table_generation_with_checkpoints_and_auprc` | yes |
| C14 | `tests/test_phase_d_artifacts.py` `test_artifacts_reject_synthetic_or_missing_real_data` | yes |
| C15 | `tests/test_phase_d_gate.py` `test_verify_gate_d_passes_complete_campaign` | yes |
| C16 | `tests/test_phase_d_gate.py` `test_verify_gate_d_rejects_incomplete_or_synthetic_campaign` | yes |
| C17 | `tests/test_phase_d_gate.py` `test_verify_gate_d_generates_signed_audit_manifest` | yes |
| Build | `.venv/bin/python -m compileall -q src tests` | yes |

## Code Quality & Invariants Audit

- **Trainer Instrumentation (`src/training/trainer.py`)**:
  - `evaluate_scenario1_metrics()` calculates AUPRC using `sklearn.metrics.precision_recall_curve` and `auc`, along with precision, recall, and calibrated F1 at threshold $\tau_{\text{val}}$.
  - `evaluate_scenario2_metrics()` calculates multiclass top-1 accuracy, macro-F1, balanced accuracy, and Worst-10 class recall.
  - `train_scenario1()` and `train_scenario2()` accurately record computational cost telemetry: `total_updates`, `gradient_samples`, `mining_forwards` (specifically nonzero for OHEM mining step evaluations), and `elapsed_seconds`.
  - Both `last` and `best_validation` checkpoint metrics and state dictionaries are tracked and preserved without early-stopping bias.
- **Inferential Statistics (`src/utils/statistics.py`)**:
  - `compute_paired_method_differences()` validates array shapes, finite numbers, matching lengths, and optional pairing keys; correctly calculates sample paired difference mean, standard deviation ($ddof=1$), standard error, Cohen's d ($d = \bar{\Delta} / s_{\Delta}$), 95% Student's t confidence interval, paired t-test (`ttest_rel`), and Wilcoxon signed-rank test.
  - `compute_method_by_ratio_interaction()` computes the difference of differences $\Delta_s = (A(s, r_2) - B(s, r_2)) - (A(s, r_1) - B(s, r_1))$, estimating mean interaction, Cohen's d, 95% CI, and one-sample t-test against null mean zero.
- **Scientific Campaign Runner (`src/training/campaign_runner.py`)**:
  - Orchestrates frozen matrix across Scenarios 1 and 2 with 5 paired seeds (42, 43, 44, 45, 46).
  - Guarantees shared identical positive instances for Scenario 1 across ratios per seed.
  - Serializes atomic `.json` and `.meta.json` per run with git commit SHA and execution mode.
  - Generates consolidated `campaign_manifest.json` indexing all runs.
  - Gracefully records failures with structured error payload.
- **Paper Artifacts & Visualizations (`src/utils/paper_artifacts.py`)**:
  - Generates validation curves with shaded variability bands across paired seeds (PDF and PNG).
  - Generates class exposure heatmaps and $2 \times 2$ QNorm $\times$ DualStream ablation charts.
  - Generates publication-ready LaTeX tables distinguishing `last` and `best_validation` checkpoints, displaying AUPRC and computational costs.
  - Strict anti-synthetic enforcement: rejects missing runs or synthetic smoke runs when running in REAL mode.
- **Gate D Verifier (`src/utils/gate_d_verifier.py`)**:
  - Audits matrix completeness, finite numerical metrics, dual checkpoint records, and cost telemetry.
  - Enforces strict anti-synthetic policy in REAL mode.
  - Computes file SHA-256 checksums and serializes signed `campaign_audit_manifest.json` with `.meta.json` sidecar.

## Findings

None - All unit and integration tests pass, boundary conditions are verified, anti-synthetic guards strictly reject invalid data, and 100% of proof criteria are satisfied.

Gaps: None
