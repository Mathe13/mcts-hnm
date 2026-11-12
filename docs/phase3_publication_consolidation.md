# Patch 3 — Publication-only consistency, measured aggregation and inference

This patch **does not rerun or modify training**, change model hyperparameters,
or modify historical `results/real` JSON files. It reads those files through
explicit suite run IDs and verifies REAL provenance sidecars.

## Changes

- The paper reference ratio is fixed at **1:50** so it cannot silently disagree with the manuscript. The Pareto analyses for 1:10 and 1:100 are exported under separate names; a second set of explicitly labeled figures uses validation-selected checkpoint test F1 for comparison, never merged with the primary last-checkpoint results.
- Figure 1 now shows the validation F1 and validation BCE curves averaged across matched seeds, with ±1 SD shading; the title and main-paper caption describe the validation split and 1:50 retention setting.
- Figure 5 now uses the **last checkpoint TEST F1** and measured total wall-clock per seed, restricted to one positive-retention condition and the Axis-1 campaign. `fig_pareto_cost_vs_accuracy` is the primary reference ratio; the other ratios get `fig_pareto_cost_vs_accuracy_r1_10` and `_r1_100`.
- The Scenario-2 table includes Uniform, Uniform Grouped, Coverage-Balanced MCTS and Enhanced MCTS. The table's raw p-values are expressly exploratory.
- The manuscript's experimental setup, results, discussion and conclusion now describe all three positive-retention settings and do not confuse them with literal class ratios.
- `fig_class_exposure_concentration` reports whole-run Shannon-effective class count and class exposure Gini, both from recorded class-exposure vectors across matched Scenario-2 seeds. This is not a per-step diversity history; if the raw arrays are missing, the figure is omitted.
- `tab_paired_effects.tex` reports seed-matched differences in percentage points, 95% paired Student-t confidence intervals and **Holm-adjusted p-values** across all 12 planned comparisons (9 Scenario-1 F1 and 3 Scenario-2 Top-1), when all 3 ratios are available. The original primary tables contain raw, exploratory p-values.
- No p-values are computed from unpaired summary arrays, and no plots are generated using guesses about missing runs.

## Limitations

- With only five seeds, paired-t intervals and p-values are sensitive to distributional assumptions. Non-significance never means equivalence. Constant paired differences receive a conservative raw p=1 and are tagged in the supplementary table.
- Aggregating wall-clock time assumes comparable hardware and system load; a Pareto diagram of mean wall-clock should not be presented as statistically significant dominance.
- The configured labels `1:10`, `1:50`, `1:100` are percentages of retained positive training examples, not empirical positive:negative pool ratios.
- `pdflatex` must still be installed to build the paper PDF. Author placeholders must be filled manually.

## Apply

From the root of the project, copy/extract the patch ZIP and run:

```bash
python3 apply_mcts_hnm_patch3.py --check
python3 apply_mcts_hnm_patch3.py --emit-patch /tmp/mcts_phase3.patch
git apply --check /tmp/mcts_phase3.patch
git apply /tmp/mcts_phase3.patch
python3 -m pytest tests/test_phase3_publication.py -q
python3 -m pytest tests/ -q --tb=short
git diff --check
```

Then rerun **notebook 05 only** in REAL mode. No need to rerun notebooks 00–04.

The patch application script never changes repository files; only `git apply` does.
