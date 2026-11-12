# MCTS-HNM — Phase 2: measured evidence before rerunning notebooks

## Scope

This patch is designed for the November 2026 `main` code after Phase 1. It does
**not** rerun training during application. It adds measurement collection for the
next full run of notebooks 00–05 while retaining strict SMOKE/REAL separation.

### What is fixed

- **Figure 1** uses *validation* F1 and dynamic axes (no clipping above 0.65).
- **Figure 2** limits displayed x ticks without dropping any heatmap samples.
- **Figure 8** uses Ward on Euclidean row-normalized confusion vectors, matching
  `build_confusion_tree()`. The recovered confusion-matrix provenance remains
  explicit and the plot must not be presented as a reconstruction of a different
  algorithm.
- **Figure 6** measures cumulative class coverage (>=1 and >=5 selected samples)
  at each evaluation checkpoint, along with real MCTS visitation entropy. It
  plots means and SD across 2+ seeds, requiring fully measured checkpoints.
- **Figure 7** is produced by the Phase-1 plotter once new atomic Scenario-2 runs
  retain exposure counts and per-class recall. Pairing is by seed.
- **Figure 10** stores exact test PR points after training and model selection.
  It aggregates measured curves across paired seeds, reports original sklearn
  **Average Precision** (not the trapezoidal area of interpolated curves).
- **Figure 11** uses per-seed post-training evaluations of final and
  validation-selected checkpoints at fixed and validation-calibrated thresholds.
  The Phase-1 plotter is re-used; nothing is synthesized.
- Atomic `run_id` includes `benchmark_name` so an Axis-2 run cannot overwrite an
  Axis-1 run for the same strategy, class, ratio and seed.
- Global inferential comparisons that inadvertently pool multiple classes or
  ratios are suppressed; the ratio-specific analysis is retained.

### Optional ratio expansion, not automatic

The notebook 03 continues to run only the configured positive-subset ratio
`1:50` unless you explicitly set an environment variable **before starting
Jupyter**, for example:

```bash
export MCTS_IMBALANCE_RATIOS=1:10,1:50,1:100
jupyter lab
```

These values control the positive subsampling fractions, not the empirical
ratio of all positive to all negative training-pool instances. The baseline
1:50 experiment has only about nine training positives out of 450 available.
For resource planning, 3 ratios × 6 strategies × 5 seeds is a sizeable run;
proceed only if this is intended. The Axis-2 experiment remains on its original
canonical configuration.

### Applying

```bash
python3 apply_mcts_hnm_phase2_patch.py --check
python3 apply_mcts_hnm_phase2_patch.py --emit-patch /tmp/mcts_phase2.patch
git apply --check /tmp/mcts_phase2.patch
git apply /tmp/mcts_phase2.patch
python3 -m pytest tests/test_phase2_evidence.py -q
python3 -m pytest tests/ -q --tb=short
```

Close and restart Jupyter kernels after applying, and re-run notebooks in order
00, 01, 02, 03, 04, 05 as needed. Do **not** blindly overwrite old results;
backup `results/real/` beforehand and confirm the requested mode and ratio.

### Provenance and limitations

- No smoke results are accepted as REAL evidence. No figure is generated from
  constants, AP alone, or unverified runs.
- Old runs lack the new fields. As designed, Phase-2 plots will skip them.
- Figures 6/7/10/11 require a new notebook-03 benchmark execution, not just 05.
- Figure 9 needs at least 2 measured ratios per strategy. It will be absent
  unless the optional expansion is enabled.
- Test PR curves and evaluation quadrants are for final reporting, not for
  changing hyperparameters or selecting checkpoints.
- The 5-seed mean ± SD is dispersion, not a 95% confidence interval.
- Cached confusion matrices must be saved by notebook 02 with sidecar metadata.
- `pdflatex` and author details are independent system/manuscript tasks.
