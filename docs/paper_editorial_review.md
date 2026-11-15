# Paper review against supplied execution results

Updated after the REAL execution files were supplied. The manuscript follows the natural-language-editor skill in Moderate Mode. Effective passages, citations, displayed equations, and the primary numerical results were preserved. This update recomputes summaries from saved outcomes; it does not rerun training or modify raw result files.

## Evidence checked

The review selects 90 binary sampler records, 10 hierarchy records, and 20 multiclass records by the run IDs listed in their campaign summaries. Each setting contains seeds 42–46. The sensitivity summary contains 16 configurations with five per-seed outcomes each. The two-seed end-to-end summary remains an implementation check. One hundred historical atomic records without a campaign name are excluded from the main aggregation.

The 116 available source files referenced by the historical artifact manifest match their SHA-256 hashes. The validation confusion matrix and its sidecar are the two missing entries. The original manifest is retained as an export record. Derived values and hashes of all supplied raw result files are recorded in `paper/result_verification.json`.

| Evidence | Verification | Manuscript change |
| --- | --- | --- |
| Binary benchmark | Campaign summaries agree with individual final metrics; histories end at 1,000 updates | Principal numerical results retained |
| Multiclass benchmark | Campaign summaries agree with individual final metrics; histories end at 2,000 updates | Principal numerical results and full cumulative coverage retained |
| Paired comparisons | Independent Student-t calculations and Holm adjustment agree with publication functions | Same nine binary F1 and three multiclass Top-1 contrasts |
| Mechanism ablations | Means, sample SDs, paired main effects, and interaction recomputed | Dispersion added to the ablation table; exploratory effects tabulated separately |
| Hierarchy comparison | Paired differences, confidence intervals, and raw p-values recomputed | Numerical comparison verified; non-significance does not establish equivalence |
| Checkpoint and threshold outcomes | Four evaluations recomputed for each principal sampler at 2% retention | New table separates final and validation-selected checkpoints and fixed and calibrated thresholds |
| Computational cost | Durations and operation counts recomputed at 2% retention | New table reports training-loop time and candidate-example forwards |
| Sensitivity sweep | All 16 F1/AUPRC means and sample SDs agree with five per-seed outcomes | Plot protocol identified; observed best test mean remains exploratory |

## Findings carried into the paper

The labels 1:10, 1:50, and 1:100 denote retention of 10%, 2%, and 1% of available positive examples, producing 45, 9, and 4 positives against 44,550 negatives. The evaluated target remains beaver. The supplied files do not add replicated target classes or a five-seed end-to-end benchmark.

Among the four principal binary samplers, OHEM has the highest mean F1 at 10% retention; Strict MCTS has the highest at 2% and 1%. None of the nine binary F1 contrasts against Enhanced survives the 12-test Holm correction. The three multiclass Top-1 contrasts against Enhanced remain significant under that correction. This family does not test every method pair.

At 2% retention, the exploratory normalization main effect on AUPRC is -0.050364, with between-seed SD 0.024237 and unadjusted 95% paired-t interval [-0.080458, -0.020270]. The Dual-Stream main effect is -0.000597, SD 0.015054, interval [-0.019289, 0.018096]. The interaction is 0.003195, SD 0.029988, interval [-0.034040, 0.040430]. Main effects average paired on-minus-off differences over the other factor. The interaction is the difference between Dual-Stream effects with and without normalization. These estimates are outside the primary correction family.

The Semantic-minus-Ward AUPRC difference is 0.009906, SD 0.054044, with 95% paired-t interval [-0.057198, 0.077010] and raw p=0.702912. Semantic maximum and mean leaf depths are 2; the benchmark Ward maximum depth is 39 and mean leaf depth is 17.4545. The comparison changes both hierarchy information and topology.

Enhanced final-checkpoint F1 at 2% retention is 52.4% ± 9.4% with a fixed threshold and 68.0% ± 3.1% with a validation-calibrated threshold. At the checkpoint selected by validation AUPRC, the values are 69.3% ± 2.4% and 70.8% ± 2.9%. These are means and sample SDs over the same five seeds, not independent replications.

Each principal binary run uses 1,000 updates and 64,000 gradient-sample exposures. OHEM records 192,000 additional candidate-example forwards. Mean recorded times range from 8.31 to 9.01 seconds. The trainer's timer covers training and periodic validation, ending before final test evaluation; feature extraction is also excluded. Captions now say training-loop time instead of total whole-run time. Zero extra candidate forwards does not mean zero model computation.

## Resolved and remaining reproducibility issues

The earlier claim that REAL results were absent has been removed. Means, dispersion, and paired statistics can now be recomputed from the supplied records.

The source revision recorded in the result metadata, `2f3af6b301c2d10cc616fc2c5fa749038f821c3e`, is available in Git. Its benchmark calls `build_confusion_tree` without a linkage override, and the default Ward implementation clusters row-normalized confusion vectors in Euclidean space. This resolves the algorithm choice at the recorded source level. The absent validation confusion matrix still prevents reconstruction of the exact benchmark hierarchy. A recorded revision does not independently establish that the working tree was unchanged during execution.

All 120 selected campaign sidecars retain the orchestration default of 100 steps; many also retain seed 42 for records with other seeds. Individual records and histories instead show the five seeds and 1,000 or 2,000 updates. The discrepancy is recorded in the verification JSON and disclosed in the limitations. The sidecars alone are insufficient as a per-run configuration audit. No raw metadata were rewritten.

The files do not independently establish when the 12 contrasts were specified, so the paper no longer calls them prespecified. Hierarchy, mechanism, and sensitivity analyses remain exploratory. The largest observed sensitivity mean is a test outcome after validation-based checkpoint selection; choosing hyperparameters from those test results requires a fresh evaluation before claiming a final estimate.

The short Tier-2 and leaf-expansion checks do not meet the main benchmark's five-seed replication level. Author placeholders remain because replacement information was not supplied.

## Validation and generated artifacts

The analysis checks campaign/individual metric agreement, history endpoints, update counts, paired Student-t intervals, Holm values, and sensitivity means and sample SDs. Displayed equations, existing labels, citation keys, and primary table values are checked against the manuscript before this update. The revised LaTeX compiles; the final log contains no unresolved references or overflowing boxes.

New tables and editorial captions are manuscript files. Running Notebook 05 may overwrite tables controlled by its unchanged generator. The historical artifact manifest is preserved; `paper/result_verification.json` records the evidence used for this update.
