# Paper review against saved notebook outputs

The manuscript was revised using the natural-language-editor skill in Moderate Mode. The review used saved notebook cells and outputs, the exported paper tables and figures, the artifact manifest, and the current implementation. Experiments were not rerun. Displayed equations, citation keys, labels, and numerical table entries were preserved.

## Evidence available in this checkout

| Notebook | Saved evidence | Use in the paper |
| --- | --- | --- |
| 00, master pipeline | SMOKE execution with short synthetic runs | Excluded from scientific results |
| 01, data and embeddings | REAL execution; 45,000 training, 5,000 validation, and 10,000 test embeddings, each with 512 features | Supports split sizes and positive-pool counts |
| 02, tree construction | REAL semantic and confusion-tree construction | Supports tree construction checks; does not establish which linkage mode produced the separate benchmark |
| 03, Tier-1 benchmarks | REAL execution; seeds 42--46; 1,000 binary and 2,000 multiclass steps | Main experimental evidence; saved means agree with exported tables |
| 04, end-to-end training | Two-seed, 100-step quarter-width ResNet comparison; separate single-seed, 30-step expansion check | Implementation checks, with no five-seed end-to-end performance claim |
| 05, paper artifacts | REAL export recorded as complete; compilation unavailable on the original machine | Records exported artifacts; the revised manuscript compiles locally |

Before editing, all 40 exported artifacts matched their SHA-256 hashes in `paper/artifact_manifest.json`. The raw REAL JSON files referenced by that manifest are absent from this checkout. Matching export hashes establishes consistency with the recorded exports; it does not independently validate the source measurements or statistical calculations.

## Results and interpretation

The binary labels describe retention of 10%, 2%, and 1% of the available target-class training examples. The resulting pools contain 45, 9, and 4 positives, respectively, against 44,550 negatives. They are not literal positive-to-negative population ratios. Notebook 01 also prepares oak_tree splits, but the main sampler benchmark evaluates beaver only.

Among the four principal binary samplers, OHEM has the highest mean F1 at 10% retention (79.5%). Strict MCTS has the highest mean at 2% and 1% retention (69.7% and 60.0%). None of the nine listed binary F1 comparisons against Enhanced survives the exported Holm correction. This does not establish equivalence or rank all pairwise comparisons.

At 2% retention, the mechanism table reports AUPRC 0.736 for Strict, 0.684 for Q-normalization alone, 0.734 for Dual-Stream alone, and 0.685 for their combination. These point estimates do not support the original introduction's guarantees about stabilization or improved representations. The paper now describes those mechanisms as design choices and reports their observed performance separately.

For multiclass classification, the saved means are 63.854% Top-1 for Uniform, 52.240% for Uniform Grouped, 53.058% for Coverage-Balanced MCTS, and 33.434% for Enhanced. All four reach full cumulative class coverage. The three exported Top-1 contrasts against Enhanced remain significant after Holm correction. Those contrasts do not test Coverage-Balanced against Uniform Grouped or identify a single cause of the deficit.

## Issues requiring care

- **Hierarchy reproducibility:** The original manuscript described Ward linkage on a confusion-distance matrix. The current default clusters row-normalized confusion vectors in Euclidean space and exposes a separate legacy mode. Saved tree depths also differ between Notebooks 02 and 03. The revised method distinguishes the current implementation from the unresolved historical benchmark construction; it does not assume they are identical.
- **Statistical language:** Notebook 03's saved hierarchy verdict refers to a null hypothesis of equivalence. A non-significant difference test does not establish equivalence. Its multiclass code also labels descriptive comparisons as causal confirmation. Neither interpretation is carried into the revised paper.
- **Evaluation protocols:** Final-checkpoint and validation-selected-checkpoint results, and fixed and validation-calibrated thresholds, are distinct outcomes. The manuscript keeps them separate. Sensitivity outputs should not be pooled across those protocols.
- **Replication:** The short Tier-2 and leaf-expansion checks do not meet the repository's five-seed standard for experimental performance claims. They are described as implementation checks without presenting their means as benchmark evidence.
- **Historical provenance:** Raw REAL runs are needed to independently recompute standard deviations, paired confidence intervals, and corrected p-values. The exported table calls its comparisons prespecified; their timing cannot be independently verified from this checkout.
- **Author metadata:** The corresponding email and co-author remain placeholders. They were preserved because replacement details were not supplied.

## Editorial changes and validation

The introduction and related work now state the sampling problem directly, without unsupported claims that OHEM necessarily causes instability or that the proposed mechanisms guarantee improvements. Methods distinguish the mined-sub-batch reward from the full optimizer loss. Results explain the corrected and uncorrected comparisons, while the discussion limits conclusions to the available experiments.

Table captions clarify retention labels and evaluation protocols. Three wide tables were fitted to the text width. Their numerical rows are unchanged. These editorial table edits intentionally differ from the historical hashes in the original artifact manifest, which was preserved as an export record. Running Notebook 05 again may overwrite the edited captions and table formatting because its generator was not changed.

Validation checked numerical table rows against the original files, preserved display equations and citation keys, and rebuilt `paper/main.pdf` with `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`. The final LaTeX log contains no undefined references or overfull boxes.
