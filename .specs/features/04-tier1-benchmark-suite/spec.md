# F4: Downstream Classifiers, Baselines & Tier-1 Benchmark Notebook Specification

**Approved**: 2026-11-09 - user: "prossiga"

## Problem Statement

With the CIFAR-100 embedding pipeline (`F1`), Dual Hierarchy Construction Engine (`F2`), and Core MCTS-HNM Agent (`F3`) verified, we need a unified downstream classification and multi-seed benchmark suite to empirically test our central scientific hypotheses (`AD-001`, `AD-002`, `AD-004`, `AD-005`). Specifically, we must compare Uniform Random Sampling, Online Hard Example Mining (OHEM), Strict GiantHunter MCTS (`q_norm=False`, `dual_stream_ratio=0.0`), and Enhanced MCTS-HNM (`q_norm=True`, `dual_stream_ratio=0.5`) across both Scenario 1 (Imbalanced Binary One-vs-Rest) and Scenario 2 (100-way Multi-class Curriculum), evaluate Semantic vs. Ward Confusion hierarchies, and run paired statistical significance tests across $N \ge 5$ random seeds from a versioned Jupyter notebook (`notebooks/03_tier1_mcts_benchmarks.ipynb`, `AD-008`).

## Goals

- [x] Implement the Tier-1 2-layer MLP classification head (`src/models/mlp_head.py`) and `Adam(lr=1e-3)` optimizer factory matching GiantHunter's supervision setup (`ISS-10`, `AD-001`).
- [x] Implement `UniformSampler` and `OHEMSampler` (`src/training/baselines.py`) as non-hierarchical baseline samplers for both Scenario 1 and Scenario 2 (`ISS-10`).
- [x] Implement unified Scenario 1 (`BCEWithLogitsLoss`, evaluating F1, Precision, Recall, AUPRC, FPR@95) and Scenario 2 (`CrossEntropyLoss(label_smoothing=0.1)`, evaluating Top-1 Accuracy, Macro-F1, Worst-10-Class Recall) training loops and metric evaluators in `src/training/trainer.py` (`ISS-11`, `AD-004`).
- [x] Implement paired statistical significance tests (`src/utils/statistics.py`: paired Student's $t$-test and Wilcoxon signed-rank test) and the programmatic multi-seed benchmark runner (`src/training/benchmark_runner.py`) exporting atomic JSON and CSV results to `results/` (`ISS-12`).
- [x] Deliver versioned notebook `notebooks/03_tier1_mcts_benchmarks.ipynb` orchestrating the 3 experimental axes (Sampler Comparison, Semantic vs. Ward Confusion Tree Comparison, and $c$/$\alpha$ Hyperparameter Ablations) across $N = 5$ seeds (`ISS-12`, `AD-008`).

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
| ------- | ------ |
| Dynamic leaf sub-clustering by prediction entropy ($n_l \ge N_{\text{split}}$) | Scoped as P2 extension in Feature `F5` (`05-extensions-and-e2e`, `AD-006`). |
| Tier-2 end-to-end ResNet-18 fine-tuning on raw $32 \times 32$ RGB images | Scoped in Feature `F5` (`05-extensions-and-e2e`, `AD-001`). |
| Publication LaTeX booktabs table formatting (`paper/tables/*.tex`) and vector PDF figure compilation for `paper/main.tex` | Scoped in Feature `F6` (`06-paper-and-artifacts`, `AD-007`). |

---

## Assumptions & Open Questions

Every ambiguity is resolved or recorded here.

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| Tier-1 MLP architecture & optimizer (`ISS-10`) | 2-layer MLP `512 -> hidden_dim (128) -> ReLU -> Dropout(0.1) -> num_outputs` (`1` for Scenario 1, `100` for Scenario 2) trained with `torch.optim.Adam(lr=1e-3)` | Matches `.specs/features/00-epic-roadmap/context.md`, `ARCHITECTURE.md` Section 5, and `FuchuanQu/GiantHunter` (`Adam(lr=0.001)`) | y |
| OHEM baseline candidate pool size (`ISS-10`) | Sample a candidate pool of size $M_{\text{cand}} = \min(|\mathcal{D}_{\text{pool}}|, \, r_{\text{ohem}} \cdot B_{\text{neg}})$ with default multiplier $r_{\text{ohem}} = 4$, evaluate per-sample loss under `torch.no_grad()`, and select the top-$B_{\text{neg}}$ highest-loss instances | Standard Online Hard Example Mining (Shrivastava et al., CVPR 2016) adapted to cached feature embeddings with bounded per-step forward overhead | y |
| Scenario 2 MCTS vs. Dual-Stream batching (`ISS-11`, `AD-004`, `AD-005`) | In Scenario 2, all 100 fine classes are leaves in `FastHierarchyTree` (`pos_batch_size=0`, `neg_batch_size=B`); `MCTSHNMAgent` selects a hard class leaf $l$, `DualStreamBatchSampler` combines $\lfloor(1-\beta)B\rfloor$ samples from leaf $l$ with $\lceil\beta B\rceil$ uniform samples across all 100 classes, and `compute_mcts_reward(..., is_binary=False)` computes the mean multi-class cross-entropy on the MCTS sub-batch | Unifies Scenario 1 and Scenario 2 under the exact same `MCTSHNMAgent` and `DualStreamBatchSampler` APIs verified in Feature `F3` | y |
| FPR@95 calculation in Scenario 1 (`ISS-11`) | Using `sklearn.metrics.roc_curve(y_true, y_scores)`, return the minimum `fpr` among all thresholds where `tpr >= 0.95` (returning `1.0` if no threshold reaches `0.95` recall, or `0.0` if all positives are ranked above all negatives) | Standard OOD / imbalanced detection definition of False Positive Rate at 95% True Positive Rate (`ARCHITECTURE.md` Section 6) | y |
| Paired statistical test zero-difference fallback (`ISS-12`) | When comparing two methods across $N \ge 5$ seeds and all paired differences are identically zero, return `p_value = 1.0` and `statistic = 0.0` without raising a SciPy `ValueError` | Prevents crashes on deterministic or tied metrics during unit tests and flat baseline regimes | y |
| Remaining implicit-requirement dimensions (auth, rate limits, TTL/expiry, external network calls, concurrency) | N/A for single-process PyTorch/NumPy Tier-1 training and statistical evaluation | Local/JupyterHub compute pipeline operating on cached tensors in memory/disk | y |

**Open questions:** none - all resolved and recorded above.

---

## User Stories

### P1: Tier-1 MLP Classifier & Non-Tree Baseline Samplers (`ISS-10`) ⭐ MVP

**User Story**: As an ML researcher benchmarking hierarchical MCTS against standard non-tree sampling strategies, I want a modular 2-layer MLP head (`src/models/mlp_head.py`) and deterministic `UniformSampler` and `OHEMSampler` classes (`src/training/baselines.py`) so that all sampling methods share an identical neural network architecture and optimizer setup.

**Why P1**: Provides the downstream neural classifier and the two primary non-hierarchical baselines (`Uniform` and `OHEM`) required by `EPIC-04` (`AD-001`, `AD-005`).

**Acceptance Criteria**:

1. WHEN `Tier1MLPHead(input_dim=512, hidden_dim=128, num_outputs=K, dropout=0.1)` is called on a 2D feature tensor of shape `[B, 512]` THEN the system SHALL return raw logits of shape `[B, K]` (or `[B]` when `squeeze_binary=True` and `num_outputs=1`).
2. WHEN `UniformSampler.sample_batch()` is called THEN the system SHALL return `pos_batch_size` positive indices and `neg_batch_size` negative indices sampled uniformly at random in Scenario 1 (or `neg_batch_size` indices sampled uniformly across all active classes in Scenario 2).
3. WHEN `OHEMSampler.sample_batch(model, features, labels, is_binary)` is called THEN the system SHALL draw a candidate pool of size $\min(|\text{pool}|, \, r_{\text{ohem}} \cdot B_{\text{neg}})$, compute per-sample losses under `torch.no_grad()`, and select the $B_{\text{neg}}$ candidate indices with the largest losses.
4. IF `Tier1MLPHead`, `UniformSampler`, or `OHEMSampler` is initialized with non-positive dimensions/batch sizes, `dropout < 0` or `dropout >= 1`, or `ohem_ratio < 1` THEN the system SHALL raise a `ValueError`.

**Independent Test**: Run `pytest tests/test_training_loop.py -k "mlp or baselines"` to verify `Tier1MLPHead` output shapes/gradients, `UniformSampler` index bounds, and that `OHEMSampler` deterministically selects the highest-loss samples from a known synthetic loss distribution.

---

### P1: Scenario 1 & Scenario 2 Unified Trainer & Evaluation Metrics (`ISS-11`) ⭐ MVP

**User Story**: As a researcher evaluating MCTS-HNM under both binary imbalance and multi-class curriculum learning (`AD-004`), I want unified training and evaluation functions in `src/training/trainer.py` supporting `uniform`, `ohem`, `mcts_strict`, and `mcts_enhanced` modes so that both classification regimes log standardized performance and RL diagnostic trajectories.

**Why P1**: Implements the core training loop coupling the downstream classifier with MCTS Selection, Rollout, Backpropagation, and periodic evaluation (`EPIC-04`, `AD-004`, `AD-005`).

**Acceptance Criteria**:

1. WHEN `train_scenario1()` is executed for a strategy in `{"uniform", "ohem", "mcts_strict", "mcts_enhanced"}` THEN the system SHALL optimize `Tier1MLPHead(num_outputs=1)` with `torch.nn.BCEWithLogitsLoss()` and record evaluation snapshots containing `f1`, `precision`, `recall`, `auprc`, `fpr_at_95_recall`, and (for MCTS strategies) `entropy_nats` and `normalized_entropy` at every `eval_every` steps.
2. WHEN `train_scenario2()` is executed for a strategy in `{"uniform", "ohem", "mcts_strict", "mcts_enhanced"}` THEN the system SHALL optimize `Tier1MLPHead(num_outputs=num_classes)` with `torch.nn.CrossEntropyLoss(label_smoothing=0.1)` and record evaluation snapshots containing `top1_accuracy`, `macro_f1`, `worst_10_class_recall`, and (for MCTS strategies) `entropy_nats` and `normalized_entropy` at every `eval_every` steps.
3. WHILE training in `"mcts_strict"` or `"mcts_enhanced"` mode, the system SHALL update the tree via `agent.update(leaf_node_id, reward)` on every training step using only the MCTS leaf sub-batch logits (`batch.mcts_indices`) so that high-loss classes accumulate higher $Q$-values and visit counts than low-loss classes.
4. IF `train_scenario1()` or `train_scenario2()` receives an unknown strategy name, `num_steps <= 0`, `eval_every <= 0`, or mismatched feature/label lengths THEN the system SHALL raise a `ValueError`.

**Independent Test**: Run `pytest tests/test_training_loop.py -k "scenario1 or scenario2"` on a synthetic hierarchical embedding fixture to verify that MCTS concentrates leaf visits on hard classes, decreases training loss, and populates all required Scenario 1 and Scenario 2 metric keys.

---

### P1: Multi-Seed Benchmark Runner, Statistical Tests & Notebook 03 (`ISS-12`) ⭐ MVP

**User Story**: As an experimenter running on a remote JupyterHub (`AD-008`), I want a multi-seed benchmark runner (`src/training/benchmark_runner.py`), a paired statistical test suite (`src/utils/statistics.py`), and a versioned notebook (`notebooks/03_tier1_mcts_benchmarks.ipynb`) so that $N = 5$ seed comparisons, $p$-values, and structured JSON/CSV result logs are generated reproducibly.

**Why P1**: Fulfills `EPIC-04` Acceptance Criterion 3 and `AD-008` by automating multi-seed aggregation, statistical hypothesis testing, and interactive notebook execution.

**Acceptance Criteria**:

1. WHEN `compute_paired_statistical_tests(scores_a, scores_b)` is called with paired multi-seed metric arrays of equal length $N \ge 2$ THEN the system SHALL return the mean and standard deviation of each method, mean paired difference, paired Student's $t$-test (`ttest_rel`) statistic and $p$-value, and Wilcoxon signed-rank (`wilcoxon`) statistic and $p$-value.
2. WHEN `run_benchmark()` is invoked across a list of random seeds (e.g., `seeds=(42, 43, 44, 45, 46)`) and strategies THEN the system SHALL execute each `(strategy, tree_type, seed)` run deterministically, compute aggregated `mean ± std` and pairwise statistical comparisons against `mcts_enhanced`, and atomically persist structured JSON and CSV summaries to `output_dir`.
3. WHEN `notebooks/03_tier1_mcts_benchmarks.ipynb` is executed end-to-end THEN the notebook SHALL run without errors, execute the Scenario 1, Scenario 2, Semantic vs. Confusion Tree, and $(c, \alpha)$ ablation benchmarks via `src.training.*` and `src.utils.statistics`, render inline summary tables and plots, and save JSON/CSV artifacts under `results/`.
4. IF `compute_paired_statistical_tests()` is called with arrays of unequal length, length $< 2$, or non-finite values (`NaN`/`Inf`) THEN the system SHALL raise a `ValueError`.

**Independent Test**: Run `pytest tests/test_training_loop.py -k "statistics or benchmark_runner or notebook"` to verify exact $t$-test/Wilcoxon $p$-values, atomic JSON/CSV persistence, and end-to-end execution of `notebooks/03_tier1_mcts_benchmarks.ipynb`.

---

## Edge Cases

- IF all paired differences between two methods across seeds are identically zero ($a_i - b_i = 0$ for all $i$) THEN `compute_paired_statistical_tests()` SHALL return `t_stat = 0.0`, `t_pvalue = 1.0`, `wilcoxon_stat = 0.0`, and `wilcoxon_pvalue = 1.0` without raising a zero-division or empty-difference exception.
- IF a class has zero positive predictions during Scenario 1 or Scenario 2 metric evaluation THEN precision and F1 calculations SHALL use `zero_division=0.0` so no `UndefinedMetricWarning` or `NaN` is produced.
- IF `OHEMSampler` is configured with `ohem_ratio * neg_batch_size` exceeding the total available pool size THEN the system SHALL clamp the candidate pool size to the available pool size and sample without replacement when `pool.size >= neg_batch_size` (or with replacement if `pool.size < neg_batch_size`).
- WHEN `"mcts_strict"` mode is selected (`dual_stream_ratio=0.0`), `DualStreamBatchSampler` sets $K_{\text{mcts}} = B_{\text{neg}}$ and $K_{\text{uni}} = 0$, and the trainer computes the MCTS reward over all $B_{\text{neg}}$ negative samples in the batch.

---

## Requirement Traceability

Each requirement maps to a unique ID for tracking across design, tasks, and verification.

| Requirement ID | Story | Phase | Status |
| -------------- | ----- | ----- | ------ |
| BENCH-01 | P1: Tier-1 MLP Classifier & Non-Tree Baseline Samplers (`ISS-10`) | Execute | Verified |
| BENCH-02 | P1: Scenario 1 & Scenario 2 Unified Trainer & Evaluation Metrics (`ISS-11`) | Execute | Verified |
| BENCH-03 | P1: Multi-Seed Benchmark Runner, Statistical Tests & Notebook 03 (`ISS-12`) | Execute | Verified |

**Coverage:** 3 total requirements mapped to `EPIC-04`.

---

## Differences from the source

Every departure from the original *GiantHunter* paper (`btaf239.pdf`) and official repository (`FuchuanQu/GiantHunter`) is linked to an active architectural decision in `.specs/STATE.md`.

| Source shows | Product shows | Confirmed |
| ------------ | ------------- | --------- |
| Official `FuchuanQu/GiantHunter` repo uses a 1-layer Transformer on genomic Protein-Cluster tokens | Uses `Tier1MLPHead` (`512 -> 128 -> K`) on frozen ResNet-18 CIFAR-100 embeddings for fast 5-seed RL ablations | y - AD-001 |
| GiantHunter benchmarks only against a static taxonomy tree without OHEM or Confusion Tree baselines | Benchmarks Uniform Sampling, OHEM, Strict GiantHunter, and Enhanced MCTS-HNM across both Semantic and Ward Confusion trees | y - AD-002 |
| GiantHunter trains only with binary `BCEWithLogitsLoss` | Implements both Scenario 1 (`BCEWithLogitsLoss`) and Scenario 2 (`CrossEntropyLoss(label_smoothing=0.1)`) | y - AD-004 |
| GiantHunter executes via a monolithic CLI script (`gianthunter.py`) | Exposes modular Python APIs (`src/models/`, `src/training/`, `src/utils/statistics.py`) orchestrated by `notebooks/03_tier1_mcts_benchmarks.ipynb` | y - AD-008 |

## Sources

- `.specs/features/00-epic-roadmap/spec.md` (`EPIC-04`) - parent epic specification and acceptance criteria for Feature `F4`.
- `ROADMAP.md` (`Feature F4`, `ISS-10`–`ISS-12`, `T4.1`–`T4.6`) - planned issue and task decomposition.
- `ARCHITECTURE.md` (`Section 4: Dual-Stream Mini-Batch Training`, `Section 5: Two-Tier Execution`, `Section 6: Evaluation Metrics & Scientific Rigor`) - model architecture, evaluation metrics, and statistical testing specifications.
- `../btaf239.pdf` (*Qu et al., Bioinformatics 2025, 41, i30–i39*) - binding reference for `Adam(lr=1e-3)` and `BCEWithLogitsLoss` training conventions.

---

## Success Criteria

- [x] `.venv/bin/pytest -v tests/test_training_loop.py` passes 100% of unit and integration checks covering `BENCH-01`, `BENCH-02`, and `BENCH-03`.
- [x] `train_scenario1()` and `train_scenario2()` shift MCTS leaf visitation toward harder classes (higher loss) compared to uniform visitation and export finite metric trajectories (`f1`, `auprc`, `fpr_at_95_recall`, `top1_accuracy`, `macro_f1`, `worst_10_class_recall`, `entropy_nats`).
- [x] `run_benchmark()` executes multi-seed experiments deterministically, computes paired Student's $t$-test and Wilcoxon signed-rank $p$-values via `src/utils/statistics.py`, and writes valid JSON and CSV summaries to disk.
- [x] `notebooks/03_tier1_mcts_benchmarks.ipynb` executes cleanly from top to bottom via `nbclient` / `jupyter` without errors and saves structured benchmark artifacts under `results/`.
