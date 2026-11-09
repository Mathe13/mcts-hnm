# MCTS-HNM Master Epic & Project Roadmap Specification

**Approved**: 2026-11-08 - user: "aprovado"

## Problem Statement

Training deep classifiers on datasets with hierarchical class structures and extreme class imbalance suffers from inefficient uniform negative sampling and myopic batch-level hard negative mining. While *GiantHunter* (*Bioinformatics*, 2025) introduced a Monte Carlo Tree Search (MCTS) framework to dynamically mine informative negative training samples from a taxonomy tree, its official repository (`FuchuanQu/GiantHunter`) only releases inference code and omits the MCTS training implementation, and its generalization beyond viral metagenomics remains untested. This epic defines the roadmap, sub-features (`F1`–`F6`), issues, and atomic tasks required to build a clean, modular Python library (`src/`) orchestrated end-to-end via versioned Jupyter notebooks (`notebooks/`) on a remote JupyterHub — from CIFAR-100 download and embedding extraction through Semantic vs. Confusion tree benchmarks to automated generation of paper artifacts and the LaTeX manuscript.

## Goals

- [ ] Deliver a modular, unit-tested MCTS-HNM Python package (`src/`) with zero bioinformatics dependencies, keeping all complex algorithmic and training logic in `.py` modules while exposing clean programmatic APIs (`AD-008`).
- [ ] Provide versioned Jupyter notebooks (`notebooks/*.ipynb`) that orchestrate the entire experimental workflow end-to-end on a remote JupyterHub: CIFAR-100 download $\rightarrow$ embedding extraction $\rightarrow$ tree construction $\rightarrow$ Tier-1 & Tier-2 MCTS benchmarks $\rightarrow$ paper artifact generation (`AD-008`).
- [ ] Benchmark Semantic Hierarchy (CIFAR-100 20 superclasses $\rightarrow$ 100 classes) against Data-Driven Confusion Hierarchy (Ward clustering on validation confusion matrices) (`AD-002`).
- [ ] Evaluate both Scenario 1 (Imbalanced Binary One-vs-Rest with `BCEWithLogitsLoss`, faithful to GiantHunter) and Scenario 2 (100-way Multi-class Curriculum with Cross-Entropy and label smoothing) across $N \ge 5$ random seeds (`AD-004`).
- [ ] Quantify the ablation gain of Enhanced MCTS-HNM (Min-Max $Q$-normalization + 50/50 Dual-Stream batching) over Strict GiantHunter MCTS (raw $Q$-values + 100% MCTS batches), Uniform Random Sampling, and Online Hard Example Mining (OHEM) (`AD-005`).
- [ ] Automate the generation of publication figures, statistical tables ($p$-values from paired $t$-tests and Wilcoxon tests), and the complete LaTeX manuscript in the arXiv/Prime AI template (`AD-007`).

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
| ------- | ------ |
| Bioinformatics sequence tools (Prodigal, DIAMOND, MCL, HMMER) | Excluded to isolate Reinforcement Learning dynamics from genomic preprocessing bottlenecks (`AD-001`, `AD-002`). |
| Metagenomic FASTA datasets (NCBI RefSeq / Yangtze River contigs) | Replaced by the standardized CIFAR-100 hierarchical benchmark (`AD-002`). |
| Monolithic business logic inside `.ipynb` notebook cells | Forbidden by `AD-008`: complex logic must live in modular `.py` files under `src/`; notebooks are strictly for orchestration, visualization, and interactive execution. |
| Interactive web UI / Human-in-the-Loop annotation portal | Project targets Modalidade 1 & 2 (benchmark generalization and algorithmic modification), not Modalidade 3. |

---

## Assumptions & Open Questions

Every ambiguity is resolved or recorded here.

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| Downstream execution tiers (`AD-001`) | Two-tier pipeline: Tier 1 frozen ResNet-18 embeddings + MLP head; Tier 2 end-to-end ResNet-18 | Enables 60k MCTS steps in minutes for 5-seed ablations while verifying full CNN training at the end | y |
| Hierarchy structures (`AD-002`) | Implement and compare both Native Semantic Tree (20 superclasses) and Ward Confusion Tree | Tests whether empirical model confusion outperforms human semantic taxonomy for MCTS search | y |
| Package manager & language (`AD-003`) | `uv` with `pyproject.toml`; 100% English across code, docs, and paper | Matches international scientific repository standards and deterministic dependency resolution | y |
| Classification regimes (`AD-004`) | Both Scenario 1 (One-vs-Rest imbalanced binary) and Scenario 2 (100-way multi-class curriculum) | Covers strict GiantHunter fidelity (`BCEWithLogitsLoss`) and multi-class curriculum generalization | y |
| MCTS equation variants (`AD-005`) | Controlled ablation between Strict GiantHunter (raw Q, 100% MCTS batch) and Enhanced MCTS-HNM (Min-Max Q, 50/50 Dual-Stream) | Isolates the empirical contribution of Q-normalization and uniform anchor stabilization | y |
| Dynamic leaf sub-clustering (`AD-006`) | Scoped as P2 extension feature (`F5`); P1 MVP keeps leaves at the 100 CIFAR-100 fine classes | Secures a verified class-level MCTS baseline before adding instance-level entropy splitting | y |
| Roadmap & paper tracking (`AD-007`) | Track sub-features `F1`–`F6` in `.specs/features/` + root `ROADMAP.md`, including `F6` for LaTeX paper compilation | Ensures end-to-end traceability from code and unit tests to the final academic manuscript | y |
| Remote JupyterHub & Notebook Orchestration (`AD-008`) | Modular `.py` library in `src/` invoked by versioned `.ipynb` notebooks in `notebooks/` covering the full end-to-end flow | Enables seamless execution on remote JupyterHub without sacrificing modularity or testability | y |
| Rare target class selection in Scenario 1 | 5 fixed target classes sampled from 5 distinct CIFAR-100 superclasses (`beaver`, `oak_tree`, `pickup_truck`, `rocket`, `shark`) | Agent default providing diverse visual domains and reproducible cross-target aggregation | y |

**Open questions:** none - all resolved and recorded above.

---

## User Stories

### P1: Epic Decomposition & Foundation (`F1: Project, Data Pipeline & Setup Notebook`) ⭐ MVP

**User Story**: As a Reinforcement Learning researcher running on a remote JupyterHub, I want a modular CIFAR-100 data/embedding package (`src/data/`) and a versioned notebook (`notebooks/01_data_and_embeddings.ipynb`) so that CIFAR-100 download, frozen ResNet-18 feature extraction, and imbalanced splits run deterministically with a single "Run All".

**Why P1**: All tree builders, MCTS agents, and downstream classifiers depend on standardized dataset splits and cached feature tensors.

**Acceptance Criteria**:

1. WHEN `src.data.extract_embeddings.extract_cifar100_embeddings()` is called from Python or `notebooks/01_data_and_embeddings.ipynb` THEN the system SHALL automatically download CIFAR-100 (if absent) and export deterministic train, validation, and test feature tensors (`shape: [N, 512]`), fine labels (`0..99`), and coarse superclass labels (`0..19`) to disk.
2. WHEN Scenario 1 (One-vs-Rest) is configured for a target class and imbalance ratio $\rho \in \{1:10, 1:50, 1:100\}$ THEN the system SHALL subsample the positive training instances deterministically according to the active random seed while preserving all 99 negative classes in the search pool.
3. The system SHALL enforce deterministic seeding across Python `random`, `numpy`, and `torch` in every module and notebook execution.

**Independent Test**: Run `pytest tests/test_data_pipeline.py` to verify tensor shapes, coarse-to-fine label integrity, and bit-identical splits across two runs with `seed=42`.

---

### P1: Dual Hierarchy Construction Engine (`F2: Semantic & Confusion Trees`) ⭐ MVP

**User Story**: As a researcher testing tree geometry, I want both a static CIFAR-100 Semantic Tree and a data-driven Ward Confusion Tree in `src/trees/`, accompanied by an interactive inspection notebook (`notebooks/02_tree_construction.ipynb`), so that I can build, visualize, and serialize both hierarchies on JupyterHub.

**Why P1**: Comparing semantic taxonomy against empirical model confusion is the central scientific hypothesis of the paper (`AD-002`).

**Acceptance Criteria**:

1. WHEN `build_semantic_tree()` is executed THEN the system SHALL construct a 3-level tree (Virtual Root $\rightarrow$ 20 Superclasses $\rightarrow$ 100 Fine Classes, or 99 negative classes in Scenario 1) backed by contiguous NumPy arrays (`parents`, `children`, `is_leaf`, `instance_counts`).
2. WHEN a baseline validation confusion matrix $C \in \mathbb{R}^{K \times K}$ is passed to `build_confusion_tree()` THEN the system SHALL compute the symmetric distance matrix $D_{i,j} = 1 - \frac{C_{i,j} + C_{j,i}}{2 \cdot \max(C)}$ and build a binary dendrogram using Ward's minimum variance linkage (`scipy.cluster.hierarchy.linkage(method="ward")`).
3. IF a confusion matrix contains zero-sum rows or NaN entries THEN the system SHALL raise a `ValueError` with a descriptive diagnostic message before tree construction.

**Independent Test**: Run `pytest tests/test_trees.py` to verify that every fine class is reachable from the virtual root and that parent instance counts equal the sum of their children's instance counts.

---

### P1: Core MCTS Hard Negative Mining Agent (`F3: Strict & Enhanced MCTS Engine`) ⭐ MVP

**User Story**: As a Reinforcement Learning researcher, I want a configurable MCTS agent (`src/mcts/`) supporting both Strict GiantHunter equations and Enhanced MCTS-HNM (Min-Max Q-normalization + Dual-Stream batching) so that I can quantify the effect of each RL component.

**Why P1**: Implements the core sequential decision-making algorithm (`AD-005`) missing from the official GiantHunter code repository.

**Acceptance Criteria**:

1. WHEN the MCTS agent performs the Selection phase at an internal node $s$ with `q_norm=False` (Strict GiantHunter mode) THEN the system SHALL select the child $i$ maximizing $U_i = Q_i + c \sqrt{\frac{\ln N_s}{n_i + \epsilon}}$.
2. WHERE `q_norm=True` (Enhanced MCTS-HNM mode) is enabled, the system SHALL normalize sibling $Q$-values into $[0, 1]$ via $\bar{Q}_i^{\text{norm}} = \frac{Q_i - Q_{\min}}{Q_{\max} - Q_{\min} + \epsilon}$ before adding the UCB1 exploration bonus.
3. WHEN the MCTS agent receives a batch loss reward $r$ for leaf node $l$ during Backpropagation THEN the system SHALL update the leaf value via momentum $Q_l^{(t)} = \alpha r + (1 - \alpha) Q_l^{(t-1)}$ and update every ancestor $p$ via instance-weighted aggregation $Q_p = \frac{1}{M_p} \sum_{c \in \text{Children}(p)} M_c Q_c$.
4. WHILE training in Dual-Stream mode with ratio $\beta \in [0, 1]$, the system SHALL assemble each negative mini-batch of size $B_{\text{neg}}$ from $\lfloor (1 - \beta) B_{\text{neg}} \rfloor$ MCTS-selected leaf instances and $\lceil \beta B_{\text{neg}} \rceil$ uniformly sampled negative instances.

**Independent Test**: Run `pytest tests/test_mcts_agent.py` to verify UCB1 unvisited-node priority, exact momentum updates ($\alpha=0.3$), weighted parent propagation, and tree visitation entropy computation.

---

### P1: Downstream Classifiers, Baselines & Tier-1 Benchmark Notebook (`F4: Training & Benchmark Suite`) ⭐ MVP

**User Story**: As an experimenter on a remote JupyterHub, I want a unified training API (`src/training/`) and versioned benchmark notebooks (`notebooks/03_tier1_benchmarks.ipynb` and `notebooks/00_master_end_to_end.ipynb`) supporting Uniform Sampling, OHEM, Strict GiantHunter, and Enhanced MCTS-HNM across Scenario 1 and Scenario 2 so that all multi-seed experiments can be triggered and monitored interactively.

**Why P1**: Produces the primary quantitative results, convergence curves, and statistical tests required by the course rubric (`AD-001`, `AD-004`, `AD-008`).

**Acceptance Criteria**:

1. WHEN a benchmark experiment is launched in Scenario 1 (One-vs-Rest) via `src.training.run_benchmark()` from `notebooks/03_tier1_benchmarks.ipynb` THEN the system SHALL train the downstream classifier with `torch.nn.BCEWithLogitsLoss()` and persist test F1-Score, Precision, Recall, AUPRC, FPR@95, and Tree Visitation Entropy at configurable step intervals.
2. WHEN a benchmark experiment is launched in Scenario 2 (100-way Multi-class) THEN the system SHALL train the downstream classifier with `torch.nn.CrossEntropyLoss(label_smoothing=0.1)` and persist Top-1 Accuracy, Macro-F1, Worst-10-Class Recall, and Tree Visitation Entropy.
3. WHEN multi-seed evaluation (`seeds=5`) completes THEN the system SHALL compute mean $\pm$ standard deviation and paired Student's $t$-test / Wilcoxon signed-rank $p$-values comparing Enhanced MCTS-HNM against Uniform, OHEM, and Strict GiantHunter baselines.

**Independent Test**: Run `pytest tests/test_training_loop.py` on a synthetic 20-class hierarchical fixture to confirm that MCTS shifts visit counts toward high-loss classes and exports valid JSON/CSV metric logs.

---

### P2: Dynamic Leaf Expansion & Tier-2 End-to-End Vision Notebook (`F5: Extensions & E2E Validation`)

**User Story**: As a researcher investigating deeper hierarchies and representation learning, I want dynamic entropy-based leaf sub-clustering in `src/mcts/` and a Tier-2 End-to-End ResNet-18 training notebook (`notebooks/04_tier2_e2e_and_expansion.ipynb`) so that I can evaluate MCTS beyond static class leaves and frozen embeddings on the remote JupyterHub GPU.

**Why P2**: Extends the P1 MVP with dynamic sub-cluster expansion (`AD-006`) and Tier-2 full-backbone verification (`AD-001`, `AD-008`).

**Acceptance Criteria**:

1. WHERE dynamic leaf expansion (`enable_leaf_expansion=True`) is enabled, WHEN a class leaf node's visit count exceeds threshold $N_{\text{split}}$ THEN the system SHALL partition that class's instances into $K_{\text{sub}}$ child sub-clusters sorted by model prediction entropy $H(p)$ and attach them as new leaf nodes.
2. WHEN `run_e2e_experiment()` is invoked from `notebooks/04_tier2_e2e_and_expansion.ipynb` THEN the system SHALL fine-tune a ResNet-18 backbone on raw $32 \times 32$ CIFAR-100 images using the winning MCTS-HNM configuration and compare final test metrics against Uniform and OHEM baselines.

**Independent Test**: Run `pytest tests/test_dynamic_expansion.py` to verify that a leaf node splits into $K_{\text{sub}}$ children when $n_l \ge N_{\text{split}}$ and preserves total instance count.

---

### P2: Scientific Paper & Automated Artifact Notebook (`F6: LaTeX Paper & Plots`)

**User Story**: As a master's student submitting the final course deliverable, I want an artifact generator module (`src/utils/paper_artifacts.py`) triggered from `notebooks/05_paper_artifacts.ipynb` (and `notebooks/00_master_end_to_end.ipynb`) that renders publication-grade PDF figures and LaTeX tables directly into the arXiv/Prime AI paper template so that the manuscript is 100% reproducible.

**Why P2**: Fulfills the mandatory final deliverable of the course (scientific paper with Abstract, Introduction, Related Work, Methodology, Results, Discussion, Conclusion).

**Acceptance Criteria**:

1. WHEN `generate_all_paper_artifacts()` is executed from `notebooks/05_paper_artifacts.ipynb` on the `results/` directory THEN the system SHALL render inline notebook previews AND export vector PDF plots (convergence curves, tree visitation heatmaps, hyperparameter sensitivity for $c$ and $\alpha$) and formatted `.tex` booktabs tables under `paper/figures/` and `paper/tables/`.
2. The system SHALL maintain a complete scientific manuscript in `paper/main.tex` structured according to the arXiv and Prime AI template (`Abstract`, `Introduction`, `Related Work`, `Methodology`, `Experimental Setup`, `Results`, `Discussion & Limitations`, `Conclusion`) that compiles cleanly to `paper/main.pdf`.

**Independent Test**: Run `pytest tests/test_paper_artifacts.py` and compile `paper/main.tex` via `pdflatex` / `tectonic` with zero missing references or undefined citations.

---

## Edge Cases

- IF a leaf node contains fewer instances than the requested MCTS sub-batch size $K_{\text{neg}}$ THEN the system SHALL sample with replacement and log a warning once per run.
- IF all sibling nodes at an internal tree level have identical $Q$-values ($Q_{\max} = Q_{\min}$) THEN the system SHALL set $\bar{Q}_i^{\text{norm}} = 0.0$ so selection relies purely on the UCB1 exploration term without division-by-zero (`NaN`).
- WHEN a node has never been visited ($n_i = 0$) THEN the system SHALL prioritize it deterministically ($\epsilon = 10^{-5}$ in the denominator) before exploiting already-visited siblings.
- IF a notebook is re-executed on JupyterHub when CIFAR-100 raw files or extracted embeddings already exist on disk (`overwrite=False`) THEN the system SHALL verify checksums/shapes and reuse the cached tensors without re-downloading.

---

## Requirement Traceability

Each requirement maps to a unique ID across the Epic Roadmap and its child feature specifications (`F1`–`F6`).

| Requirement ID | Story | Phase | Status |
| -------------- | ----- | ----- | ------ |
| EPIC-01 | P1: F1 Project, Data Pipeline & Setup Notebook (`01_data_and_embeddings.ipynb`) | Specify | Pending |
| EPIC-02 | P1: F2 Dual Hierarchy Construction (`02_tree_construction.ipynb`) | Specify | Pending |
| EPIC-03 | P1: F3 Core MCTS-HNM Agent (Strict vs. Enhanced UCB1, Momentum & Dual-Stream) | Specify | Pending |
| EPIC-04 | P1: F4 Classifiers, Baselines & Tier-1 Benchmark Notebook (`03_tier1_benchmarks.ipynb`) | Specify | Pending |
| EPIC-05 | P2: F5 Dynamic Entropy Expansion & Tier-2 E2E Notebook (`04_tier2_e2e_and_expansion.ipynb`) | Specify | Pending |
| EPIC-06 | P2: F6 Automated Paper Artifacts Notebook (`05_paper_artifacts.ipynb`) & LaTeX Manuscript | Specify | Pending |

**Coverage:** 6 total epic requirements, mapped 1-to-1 to child features `F1`–`F6`.

---

## Differences from the source

Every departure from the original *GiantHunter* paper (`btaf239.pdf`) and official repository (`https://github.com/FuchuanQu/GiantHunter`) is documented and linked to an active architectural decision in `.specs/STATE.md`.

| Source shows | Product shows | Confirmed |
| ------------ | ------------- | --------- |
| Official `FuchuanQu/GiantHunter` repo provides only inference code (`gianthunter.py`) and omits the MCTS training loop | Full open-source implementation of the 4-phase MCTS training loop, tree builders, and ablation suite | y - AD-001 |
| GiantHunter operates on metagenomic Protein-Cluster token sequences processed by a 1-layer Transformer + Prodigal/DIAMOND | Operates on CIFAR-100 visual benchmark using Tier-1 frozen ResNet-18 embeddings + MLP head and Tier-2 end-to-end ResNet-18 | y - AD-001 |
| GiantHunter uses a single static NCBI biological taxonomy tree (Family $\rightarrow$ Genus $\rightarrow$ Species) | Implements and compares both static CIFAR-100 Semantic Hierarchy and empirical Ward Confusion-Driven Hierarchy | y - AD-002 |
| GiantHunter evaluates only binary classification (`BCEWithLogitsLoss`, 1 positive phylum vs. negative phages) | Evaluates both Scenario 1 (Imbalanced Binary One-vs-Rest with `BCEWithLogitsLoss`) and Scenario 2 (100-way Multi-class Curriculum with Cross-Entropy) | y - AD-004 |
| GiantHunter Eq. (1) uses raw unnormalized $Q_i$ in UCB1 and samples 100% of negative batches from the selected leaf | Benchmarks both Strict GiantHunter (raw $Q$, 100% MCTS batch) and Enhanced MCTS-HNM (Min-Max normalized $\bar{Q}^{\text{norm}} \in [0, 1]$ + 50/50 Dual-Stream uniform anchor batching) | y - AD-005 |
| GiantHunter tree stops permanently at species-level leaves | P1 stops at class leaves; P2 (`F5`) adds dynamic entropy-based sub-clustering inside high-visit leaves | y - AD-006 |
| GiantHunter executes via a monolithic CLI script (`gianthunter.py`) calling shell `subprocess.run` | Separates modular, unit-tested Python packages (`src/`) from versioned Jupyter notebooks (`notebooks/*.ipynb`) for end-to-end execution on remote JupyterHub | y - AD-008 |

## Sources

- `../btaf239.pdf` (*Qu et al., Bioinformatics 2025, 41, i30–i39*) - binding reference for MCTS equations (Eq. 1–4: UCB1 selection, cross-entropy reward, leaf momentum $\alpha=0.3$, weighted parent backpropagation).
- `https://github.com/FuchuanQu/GiantHunter` - binding code reference for optimizer/loss conventions (`Adam(lr=0.001)`, `BCEWithLogitsLoss`, `batch_size=200` in `src/gianthunter/gianthunter.py`).
- `../GiantHunter_Material_Apoio_Seminario.pdf` - non-binding - user: `"pesquise por opções para atender aos requisitos do enunciado abaixo de preferencia seguindo o paper giant hunter se possivel"` - seminar mapping to Sutton & Barto (Chapters 2, 5, 6, 8).

---

## Success Criteria

- [ ] All 6 child features (`F1`–`F6`) pass their unit/integration test suites (`pytest`) and `tlc-spec-driven` verification gates.
- [ ] Every experimental stage — from CIFAR-100 download to paper artifact generation — executes end-to-end from versioned `.ipynb` notebooks in `notebooks/` on a remote JupyterHub (`AD-008`).
- [ ] Tier-1 multi-seed benchmark ($N = 5$ seeds) completes in $< 30$ minutes on a single machine and produces statistically valid comparisons ($p$-values logged).
- [ ] Enhanced MCTS-HNM demonstrates measurable improvement in AUPRC / F1-Score or sample convergence speed over Uniform Sampling, with full ablation tables comparing Semantic vs. Confusion trees.
- [ ] Final scientific paper (`paper/main.pdf`) compiles cleanly in the arXiv/Prime AI template with all sections, figures, tables, and limitation discussions complete.
