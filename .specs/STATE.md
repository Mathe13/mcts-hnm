# STATE

## Decisions

### AD-001
- **Decision**: Execute experiments in two tiers: Tier 1 on frozen backbone embeddings with an MLP head for rapid MCTS ablations and multi-seed validation, and Tier 2 end-to-end CNN training for final verification.
- **Reason**: User selected "Ambos: prototipar e validar a busca com Embeddings e rodar um experimento final End-to-End" when asked about the downstream training architecture.
- **Trade-off**: Maintaining two training pipelines (cached tensor loader vs. raw image dataloader) adds code surface in exchange for orders-of-magnitude faster iteration on RL hyperparameters.
- **Scope**: `src/models/`, `src/training/`, `experiments/`
- **Date**: 2026-11-08
- **Status**: active (AD-001)

### AD-002
- **Decision**: Implement and benchmark both the native CIFAR-100 Semantic Tree (20 superclasses to 100 classes) and the Data-Driven Confusion Tree (Ward hierarchical clustering on validation confusion matrices).
- **Reason**: User selected "Comparativa: implementar tanto a Árvore Semântica Nativa (Superclasses) quanto a Árvore Orientada por Confusão (Ward Clustering) para servir como ablação central do artigo".
- **Trade-off**: Requires a baseline warm-up pass to compute the confusion matrix before constructing the data-driven tree.
- **Scope**: `src/trees/`, `experiments/`
- **Date**: 2026-11-08
- **Status**: active (AD-002)

### AD-003
- **Decision**: Standardize repository documentation, code, and paper artifacts in English, managed via `uv` and `pyproject.toml`.
- **Reason**: User selected `uv / venv padrão` and `100% em Inglês (padrão de repositórios científicos internacionais)`.
- **Trade-off**: None.
- **Scope**: Entire repository (`mcts-hnm/`)
- **Date**: 2026-11-08
- **Status**: active (AD-003)

### AD-004
- **Decision**: Evaluate MCTS-HNM under two classification scenarios: Scenario 1 (Imbalanced Binary One-vs-Rest with `BCEWithLogitsLoss`, faithful to GiantHunter) and Scenario 2 (100-way Multi-class Active Batch Curriculum with Cross-Entropy and label smoothing).
- **Reason**: Question asked "Qual formulação de classificação o Roadmap deve adotar no CIFAR-100?"; user chose "Ambos como cenários do artigo: Cenário 1 (One-vs-Rest fiel ao GiantHunter) e Cenário 2 (Currículo Multi-classe 100 vias)".
- **Trade-off**: Doubles the experimental evaluation matrix in exchange for testing both rare-target detection and full multi-class curriculum generalization.
- **Scope**: `src/models/`, `src/training/`, `experiments/`
- **Date**: 2026-11-08
- **Status**: active (AD-004)

### AD-005
- **Decision**: Implement both Strict GiantHunter MCTS (raw unnormalized Q-values and 100% MCTS negative batches) and Enhanced MCTS-HNM (Min-Max Q-normalization in `[0, 1]` and 50/50 Dual-Stream uniform+MCTS batches) as a controlled ablation study.
- **Reason**: Question asked "Sobre as duas melhorias que desenhamos em relação às equações originais do GiantHunter (Normalização Min-Max de Q e Lote Híbrido Dual-Stream 50/50), como devemos posicioná-las no Roadmap?"; user chose "(Recommended) Ablação Científica: implementar tanto o GiantHunter Estrito (Q bruto, lote 100% MCTS) quanto o MCTS-HNM Aprimorado (Q normalizado Min-Max + Dual-Stream Batch) para demonstrar o ganho no artigo".
- **Trade-off**: Requires configurable strategy flags in the MCTS selector and batch sampler.
- **Scope**: `src/mcts/`, `src/training/`, `experiments/`
- **Date**: 2026-11-08
- **Status**: active (AD-005)

### AD-006
- **Decision**: Scope leaf-level dynamic sub-clustering by prediction entropy as a P2 extension feature, keeping P1 MVP leaves fixed at the 100 fine classes of CIFAR-100.
- **Reason**: Question asked "Sobre a Expansão Dinâmica de nós-folha (dividir uma classe fina em sub-clusters de imagens difíceis pela entropia softmax), qual prioridade ela deve ter no Épico?"; user chose "(Recommended) Feature P2 (Extensão/Ablação): fechar o MVP (P1) com folhas no nível das classes finas (100 folhas) e implementar a sub-clusterização dinâmica de imagens por entropia como Feature P2".
- **Trade-off**: Defers instance-level sub-tree expansion until after class-level MCTS is verified.
- **Scope**: `src/trees/`, `src/mcts/`
- **Date**: 2026-11-08
- **Status**: active (AD-006)

### AD-007
- **Decision**: Track the project roadmap and feature breakdown via `.specs/features/` (F1 through F6) plus a root `ROADMAP.md`, including a dedicated feature (F6) for automated figure/table generation and LaTeX paper compilation using the arXiv/Prime AI template.
- **Reason**: Question asked "Como você prefere organizar a quebra do Épico em Features/Issues e o entregável final do artigo científico?"; user chose "(Recommended) Rastreamento completo via .specs/features/ (F1 a F6) + ROADMAP.md na raiz, incluindo uma Feature final dedicada à geração automática dos gráficos/tabelas e estruturação do artigo LaTeX (template arXiv/Prime AI)".
- **Trade-off**: None.
- **Scope**: `.specs/features/`, `ROADMAP.md`, `paper/`
- **Date**: 2026-11-08
- **Status**: active (AD-007)

### AD-008
- **Decision**: Orchestrate all end-to-end experimental workflows (CIFAR-100 download, embedding extraction, tree construction, MCTS benchmarks, and paper artifact generation) through versioned Jupyter notebooks (`notebooks/*.ipynb`) calling modular Python library code in `src/` for remote JupyterHub execution.
- **Reason**: User specified: "irei rodar os exeprimentos em jupyter hub remoto, netão seria interessante que todo o fluxo de experiemtnos pudesse ser disparado a partir de notebooks verisonados neste projeto. codigo complexo fica em arquivos .py seguindo boas praticas de modularidade e programação mas a execução deve ser disparada de notebooks que faze end-toend desde o download do cifar até a geração dos artefatos".
- **Trade-off**: Requires clean programmatic Python APIs in `src/` (avoiding CLI-only entrypoints) and keeping `.ipynb` cells lightweight and version-control friendly.
- **Scope**: `notebooks/`, `src/`, `pyproject.toml`
- **Date**: 2026-11-08
- **Status**: active (AD-008)

### AD-009
- **Decision**: Configure paper manuscript metadata (title, authors list, affiliation, contact email, and abstract) via a root YAML file (`paper_metadata.yaml`) loaded dynamically during paper compilation and artifact generation, targeting PPGC UFRGS.
- **Reason**: User specified: "múltiplos autores, informados posteriormente; PPGC UFRGS; MCTS-HNM: Monte Carlo Tree Search for Hierarchical Hard Negative Mining in Class-Imbalanced Learning; idealmente estes dados devem vir de um yaml na raiz do projeto".
- **Trade-off**: Requires PyYAML (or lightweight stdlib YAML parser) to dynamically inject authors and metadata into the LaTeX manuscript template and paper artifact headers.
- **Scope**: `paper_metadata.yaml`, `src/utils/paper_artifacts.py`, `paper/`
- **Date**: 2026-11-09
- **Status**: active (AD-009)

### AD-010
- **Decision**: Implement a centralized `ExperimentConfig` and `ExecutionMode` (`SMOKE` vs `REAL`) enforcing strict isolation across directory trees (`embeddings/{mode}`, `results/{mode}`, `checkpoints/{mode}`, `results/legacy`), explicit ASCII warning banners, and `.meta.json` provenance sidecars, prohibiting synthetic fallbacks in REAL mode.
- **Reason**: ROADMAP_Improvements.md mandates strict reproducibility, elimination of silent fallbacks to random data, and full auditability of experimental scientific artifacts.
- **Trade-off**: Requires explicit configuration in all entrypoints and rejecting unannotated or synthetic artifacts when in REAL mode.
- **Scope**: `src/utils/experiment_config.py`, `notebooks/`, `tests/`
- **Date**: 2026-11-10
- **Status**: active (AD-010)

### AD-011
- **Decision**: Standardize notebooks 00, 01, and 02 under the `ExecutionMode` contract with prominent banners, eliminate silent fallbacks to `torch.randn` in REAL mode via `extract_or_generate_embeddings`, isolate trees into `embeddings/{mode}/trees/`, and parameterize `build_confusion_tree` with configurable linkage (`linkage_method="ward"` default vs `"average"`).
- **Reason**: Preserves exact historical Ward clustering results while addressing the methodological review of confusion distances and ensuring clean isolation between test fixtures and scientific artifacts.
- **Trade-off**: In REAL mode, execution halts with `FileNotFoundError` if CIFAR-100 or real embeddings are absent instead of continuing on random prototypes.
- **Scope**: `notebooks/00_master_end_to_end.ipynb`, `notebooks/01_data_and_embeddings.ipynb`, `notebooks/02_tree_construction.ipynb`, `src/data/`, `src/trees/`
- **Date**: 2026-11-10
- **Status**: active (AD-011)

### AD-012
- **Decision**: Standardize Notebook 03 under `ExecutionMode` contract (rejection of missing embeddings in REAL mode without fallback), enforce strict negative batch budget equivalence ($K_{\text{neg}} = 48$) across Uniform, OHEM, Strict GiantHunter, and Enhanced MCTS, instrument Scenario 1 imbalance audit (available vs effective batch ratios), track Scenario 2 MCTS multiclass diagnostics (visitation entropy, unvisited class counts, per-class recall, and coverage percentage), implement executable sensitivity ablation for $(c, \alpha)$, and serialize isolated JSON/CSV with `.meta.json` sidecars.
- **Reason**: Fixes critical scientific risks in Tier-1 benchmarks where synthetic prototype embeddings could silently pollute experimental tables, resolves methodological ambiguity in 1:50 imbalance sampling, and provides empirical verification for announced sensitivity claims.
- **Trade-off**: Increases telemetry payload and results directory files, with each artifact now carrying a sidecar `.meta.json`.
- **Scope**: `src/training/benchmark_runner.py`, `src/data/splits.py`, `notebooks/03_tier1_mcts_benchmarks.ipynb`, `tests/test_tier1_benchmarks_real.py`
- **Date**: 2026-11-10
- **Status**: active (AD-012)

### AD-013
- **Decision**: Standardize Notebook 04 under `ExecutionMode` contract (rejecting missing CIFAR-100 datasets in REAL mode without fallback), implement `load_cifar100_raw_images` in `src/data/cifar100_images.py`, persist ResNet-18 model weights to `checkpoints/{mode}/`, export E2E metrics to `results/{mode}/` with `.meta.json` sidecars, and audit dynamic entropy leaf expansion and tree structural invariants.
- **Reason**: Eliminates silent synthetic random image tensor fallbacks in Tier-2 vision training, ensuring scientific legitimacy for end-to-end benchmarks and dynamic tree expansion.
- **Trade-off**: Requires local or downloaded CIFAR-100 dataset when executing in REAL mode.
- **Scope**: `src/data/cifar100_images.py`, `src/training/e2e_trainer.py`, `notebooks/04_tier2_e2e_and_expansion.ipynb`, `tests/test_tier2_e2e_real.py`
- **Date**: 2026-11-10
- **Status**: active (AD-013)

### AD-014
- **Decision**: Standardize Notebook 05 and `src/utils/paper_artifacts.py` under the `ExecutionMode` contract. In `ExecutionMode.REAL`, `load_or_synthesize_benchmark_results` strictly ingests real benchmark JSON files (`tier1_benchmark_summary.json` or component outputs `tier1_axis1_samplers.json`, `tier1_scenario2_curriculum.json`, `tier1_axis2_tree_ablation.json`) and raises `FileNotFoundError` if absent, entirely prohibiting synthetic distribution fallbacks. In `ExecutionMode.SMOKE`, figures are visibly watermarked with `SMOKE TEST — NOT FOR PUBLICATION` and saved strictly to `results/smoke/paper/`. Export an auditable `artifact_manifest.json` tracking SHA-256 digests of all inputs and generated artifacts, git commit, mode, and UTC timestamp.
- **Reason**: Eliminates silent fabrication of experimental metrics in publication figures and tables, guarantees strict traceability from paper artifacts back to experimental raw JSON outputs, and segregates smoke demonstration figures from final publication PDFs/PNGs.
- **Trade-off**: Requires running experimental benchmarks before paper artifacts can be generated in REAL mode; attempting generation without results produces an immediate fail-stop error.
- **Scope**: `src/utils/paper_artifacts.py`, `notebooks/05_paper_artifacts.ipynb`, `tests/test_paper_artifacts_real.py`
- **Date**: 2026-11-10
- **Status**: active (AD-014)

### AD-015
- **Decision**: Codify a unified automated regression testing suite (`tests/test_automated_suite_real.py`) executing all six notebooks (`00` through `05`) end-to-end in `ExecutionMode.SMOKE`, verifying cross-mode cache rejection (REAL rejects SMOKE cache with `ValueError`; SMOKE rejects REAL cache), enforcing session mode transition immutability (`IllegalStateError`), confirming notebook 05 rejects SMOKE results during REAL generation with `FileNotFoundError`, and exercising a labeled reduced real integration path within a strict <60s SLA.
- **Reason**: ROADMAP_Improvements.md Phase 6 establishes full automated end-to-end regression validation, preventing accidental divergence, silent fallbacks, or cross-mode pollution across all pipeline stages.
- **Trade-off**: Adds 12 automated tests (~12s) while keeping total repository test suite runtime (~30s across 142 tests) well under the 60-second budget.
- **Scope**: `tests/test_automated_suite_real.py`, `src/data/extract_embeddings.py`, `notebooks/01_data_and_embeddings.ipynb`
- **Date**: 2026-11-10
- **Status**: active (AD-015)

### AD-016
- **Decision**: Codify a comprehensive experimental reproducibility guide (`docs/experimental_reproducibility.md`) and revise root `README.md` to document execution mode semantics (`SMOKE` vs `REAL`), directory layout with `.meta.json` sidecars, step-by-step reproduction recipes, tree geometry methodological review (Ward vs Average linkage), negative mining budget equivalence ($K_{\text{neg}}=48$), dynamic leaf expansion, known experimental limitations, and auditable classification of completed vs pending vs historical experiments.
- **Reason**: ROADMAP_Improvements.md Phase 7 establishes scientific transparency, complete auditability, and publication-ready experimental reproducibility across the entire project lifecycle.
- **Trade-off**: Requires maintaining documentation synchronized with code changes, verified by automated link-checking and structure tests in `tests/test_documentation_and_reproducibility.py`.
- **Scope**: `README.md`, `docs/experimental_reproducibility.md`, `tests/test_documentation_and_reproducibility.py`
- **Date**: 2026-11-10
- **Status**: active (AD-016)

### AD-017
- **Decision**: Mandate `.meta.json` provenance sidecars unconditionally in both SMOKE and REAL modes, reject cross-mode artifacts with explicit `ValueError`, eliminate silent numeric defaults in LaTeX table and figure generators, enforce complete artifact validation (`validate_artifact_completeness`), expand Tier-2 synthetic fixtures to multiple samples per class, and strictly confine I/O paths to mode-segregated directories (`results/smoke`, `results/real`).
- **Reason**: Fixes P0 and P1 audit vulnerabilities: prevents unverified artifacts from silently being used, eliminates synthetic metric fabrication in paper deliverables, prevents sample-with-replacement warnings during dynamic entropy leaf expansion in smoke mode, and guarantees strict I/O path isolation.
- **Trade-off**: Requires every generated result file to carry a sidecar `.meta.json` even in smoke mode; attempting to access unsegregated `results/` without mode prefix raises `ValueError`.
- **Scope**: `src/utils/experiment_config.py`, `src/utils/paper_artifacts.py`, `src/data/cifar100_images.py`, `tests/`
- **Date**: 2026-11-10
- **Status**: active (AD-017)

### AD-018
- **Decision**: Strictly eliminate zero-defaults in the benchmark results loader (`src/utils/paper_artifacts.py`), mandate suite completeness in REAL mode (requiring complete Scenario 1, Scenario 2, and ablation results without partial fallbacks), re-raise raw parsing exceptions immediately without swallowing as warnings, modernize test contracts from legacy `FAST_MODE` to `ExecutionMode`, switch legacy migration default in `get_isolated_paths` to `migrate_legacy=False` (explicit-only), and enforce mode path isolation on custom output directories in `benchmark_runner.py`.
- **Reason**: Closes audit vulnerabilities: prevents unrecorded missing metrics from silently masquerading as 0.0 in scientific publications, guarantees that incomplete benchmark suites cannot generate final paper artifacts, avoids unexpected disk file moves during path queries, and ensures mode boundary confinement for custom directories.
- **Trade-off**: Requires all benchmark suites to be run prior to REAL paper artifact generation; attempting to generate deliverables from partial runs raises `FileNotFoundError` or `ValueError`.
- **Scope**: `src/utils/paper_artifacts.py`, `src/utils/experiment_config.py`, `src/training/benchmark_runner.py`, `tests/`
- **Date**: 2026-11-10
- **Status**: active (AD-018)

### AD-019
- **Decision**: Parameterize Scenario 1 positive class subsampling in `run_benchmark()` via `imbalance_ratios: tuple[str, ...] = ("1:50",)` (using `build_scenario1_split` and `IMBALANCE_RATIOS`), embed `imbalance_ratio` metadata in `BenchmarkRunResult` and JSON/CSV outputs, normalize ratio extraction in `load_real_benchmark_results` without hardcoded fallbacks (defaulting legacy unlabeled runs to `"50"`), and dynamically adapt `generate_scenario1_table` and table captions to the set of executed ratios without failing or requiring unexecuted ratios.
- **Reason**: Resolves P1 audit finding where Notebook 05 expected 1:10, 1:50, 1:100 while Notebook 03 only executed 1:50. Eliminates data fabrication risks, provides immediate execution support for arbitrary ratio subsets or the full grid (`"1:10"`, `"1:50"`, `"1:100"`), and ensures publication tables dynamically and explicitly adapt to available empirical data without formatting defects.
- **Trade-off**: Requires passing `imbalance_ratios` if evaluating multi-ratio grids; single-ratio runs generate single-ratio LaTeX tables with adapted captions.
- **Scope**: `src/training/benchmark_runner.py`, `src/utils/paper_artifacts.py`, `notebooks/03_tier1_mcts_benchmarks.ipynb`, `tests/`
- **Date**: 2026-11-10
- **Status**: active (AD-019)

### AD-020
- **Decision**: Standardize experimental protocol under Phase A of the revised experimental roadmap (`roadmap_mcts_hnm_proxima_run_v2.md`): operational imbalance definition separating training set real ratios (1:50, 1:100, 1:500) with fixed 80 unique positives in `train_fit` and nested stratified negatives ($N_{50} \subset N_{100} \subset N_{500}$); strict data segregation prohibiting in-training test evaluation or tree calibration on test data; shared immutable runner configuration (`ResolvedTrainingConfig`); atomic per-`run_id` serialization with strict key pairing in statistical tests; and mathematically validated Euclidean row-normalized confusion profiling for Ward clustering with Average Linkage (UPGMA) for general dissimilarity matrices.
- **Reason**: Fixes P0 methodological audit findings: eliminates the false equivalence between `pos_fraction=0.02` and 1:50; prevents test-set data leakage into training curves and tree construction; eliminates parameter divergences between runners; prevents run overwrites; and replaces heuristic `sqrt(2*distance)` transformations with legitimate Euclidean representations.
- **Trade-off**: Requires passing validation data to training loops and runners (failing fast if omitted), and mandates structured run metadata.
- **Scope**: `src/data/splits.py`, `src/training/`, `src/trees/confusion_tree.py`, `src/utils/`, `docs/`, `tests/`
- **Date**: 2026-11-11
- **Status**: active (AD-020)

## Handoff

- **Feature**: `17-fase-a-auditoria-e-integridade` (`.specs/features/17-fase-a-auditoria-e-integridade/`)
- **Phase / Task**: Tasks Phase completed, awaiting user approval of `tasks.md` to begin execution.
- **Completed**:
  - Specified requirements in `spec.md` (validated with 0 errors, 0 warnings).
  - Designed architecture and components in `design.md`.
  - Structured 9 atomic tasks across 7 phases in `tasks.md` with 16 verifiable checks (validated with 0 errors, 0 warnings).
- **In-progress**: Awaiting user approval of `tasks.md`.
- **Next step**: Execute Task T1 upon user approval.
- **Blockers**: None
- **Branch**: `main`
