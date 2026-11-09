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

## Handoff

- **Feature**: `01-data-and-embeddings` (`.specs/features/01-data-and-embeddings/`)
- **Phase / Task**: Done & Verified (`validate_verification.py --strict` exited 0)
- **Completed**:
  - `T1` (`6b36d03`): `pyproject.toml`, `src/` layout, `.gitignore` root-anchored `/data/` fix
  - `T2` (`0560fee`): `src/utils/seed.py` (`set_global_seed`, `get_torch_generator`)
  - `T3` (`64216ad`): `src/data/cifar100_taxonomy.py` (`CIFAR100Taxonomy`, `fine_to_coarse_tensor`)
  - `T4` (`645c460`): `src/data/extract_embeddings.py` (stratified 45k/5k split, frozen ResNet-18 extractor, atomic `.pt` cache)
  - `T5` (`ec6f986`): `src/data/splits.py` (`Scenario1Split` 1:10/1:50/1:100 & `Scenario2Split` 100-class pools)
  - `T6` (`c1e031a`): `notebooks/01_data_and_embeddings.ipynb` end-to-end JupyterHub orchestrator
  - Independent Verifier (`838a6c52-70ba-4178-b2f0-ba32cfd05045`): `PASS` across all 14 checks (`C1`–`C14`)
- **In-progress**: None
- **Next step**: Specify, Design, Tasks, and Execute for Feature `F2` (`02-dual-hierarchy-trees`)
- **Blockers**: None
- **Uncommitted files**: None (clean working tree)
- **Branch**: `main`
