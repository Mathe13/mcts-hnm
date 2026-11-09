# 00-epic-roadmap Context

**Gathered:** 2026-11-08
**Spec:** `.specs/features/00-epic-roadmap/spec.md`
**Status:** Ready for design

---

## Feature Boundary

Define the master Epic Roadmap for the **MCTS-HNM** research project (`mcts-hnm`), decomposing the adaptation of GiantHunter's MCTS Hard Negative Mining algorithm to CIFAR-100 into sequential, independently verifiable sub-features (`F1` through `F6`), issues, and atomic tasks, culminating in a reproducible benchmark suite and a compiled scientific paper in the arXiv/Prime AI LaTeX template.

---

## Implementation Decisions

### 1. Classification Paradigms on CIFAR-100 (Scenario 1 & Scenario 2)

- User decision: `"Ambos como cenários do artigo: Cenário 1 (One-vs-Rest fiel ao GiantHunter) e Cenário 2 (Currículo Multi-classe 100 vias)"` (`AD-004`).
- Scenario 1 reproduces the imbalanced binary classification setting of GiantHunter (`BCEWithLogitsLoss`, 1 rare target positive class vs. 99 negative classes organized in a hierarchy), evaluated across multiple target classes sampled from distinct superclasses.
- Scenario 2 extends MCTS batch selection to 100-way multi-class classification with Cross-Entropy loss and label smoothing.

### 2. Algorithmic Fidelity vs. MCTS-HNM Enhancements (Strict vs. Enhanced Ablation)

- User decision: `"(Recommended) Ablação Científica: implementar tanto o GiantHunter Estrito (Q bruto, lote 100% MCTS) quanto o MCTS-HNM Aprimorado (Q normalizado Min-Max + Dual-Stream Batch) para demonstrar o ganho no artigo"` (`AD-005`).
- The codebase will expose configurable toggles for `q_norm: bool` (raw $Q$ vs. Min-Max normalized $\bar{Q}^{\text{norm}} \in [0, 1]$) and `dual_stream_ratio: float` ($0.0$ for 100% MCTS negatives vs. $0.5$ for 50% uniform anchors + 50% MCTS negatives).

### 3. Dynamic Leaf Expansion (Entropy-Based Sub-Clustering)

- User decision: `"(Recommended) Feature P2 (Extensão/Ablação): fechar o MVP (P1) com folhas no nível das classes finas (100 folhas) e implementar a sub-clusterização dinâmica de imagens por entropia como Feature P2"` (`AD-006`).
- P1 MVP uses the 100 CIFAR-100 fine classes as terminal leaf nodes.
- P2 adds dynamic leaf expansion that partitions high-visit leaf classes into sub-clusters based on model prediction entropy.

### 4. Roadmap Organization & Scientific Paper Deliverable

- User decision: `"(Recommended) Rastreamento completo via .specs/features/ (F1 a F6) + ROADMAP.md na raiz, incluindo uma Feature final dedicada à geração automática dos gráficos/tabelas e estruturação do artigo LaTeX (template arXiv/Prime AI)"` (`AD-007`).
- Every sub-feature (`F1` to `F6`) will live under `.specs/features/` and map to explicit issues and tasks in `ROADMAP.md`.

### 5. Remote JupyterHub Execution via Versioned End-to-End Notebooks

- User decision: `"irei rodar os exeprimentos em jupyter hub remoto, netão seria interessante que todo o fluxo de experiemtnos pudesse ser disparado a partir de notebooks verisonados neste projeto. codigo complexo fica em arquivos .py seguindo boas praticas de modularidade e programação mas a execução deve ser disparada de notebooks que faze end-toend desde o download do cifar até a geração dos artefatos"` (`AD-008`).
- All complex algorithmic, data, model, and plotting logic resides in modular Python packages under `src/` with full `pytest` unit test coverage.
- Versioned Jupyter notebooks under `notebooks/` serve as the primary user-facing orchestration layer on the remote JupyterHub, driving the full pipeline end-to-end (CIFAR-100 download $\rightarrow$ embedding extraction $\rightarrow$ tree construction $\rightarrow$ Tier-1 & Tier-2 MCTS benchmarks $\rightarrow$ paper figures and LaTeX tables).

### 6. Prior Foundational Decisions

- Two-tier execution (`AD-001`): `"Ambos: prototipar e validar a busca com Embeddings e rodar um experimento final End-to-End"`.
- Tree architectures (`AD-002`): `"(Recommended) Comparativa: implementar tanto a Árvore Semântica Nativa (Superclasses) quanto a Árvore Orientada por Confusão (Ward Clustering) para servir como ablação central do artigo"`.
- Environment and language (`AD-003`): `"(Recommended) uv / venv padrão (rápido, determinístico e compatível com pyproject.toml)"` and `"100% em Inglês (padrão de repositórios científicos internacionais)"`.

### Agent's Discretion

- Specific CIFAR-100 target classes chosen to represent the rare positive targets in Scenario 1 (e.g., selecting 5 representative classes across distinct superclasses for statistical robustness).
- Exact neural network hidden dimensions for the Tier 1 MLP head (default: `512 -> 128 -> num_outputs`) and Tier 2 ResNet-18 training hyperparameters (`Adam`, `lr=1e-3`, matching `gianthunter.py`).

### Declined / Undiscussed Gray Areas → Assumptions

- None — all four architectural and scope questions were explicitly answered by the user.

---

## Specific References

- **GiantHunter Paper (`btaf239.pdf`)**: Sections 2.1 and 2.2 define the four MCTS phases, UCB1 formula ($U_i = Q_i + c \sqrt{\ln N_i / n_i}$ with $c=2$), leaf momentum update ($Q_i^l = \alpha Q' + (1-\alpha) Q_{i-1}^l$ with $\alpha=0.3$), and parent weighted aggregation ($Q_i^p = \frac{1}{m_p} \sum m_c Q_i^c$).
- **GiantHunter Official Repository (`https://github.com/FuchuanQu/GiantHunter`)**: Inspection confirmed that the repository provides inference-only code (`src/gianthunter/gianthunter.py` and `src/gianthunter/model.py`) using `nn.BCEWithLogitsLoss()` and `optim.Adam(model.parameters(), lr=0.001)` with `batch_size=200`, while omitting the MCTS training loop.
- **Overleaf Paper Template**: [arXiv and Prime AI Style Template](https://www.overleaf.com/latex/templates/arxiv-and-prime-ai-style-template/qdnhqytdqzsc).

---

## Deferred Ideas

- Interactive web application for human-in-the-loop active learning (Modalidade 3) — explicitly out of scope to keep the research focused on the benchmark and algorithmic comparison.
- Genomic bioinformatics tools (Prodigal, DIAMOND, MCL, HMMER) — excluded by transferring the hierarchical MCTS formulation to CIFAR-100.
