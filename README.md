# MCTS-HNM: Monte Carlo Tree Search for Hard Negative Mining in Imbalanced Hierarchical Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**MCTS-HNM** explores the application of Reinforcement Learning (RL) — specifically **Monte Carlo Tree Search (MCTS)** with Upper Confidence Bound (UCB) action selection — as an active curriculum learning strategy for **Hard Negative Mining (HNM)** under extreme class imbalance.

This project is developed as the final scientific paper and implementation for the graduate course in **Reinforcement Learning** (Master's Program).

### Scientific Motivation & Paper Reference
Our core formulation is inspired by **GiantHunter** (*Qu et al., Bioinformatics, 2025; ISMB/ECCB 2025*), which formulated the selection of non-target microbial training samples as a sequential decision process over a taxonomic phylogenetic tree. 

While GiantHunter demonstrated superior F1-score and sample efficiency in metagenomic viral discovery, its authors hypothesized that the framework is *generalizable to any large-scale hierarchical classification problem*. **This project tests this generalization hypothesis on open machine learning benchmarks (CIFAR-100)**, addressing critical questions regarding:
1. **Tree Geometry**: Does MCTS perform better when guided by human semantic taxonomy (e.g., CIFAR-100 Superclasses / WordNet) or by empirical data-driven confusion geometry (Ward agglomerative clustering on model error matrices)?
2. **Reward Non-Stationarity**: How does the dynamic shift in classifier decision boundaries affect UCB exploration and value updates during sequential search?
3. **Sample Efficiency**: Can hierarchical MCTS mining achieve competitive decision boundaries with significantly fewer training exposures than uniform sampling or online hard example mining (OHEM)?

---

## Key Features

- **Dual Tree Architectures**:
  - **Semantic Hierarchy**: Natural 2-level CIFAR-100 taxonomy ($20 \text{ Superclasses} \rightarrow 100 \text{ Fine Classes}$).
  - **Confusion-Driven Hierarchy**: Dynamically constructed dendrograms via Ward hierarchical clustering over empirical confusion matrices.
- **Hierarchical MCTS Agent**:
  - Min-Max normalized UCB1 selection with tunable exploration coefficient $c$.
  - Dual-stream mini-batch generation (uniform stabilization anchors + MCTS-mined hard negatives).
  - Leaf momentum ($\alpha = 0.3$) and sample-weighted tree backpropagation.
- **Two-Tier Execution Pipeline**:
  - **Fast Embedding Prototype**: Pre-extracted feature representations (e.g., frozen ResNet-18 / ViT embeddings) paired with lightweight classification heads for rapid algorithmic search, ablation studies, and multiple random seeds ($N \ge 5$).
  - **End-to-End CNN Pipeline**: Complete vision backbone fine-tuning validating full gradient dynamics.
- **Spec-Driven Engineering**:
  - Implemented following the **TLC Spec-Driven v4** development lifecycle (`.agents/skills/tlc-spec-driven/`).

---

## Project Structure

```text
.
├── .agents/                    # Agent customizations & TLC Spec-Driven skill
│   └── skills/tlc-spec-driven/ # Spec-Driven Development engine
├── .specs/                     # TLC Spec-Driven state, decisions (AD-NNN), and feature specs
├── notebooks/                  # Versioned Jupyter notebooks for remote JupyterHub execution
│   ├── 00_master_end_to_end.ipynb
│   ├── 01_data_and_embeddings.ipynb
│   ├── 02_tree_construction.ipynb
│   ├── 03_tier1_mcts_benchmarks.ipynb
│   ├── 04_tier2_e2e_and_expansion.ipynb
│   └── 05_paper_artifacts.ipynb
├── src/                        # Modular core Python library (all complex logic lives here)
│   ├── data/                   # CIFAR-100 downloader, taxonomy map, and embedding extractor
│   ├── trees/                  # Semantic and Ward confusion-based tree builders
│   ├── mcts/                   # MCTS agent, UCB selection, dynamic expansion, and backpropagation
│   ├── models/                 # Classifier heads and ResNet-18 backbones
│   ├── training/               # Dual-stream trainer and baseline runners (Uniform, OHEM)
│   └── utils/                  # Seeding, statistical tests, and paper figure/table generators
├── paper/                      # Scientific paper LaTeX draft (arXiv / Prime AI template)
├── tests/                      # Unit and integration test suite (pytest)
├── pyproject.toml              # Project dependencies and packaging
├── ROADMAP.md                  # Master Epic breakdown (Features F1-F6, Issues, and Tasks)
├── AGENTS.md                   # Development rules for AI agents
├── ARCHITECTURE.md             # System architecture specification
└── README.md                   # This file
```

---

## Execution Modes & Directory Layout

To eliminate silent synthetic fallbacks and enforce scientific reproducibility, the repository implements a strict two-tier execution contract managed by `ExecutionMode`:

- **`ExecutionMode.SMOKE`**: Uses deterministic synthetic data and small fixtures for ultra-fast pipeline verification, test runs, and continuous integration. Outputs are strictly confined to `results/smoke/`, `embeddings/smoke/`, and `checkpoints/smoke/`. Smoke figures carry visible `[SMOKE TEST — NOT FOR PUBLICATION]` watermarks.
- **`ExecutionMode.REAL`**: Executes scientific experiments on genuine **CIFAR-100** data. If raw datasets or caches are missing, execution halts with an explicit `FileNotFoundError` rather than silently generating random prototypes. Outputs are saved to `results/real/`, `embeddings/real/`, `checkpoints/real/`, and `paper/`.

All exported result files (`.json`, `.csv`, `.pt`) are accompanied by a `.meta.json` provenance sidecar recording dataset fingerprints, execution mode, git commit SHA, and UTC timestamps.

### Directory Layout

```text
.
├── checkpoints/
│   ├── real/                   # Checkpoints from genuine CIFAR-100 training
│   └── smoke/                  # Checkpoints from smoke test passes
├── embeddings/
│   ├── real/                   # Pre-extracted frozen ResNet-18 embeddings (cifar100_resnet18.pt)
│   ├── smoke/                  # Fast deterministic synthetic embedding bundles
│   └── trees/                  # Serialized semantic and confusion hierarchies (.npz)
├── results/
│   ├── legacy/                 # Historical results pre-dating the execution contract
│   ├── real/                   # Production scientific benchmark metrics and sidecars
│   └── smoke/                  # Smoke test run metrics and demonstrative artifacts
├── paper/
│   ├── figures/                # Publication-grade vector PDF and 300 DPI PNG figures
│   ├── tables/                 # LaTeX booktabs tables (.tex)
│   ├── artifact_manifest.json  # Cryptographic SHA-256 provenance manifest
│   ├── main.tex                # Paper manuscript draft
│   └── main.pdf                # Compiled publication PDF
├── notebooks/                  # Versioned Jupyter notebooks for remote JupyterHub execution
│   ├── 00_master_end_to_end.ipynb
│   ├── 01_data_and_embeddings.ipynb
│   ├── 02_tree_construction.ipynb
│   ├── 03_tier1_mcts_benchmarks.ipynb
│   ├── 04_tier2_e2e_and_expansion.ipynb
│   └── 05_paper_artifacts.ipynb
├── docs/                       # Detailed scientific and reproducibility documentation
│   └── experimental_reproducibility.md
├── src/                        # Modular core Python library
├── tests/                      # Automated unit, integration, and regression test suite
└── pyproject.toml              # Build metadata and dependencies
```

---

## Dataset Requirements

For scientific runs in `ExecutionMode.REAL`:
- **CIFAR-100**: Downloaded automatically by torchvision into `data/` on first execution, or pre-cached into `embeddings/real/cifar100_resnet18.pt` via `notebooks/01_data_and_embeddings.ipynb`.
- Total storage required: ~170 MB for raw CIFAR-100; ~120 MB for cached ResNet-18 embedding bundles.

---

## Getting Started (Local or Remote JupyterHub)

### Prerequisites
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or standard `python3 -m venv`

### Installation with `uv`

```bash
# Enter project directory
cd mcts-hnm

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Register the virtualenv as a Jupyter kernel on remote JupyterHub
python -m ipykernel install --user --name mcts-hnm --display-name "Python (MCTS-HNM)"
```

Alternatively, with standard `pip`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m ipykernel install --user --name mcts-hnm --display-name "Python (MCTS-HNM)"
```

### Running the Regression Test Suite

All 142+ unit, integration, and end-to-end notebook tests run in under 60 seconds:

```bash
pytest -v
```

---

## How to Run Experiments and Reproduce Results

### 1. Fast Pipeline Validation (SMOKE Mode)
Set environment variable `EXECUTION_MODE=smoke` or execute pytest:
```bash
EXECUTION_MODE=smoke pytest tests/test_automated_suite_real.py -v
```

### 2. Full Scientific Benchmark (REAL Mode)
On JupyterHub or a local workstation with CIFAR-100 available:
1. Open [`notebooks/01_data_and_embeddings.ipynb`](notebooks/01_data_and_embeddings.ipynb) and run all cells to extract real embeddings.
2. Open [`notebooks/02_tree_construction.ipynb`](notebooks/02_tree_construction.ipynb) to construct both semantic and confusion-driven trees.
3. Open [`notebooks/03_tier1_mcts_benchmarks.ipynb`](notebooks/03_tier1_mcts_benchmarks.ipynb) to execute Tier-1 multi-seed benchmarks across Uniform, OHEM, Strict GiantHunter, and Enhanced MCTS-HNM.
4. Open [`notebooks/04_tier2_e2e_and_expansion.ipynb`](notebooks/04_tier2_e2e_and_expansion.ipynb) to run Tier-2 ResNet-18 end-to-end vision training and dynamic leaf expansion.
5. Open [`notebooks/05_paper_artifacts.ipynb`](notebooks/05_paper_artifacts.ipynb) to ingest real benchmark outputs, render publication figures and tables into `paper/`, and compile `paper/main.pdf`.

Or execute the complete master pipeline sequentially in one click via [`notebooks/00_master_end_to_end.ipynb`](notebooks/00_master_end_to_end.ipynb).

### 3. Reproducing Publication Figures & Tables Programmatically
```python
from src.utils.experiment_config import ExecutionMode, ExperimentConfig
from src.utils.paper_artifacts import generate_all_paper_artifacts, compile_paper

config = ExperimentConfig(execution_mode=ExecutionMode.REAL)
artifacts = generate_all_paper_artifacts(
    results_dir="results/real",
    paper_dir="paper",
    metadata_path="paper_metadata.yaml",
    config=config,
)
pdf_path = compile_paper("paper")
print(f"Compiled paper: {pdf_path}")
```

---

## Experimental Limitations & Reproducibility Guide

For an in-depth scientific audit of historical vs verified results, budget equivalence ($K_{\text{neg}} = 48$), tree linkage analysis (Ward vs Average linkage), and known limitations, consult the detailed guide:

👉 **[Experimental Reproducibility Guide](docs/experimental_reproducibility.md)**

---

## Citation & References

- Qu, F., Peng, C., Guan, J., Wang, D., Sun, Y., & Shang, J. (2025). *GiantHunter: accurate detection of giant virus in metagenomic data using reinforcement-learning and Monte Carlo tree search*. **Bioinformatics**, 41(Supplement_1), i30–i39.
- Sutton, R. S., & Barto, A. G. (2020). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Auer, P. (2000). *Using confidence bounds for exploitation-exploration trade-offs*. Journal of Machine Learning Research.
