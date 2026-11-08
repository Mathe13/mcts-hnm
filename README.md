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
├── docs/                       # Project documentation
│   ├── ARCHITECTURE.md         # Detailed technical & RL architecture
│   └── AGENTS.md               # Development rules for AI agents
├── src/                        # Core source code
│   ├── trees/                  # Semantic and confusion-based tree builders
│   ├── mcts/                   # MCTS agent, UCB selection, and backpropagation
│   ├── models/                 # Classifier heads and feature extractors
│   ├── training/               # Dual-stream trainer and baseline runners (Uniform, OHEM)
│   └── utils/                  # Metrics (F1, AUPRC, tree visit entropy) and plotting
├── experiments/                # Experiment configs and runner scripts
│   ├── configs/                # YAML configuration files
│   ├── run_prototype.py        # Fast embedding experiments
│   └── run_e2e.py              # End-to-end CNN experiments
├── paper/                      # Scientific paper LaTeX draft (ArXiv / Prime AI template)
├── tests/                      # Unit and integration test suite
├── pyproject.toml              # Project dependencies and packaging
├── README.md                   # This file
└── ARCHITECTURE.md             # System architecture specification
```

---

## Getting Started

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
```

Alternatively, with standard `pip`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Experimental Workflow

1. **Extract Embeddings (Fast Mode):**
   ```bash
   python src/utils/extract_embeddings.py --dataset cifar100 --backbone resnet18
   ```

2. **Build Confusion Tree:**
   ```bash
   python src/trees/build_confusion_tree.py --embeddings-dir data/embeddings
   ```

3. **Run Comparative Benchmark (MCTS vs. Baselines):**
   ```bash
   python experiments/run_prototype.py --config experiments/configs/comparative_study.yaml --seeds 5
   ```

4. **Generate Paper Figures & LaTeX Tables:**
   ```bash
   python src/utils/generate_paper_artifacts.py --results-dir results/
   ```

---

## Citation & References

- Qu, F., Peng, C., Guan, J., Wang, D., Sun, Y., & Shang, J. (2025). *GiantHunter: accurate detection of giant virus in metagenomic data using reinforcement-learning and Monte Carlo tree search*. **Bioinformatics**, 41(Supplement_1), i30–i39.
- Sutton, R. S., & Barto, A. G. (2020). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Auer, P. (2000). *Using confidence bounds for exploitation-exploration trade-offs*. Journal of Machine Learning Research.
