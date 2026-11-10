# Experimental Reproducibility Guide

This document provides a comprehensive scientific audit, methodological reference, and execution manual for reproducing the experiments and publication artifacts of **MCTS-HNM: Monte Carlo Tree Search for Hierarchical Hard Negative Mining in Class-Imbalanced Learning**.

---

## 1. Global Execution Contract

To prevent silent synthetic fallbacks and ensure cryptographic traceability, all experimental execution is strictly governed by `src.utils.experiment_config.ExecutionMode`:

| Execution Mode | Target Use Case | Dataset & Cache Rules | Output Directory | Fallback Allowed? |
| :--- | :--- | :--- | :--- | :--- |
| **`ExecutionMode.SMOKE`** | Rapid CI regression, unit tests, fast notebook dry-runs | Deterministic synthetic embeddings and 32x32 image fixtures | `results/smoke/`, `embeddings/smoke/`, `checkpoints/smoke/` | **YES** (strictly tagged as synthetic) |
| **`ExecutionMode.REAL`** | Definitive scientific experiments and paper figures/tables | Genuine CIFAR-100 dataset (`cifar100_resnet18.pt`) | `results/real/`, `embeddings/real/`, `checkpoints/real/`, `paper/` | **NO** (raises explicit `FileNotFoundError`) |

Cross-mode cache mixing is strictly prohibited:
- Attempting to load a `SMOKE` cache in `REAL` mode raises a `ValueError`.
- Attempting to mutate `ExecutionMode` within the same process session raises an `IllegalStateError`.
- Every generated metric file (`.json`, `.csv`) is paired with an auditable `.meta.json` sidecar containing dataset fingerprints, git commit SHA, UTC timestamp, and execution mode.

---

## 2. Experiment Catalog & Historical Audit

The sections below classify the status of all experimental investigations in the repository:

### 2.1 Completed Experiments
- **Tier-1 Axis 1: Samplers** (Scenario 1 One-vs-Rest): Multi-seed ($N=5$) benchmark comparing Uniform, OHEM, Strict GiantHunter, and Enhanced MCTS-HNM with normalized negative mining budget equivalence ($K_{\text{neg}} = 48$).
- **Tier-1 Axis 2: Tree Architectures** (Scenario 1 & 2): Systematic comparative ablation between the Native Semantic Tree and the Ward Confusion Tree.
- **Tier-1 Scenario 2 Curriculum** (100-way multiclass): Multi-class active curriculum with MCTS diagnostics (visitation entropy, unvisited class counts, per-class recall, and coverage percentage).
- **Hyperparameter Sensitivity Sweeps**: Executable parameter sweep over exploration constant $c \in [0.25, 3.0]$ and momentum $\alpha \in [0.05, 0.9]$.
- **Tier-2 ResNet-18 End-to-End**: Vision backbone training on genuine CIFAR-100 32x32 images with saved weights (`checkpoints/{mode}/`).
- **Dynamic Leaf Expansion**: Prediction-entropy sub-clustering ($k_{\text{sub}}=3$, threshold=50) with tree invariant validation (`tree.validate_tree()`).

### 2.2 Pending Experiments
- **Full 200-Epoch GPU Sweep**: Full multi-GPU vision pre-training across the complete training split for publication cameras. Validated end-to-end via the reduced real integration path.

### 2.3 Historical Results Lacking Sufficient Provenance
- **Historical Tier-1 MVP**: Early prototype runs located in `results/legacy/` pre-dating the execution contract. Archived because they lacked `.meta.json` sidecars, negative sampling budget normalization, and mode isolation.

| Experiment Identifier | Target Scenario | Primary Metrics | Provenance & Validation Status | Artifact Location |
| :--- | :--- | :--- | :--- | :--- |
| **Historical Tier-1 MVP** | Scenario 1 ($\rho=1:10, 50, 100$) | Minority F1, AUPRC, FPR@95 | **Archived**: Pre-dates execution contract; lacked sidecars and negative budget equivalence. | `results/legacy/` |
| **Tier-1 Axis 1: Samplers** | Scenario 1 (One-vs-Rest) | F1-Score, AUPRC, FPR@95 | **Completed**: Multi-seed ($N=5$), $K_{\text{neg}}=48$ budget equivalence across Uniform, OHEM, Strict, Enhanced. | `results/{mode}/tier1_axis1_samplers.json` |
| **Tier-1 Axis 2: Trees** | Scenario 1 & 2 | Macro-F1, AUPRC, Recall | **Completed**: Comparative ablation between Semantic Tree and Ward Confusion Tree. | `results/{mode}/tier1_axis2_tree_ablation.json` |
| **Tier-1 Scenario 2 Curriculum** | Scenario 2 (100-way) | Top-1 Accuracy, Macro-F1, Worst-10 Recall | **Completed**: Multiclass MCTS diagnostics (visitation entropy, unvisited class count, class coverage). | `results/{mode}/tier1_scenario2_curriculum.json` |
| **Sensitivity Analysis** | Scenario 1 ($\rho=1:10$) | Minority F1-Score | **Completed**: Executable parameter sweep over exploration constant $c \in [0.25, 3.0]$ and momentum $\alpha \in [0.05, 0.9]$. | `results/{mode}/tier1_hyperparameter_sensitivity.json` |
| **Tier-2 ResNet-18 E2E** | Scenario 1 & 2 | End-to-End F1, AUPRC | **Completed**: Vision backbone fine-tuning on genuine CIFAR-100 32x32 images with saved model weights. | `checkpoints/{mode}/tier2_resnet18_*.pt` |
| **Dynamic Leaf Expansion** | Scenario 1 ($\rho=1:10$) | Leaf count, macro-F1 | **Completed**: Entropy sub-clustering ($k_{\text{sub}}=3$, threshold=50) with tree invariant validation (`validate_tree`). | `results/{mode}/e2e_scenario_1_results.json` |
| **Full 200-Epoch GPU Sweep** | Scenario 1 & 2 | Publication Final Accuracy | **Pending Post-Seminar**: Requires dedicated compute cluster; verified with reduced integration test path. | N/A |

---

## 3. Methodological Details

### 3.1 Tree Geometry: Semantic vs. Ward vs. Average Linkage
- **Native Semantic Tree**: Constructed directly from the human conceptual taxonomy of CIFAR-100 (20 superclasses containing 5 fine classes each). Leaves represent fine classes; root node connects all 20 superclasses.
- **Data-Driven Ward Confusion Tree**: Built by applying Ward agglomerative hierarchical clustering to an empirical confusion dissimilarity matrix $D(i, j) = 1.0 - \frac{C_{ij} + C_{ji}}{C_{ii} + C_{jj} + \epsilon}$ estimated from a validation baseline.
- **Methodological Note (Linkage Comparison)**: The historical implementation used Ward linkage directly on dissimilarity values. The implementation supports both `linkage_method="ward"` (default, preserving historical baseline consistency) and `linkage_method="average"` (mathematically standard for non-Euclidean dissimilarity matrices), selectable via `build_confusion_tree`.

### 3.2 Negative Mining Budget Equivalence
Under Scenario 1 (Imbalanced Binary One-vs-Rest with target class $y^*$), previous implementations suffered from unequal computational exposure. All samplers are now strictly normalized to an identical batch budget:
- **Positive Batch Size**: $K_{\text{pos}} = 16$.
- **Negative Batch Size**: $K_{\text{neg}} = 48$.
- **Total Minibatch Size**: $B = 64$.
- **Sampler Allocations**:
  - **Uniform Random**: $K_{\text{neg}} = 48$ samples chosen uniformly at random across all 99 non-target classes.
  - **OHEM**: 96 candidates sampled uniformly, loss evaluated, and top $K_{\text{neg}} = 48$ highest-loss samples selected.
  - **Strict GiantHunter**: MCTS tree search yields class leaf $k$; $K_{\text{neg}} = 48$ samples drawn exclusively from leaf $k$.
  - **Enhanced MCTS-HNM (Dual-Stream)**: Dual-stream hybrid draws $K_{\text{mcts}} = 24$ samples from MCTS leaf $k$ and $K_{\text{uniform}} = 24$ samples uniformly across all negative classes ($K_{\text{mcts}} + K_{\text{uniform}} = 48$).

### 3.3 Dynamic Leaf Expansion Criteria
To explore sub-class instance difficulty, leaf expansion dynamically subdivides fine-class leaves when visitation exceeds `split_threshold = 50`:
1. Prediction entropy $H(x) = -\sum_{c} p_c \log p_c$ is computed for instances mapped to the leaf.
2. Instances are clustered into $k_{\text{sub}} = 3$ sub-leaves using $k$-means on feature representations.
3. The parent node's leaf flag is cleared, and children are registered into the tree index.
4. Structural invariants are verified via `tree.validate_tree()`, ensuring tree depth and leaf masks remain valid.

---

## 4. Step-by-Step Reproduction Recipes

### 4.1 Automated Smoke Regression Suite (< 60s)
Execute the complete regression test suite across all six notebooks, model components, and mode boundaries:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all 142+ unit and integration tests
pytest -v
```

### 4.2 Executing the Notebook Workflow on JupyterHub
Run notebooks sequentially in order of dependency:

1. [`notebooks/00_master_end_to_end.ipynb`](../notebooks/00_master_end_to_end.ipynb): Master pipeline orchestrator.
2. [`notebooks/01_data_and_embeddings.ipynb`](../notebooks/01_data_and_embeddings.ipynb): Download CIFAR-100 and extract ResNet-18 embeddings (`embeddings/{mode}/cifar100_resnet18.pt`).
3. [`notebooks/02_tree_construction.ipynb`](../notebooks/02_tree_construction.ipynb): Construct and serialize semantic and confusion hierarchies (`embeddings/{mode}/trees/`).
4. [`notebooks/03_tier1_mcts_benchmarks.ipynb`](../notebooks/03_tier1_mcts_benchmarks.ipynb): Execute multi-seed Tier-1 benchmarks (`results/{mode}/`).
5. [`notebooks/04_tier2_e2e_and_expansion.ipynb`](../notebooks/04_tier2_e2e_and_expansion.ipynb): Run Tier-2 ResNet-18 end-to-end vision training and dynamic leaf expansion.
6. [`notebooks/05_paper_artifacts.ipynb`](../notebooks/05_paper_artifacts.ipynb): Ingest benchmark JSON results, generate publication figures/tables, and compile `paper/main.pdf`.

To run in `REAL` mode on JupyterHub, ensure `EXECUTION_MODE = ExecutionMode.REAL` is set in Cell 1 of each notebook.

### 4.3 Reproducing Publication Figures and LaTeX Tables
To generate all paper figures and tables directly from the Python API:

```python
from src.utils.experiment_config import ExecutionMode, ExperimentConfig
from src.utils.paper_artifacts import generate_all_paper_artifacts, compile_paper

# Configure for scientific reproduction
config = ExperimentConfig(execution_mode=ExecutionMode.REAL)

# Generate publication artifacts and provenance manifest
artifacts = generate_all_paper_artifacts(
    results_dir="results/real",
    paper_dir="paper",
    metadata_path="paper_metadata.yaml",
    config=config,
)

# Compile LaTeX manuscript into PDF
compile_paper("paper")
```

The resulting files will be written to:
- `paper/figures/`: `fig_convergence_curves.pdf`, `fig_tree_visitation_heatmap.pdf`, `fig_hyperparameter_sensitivity.pdf` (vector PDF and 300 DPI raster PNG).
- `paper/tables/`: `tab_scenario1_imbalanced.tex`, `tab_scenario2_multiclass.tex`, `tab_ablation_study.tex`.
- `paper/artifact_manifest.json`: Full SHA-256 provenance manifest.
- `paper/main.pdf`: Compiled camera-ready PDF manuscript.

---

## 5. Known Experimental Limitations

1. **Fixed Backbone Representations**: Tier-1 benchmarks evaluate MCTS search efficiency on frozen ResNet-18 feature vectors. Full backpropagation through 200 epochs of CNN convolutional layers represents a significant compute requirement and is tested as an integration path in Tier-2.
2. **Class-Level Leaf Boundaries**: In the default tree hierarchy, leaf nodes correspond to 100 fine classes. Dynamic leaf expansion subdivides difficult classes into instance sub-clusters, introducing additional tree depth and branching overhead.
3. **Imbalance Setup**: Scenario 1 simulates imbalance by selecting a single rare class against 99 negative classes. Cumulative exposures depend on total step count and minibatch budget allocations.
