# F5: Dynamic Entropy Leaf Expansion & Tier-2 End-to-End Vision Pipeline Design

**Spec**: `.specs/features/05-extensions-and-e2e/spec.md`  
**Status**: Approved

---

## Architecture Overview

Feature `F5` extends the MCTS-HNM package with two P2 capabilities (`AD-001`, `AD-006`, `AD-008`):
1. **Dynamic Entropy Leaf Sub-Clustering (`src/mcts/expansion.py`)**: Evaluates per-sample Shannon prediction entropy $H(p)$ inside frequently visited class leaves ($n_l \ge N_{\text{split}}$) and mutates `FastHierarchyTree` in-place to replace leaf $l$ with $K_{\text{sub}}$ entropy-quantile child leaves.
2. **Tier-2 End-to-End Vision Pipeline (`src/training/e2e_trainer.py`)**: Trains a CIFAR-adapted ResNet-18 (`Tier2ResNet18`) on raw $32 \times 32$ RGB images across `uniform`, `ohem`, `mcts_strict`, and `mcts_enhanced` strategies, supporting both dynamic leaf expansion and periodic Ward confusion-tree rebuilds (`refresh_tree_every`), orchestrated via `notebooks/04_tier2_e2e_and_expansion.ipynb`.

```mermaid
flowchart TD
    subgraph Dynamic Leaf Expansion ["ISS-13: Dynamic Entropy Sub-Clustering (src/mcts/expansion.py)"]
        VisitCheck{"n_l >= N_split\n& base leaf?"} -->|"Yes"| ForwardPass["Forward Pass on Leaf Pool\n(torch.no_grad)"]
        ForwardPass --> Entropy["compute_prediction_entropy()\nH(p) per instance"]
        Entropy --> Quantiles["Sort by H(p) & np.array_split(K_sub)"]
        Quantiles --> MutateTree["expand_leaf_by_entropy()\nIn-place FastHierarchyTree Expansion"]
    end

    subgraph Tier2 Pipeline ["ISS-14: Tier-2 End-to-End ResNet-18 (src/training/e2e_trainer.py)"]
        Images[("Raw 32x32 RGB Images")] --> ResNet["Tier2ResNet18\n(Conv3x3 + Residual Blocks + Head)"]
        ResNet --> StepLoss["BCEWithLogitsLoss (Scen 1)\nor CrossEntropyLoss (Scen 2)"]
        StepLoss --> Backprop["MCTS Reward & Backprop\n+ Optional maybe_expand_leaf()"]
        Backprop --> RefreshCheck{"step % refresh_tree_every == 0\n& confusion_ward?"}
        RefreshCheck -->|"Yes"| RebuildTree["Compute Validation Confusion Matrix\n& Rebuild Ward Confusion Tree"]
    end

    Nb04["notebooks/04_tier2_e2e_and_expansion.ipynb"] --> Dynamic Leaf Expansion
    Nb04 --> Tier2 Pipeline
```

---

## Code Reuse Analysis

### Existing Components to Leverage

| Component | Location | How to Use |
| --------- | -------- | ---------- |
| `FastHierarchyTree` | `src/trees/base_tree.py` | Mutate arrays (`parents`, `children`, `child_counts`, `is_leaf`, `depths`, `leaf_to_class_id`, `instance_counts`, `visits`, `q_values`, `node_names`, `class_to_indices`) in-place during leaf splitting and validate via `validate_tree()` |
| `MCTSHNMAgent`, `compute_mcts_reward` | `src/mcts/agent.py` | Traverse expanded trees, compute step rewards, and propagate instance-weighted $Q$-values up to the root |
| `DualStreamBatchSampler` | `src/mcts/sampler.py` | Sample MCTS + uniform negative batches directly from newly created sub-cluster leaves via `tree.leaf_to_class_id` and `tree.class_to_indices` |
| `MCTSDiagnosticsTracker`, `compute_visitation_entropy`, `compute_superclass_visit_distribution` | `src/mcts/diagnostics.py` | Track visitation entropy and superclass distributions across static and dynamically expanded trees |
| `build_semantic_tree`, `build_confusion_tree` | `src/trees/semantic_tree.py`, `src/trees/confusion_tree.py` | Build initial hierarchies and periodically refresh Ward Confusion Trees during Tier-2 end-to-end training |
| `UniformSampler`, `OHEMSampler` | `src/training/baselines.py` | Baseline samplers reused for Tier-2 end-to-end comparisons |
| `compute_paired_statistical_tests` | `src/utils/statistics.py` | Compute paired Student's $t$-test and Wilcoxon signed-rank statistics for `run_e2e_experiment()` |
| `set_global_seed` | `src/utils/seed.py` | Enforce deterministic execution across seeds |

---

## Components

### 1. Dynamic Prediction-Entropy Leaf Expansion (`src/mcts/expansion.py`)

- **Purpose**: Compute per-instance prediction entropy $H(p)$ and dynamically split high-visit class leaves into $K_{\text{sub}}$ entropy-quantile child leaves in-place on `FastHierarchyTree`.
- **Location**: `src/mcts/expansion.py` (with superclass mapping support in `src/mcts/diagnostics.py` and trainer integration in `src/training/trainer.py`).
- **Interfaces**:
  - `SUBCLUSTER_ID_STRIDE: int = 1000`
  - `resolve_base_class_id(class_id: int) -> int`:
    Returns `int(class_id) % SUBCLUSTER_ID_STRIDE` for non-negative `class_id`.
  - `compute_prediction_entropy(logits: torch.Tensor | np.ndarray, is_binary: bool = True, eps: float = 1e-12) -> np.ndarray`:
    Computes 1D `float64` array of Shannon prediction entropies $H(p) \ge 0.0$ for binary logits (`shape: [N]` or `[N, 1]`) or multi-class logits (`shape: [N, C]`).
  - `expand_leaf_by_entropy(tree: FastHierarchyTree, leaf_node_id: int, entropies: np.ndarray, k_sub: int = 3) -> tuple[int, ...]`:
    Validates that `leaf_node_id` is an unexpanded leaf (`0 <= leaf_to_class_id[leaf_node_id] < SUBCLUSTER_ID_STRIDE`) with `instance_counts[leaf_node_id] >= k_sub` and `entropies.shape == (instance_counts[leaf_node_id],)`. Sorts the leaf's sample indices by ascending `(entropies, sample_index)`, splits into `k_sub` buckets via `np.array_split`, converts `leaf_node_id` into an internal node, appends `k_sub` new leaf nodes with virtual class IDs `(sub_idx + 1) * 1000 + base_class_id`, validates the mutated tree via `tree.validate_tree()`, and returns the tuple of new child node IDs.
  - `maybe_expand_leaf(tree: FastHierarchyTree, leaf_node_id: int, model: nn.Module, train_inputs: torch.Tensor, is_binary: bool = True, split_threshold: int = 50, k_sub: int = 3, batch_size: int = 256, device: torch.device | None = None) -> tuple[int, ...] | None`:
    Checks whether `tree.is_leaf[leaf_node_id]` is True, `0 <= tree.leaf_to_class_id[leaf_node_id] < SUBCLUSTER_ID_STRIDE`, `tree.visits[leaf_node_id] >= split_threshold`, and `tree.instance_counts[leaf_node_id] >= k_sub`. When all conditions hold, runs a batched `torch.no_grad()` forward pass on `train_inputs[leaf_indices]`, computes `compute_prediction_entropy()`, and calls `expand_leaf_by_entropy()`. Returns the new child node IDs if expanded, or `None` otherwise.

### 2. Tier-2 End-to-End ResNet-18 & E2E Trainer (`src/training/e2e_trainer.py`)

- **Purpose**: Provide a trainable CIFAR-adapted ResNet-18 architecture (`Tier2ResNet18`), image-level Scenario 1 & Scenario 2 evaluators and training loops (`train_e2e_scenario1`, `train_e2e_scenario2`), periodic Ward confusion-tree refresh (`refresh_confusion_tree_from_model`), and a multi-seed end-to-end experiment runner (`run_e2e_experiment`).
- **Location**: `src/training/e2e_trainer.py`
- **Interfaces**:
  - `Tier2ResNet18(num_outputs: int = 1, width_multiplier: float = 1.0, dropout: float = 0.1, squeeze_binary: bool = True)`:
    CIFAR-adapted ResNet-18 (`3x3` `conv1` with `stride=1, padding=1`, no initial `maxpool`, 4 residual stages `[2, 2, 2, 2]` with base channels scaled by `width_multiplier`, global average pooling, dropout, and linear classification head). Exposes `forward_features(x: torch.Tensor) -> torch.Tensor` and `forward(x: torch.Tensor) -> torch.Tensor`.
  - `refresh_confusion_tree_from_model(model: nn.Module, val_images: torch.Tensor, val_labels: np.ndarray | torch.Tensor, class_to_indices: dict[int, np.ndarray], num_classes: int = 100, is_binary: bool = False, batch_size: int = 256, smoothing: float = 1e-3, device: torch.device | None = None, seed: int = 42) -> FastHierarchyTree`:
    Extracts validation predictions/features from `model` under `torch.no_grad()`, builds a smoothed $K \times K$ confusion matrix, and returns a newly constructed Ward Confusion Tree (`build_confusion_tree`).
  - `train_e2e_scenario1(...) -> dict[str, object]`:
    Trains `Tier2ResNet18(num_outputs=1)` with `BCEWithLogitsLoss` on 4D RGB images `[N, 3, H, W]` across `uniform`, `ohem`, `mcts_strict`, and `mcts_enhanced`, supporting `enable_leaf_expansion`, `split_threshold`, `k_sub`, and `refresh_tree_every`.
  - `train_e2e_scenario2(...) -> dict[str, object]`:
    Trains `Tier2ResNet18(num_outputs=num_classes)` with `CrossEntropyLoss(label_smoothing=0.1)` on 4D RGB images `[N, 3, H, W]` across all 4 strategies with optional leaf expansion and periodic confusion-tree refresh.
  - `E2EExperimentConfig` & `run_e2e_experiment(config: E2EExperimentConfig, ...) -> dict[str, object]`:
    Executes multi-seed end-to-end experiments, aggregates `mean ± std`, computes paired statistical tests (`compute_paired_statistical_tests`) against `mcts_enhanced`, and atomically writes `.json` and `.csv` summaries to `output_dir`.

### 3. Versioned Orchestration Notebook (`notebooks/04_tier2_e2e_and_expansion.ipynb`)

- **Purpose**: Interactive JupyterHub notebook executing:
  1. Part 1: Dynamic Entropy Leaf Expansion Ablation (`enable_leaf_expansion=False` vs. `True` on Tier-1 embeddings, visualizing tree growth, entropy quantile $Q$-values, and test AUPRC/F1 gains).
  2. Part 2: Tier-2 End-to-End ResNet-18 Validation (`uniform`, `ohem`, `mcts_enhanced` on $32 \times 32$ RGB images with periodic confusion-tree refresh and dynamic leaf expansion).
  3. Summary visualization and atomic export of JSON/CSV results to `results/`.

---

## Error Handling Strategy

| Error Scenario | Handling | User Impact |
| -------------- | -------- | ----------- |
| `expand_leaf_by_entropy()` called on internal node, out-of-range node ID, `k_sub < 2`, or mismatched entropy length | Raise `ValueError` | Immediate fail-fast before corrupting tree arrays |
| `maybe_expand_leaf()` called on a sub-cluster leaf (`leaf_to_class_id >= 1000`) or leaf with `< k_sub` instances or `< split_threshold` visits | Return `None` without mutating tree | Safe no-op during training step loop |
| `k_sub` larger than existing `tree.children.shape[1]` (`B_max`) | Dynamically pad `tree.children` columns to `max(B_max, k_sub)` with `-1` | Supports arbitrary `k_sub >= 2` even on binary Ward trees (`B_max = 2`) |
| Invalid 4D image shapes, unknown strategy, or non-positive step/eval intervals in `e2e_trainer.py` | Raise `ValueError` | Descriptive diagnostic message before training starts |

---

## Risks & Concerns

| Concern | Location | Impact | Mitigation |
| ------- | -------- | ------ | ---------- |
| `compute_superclass_visit_distribution()` indexes `tax.fine_to_coarse_idx[cls_id]` directly, which would fail with `IndexError` when `cls_id >= 100` for dynamically expanded sub-clusters | `src/mcts/diagnostics.py:82` | Crash when recording diagnostics on an expanded tree | Update `compute_superclass_visit_distribution()` to map `base_cls_id = cls_id % 1000` (and `% len(tax.fine_to_coarse_idx)`) before looking up coarse superclass IDs |
| Binary Ward Confusion Trees initialize `tree.children` with shape `[V, 2]`; splitting a leaf into `k_sub = 3` or `4` children exceeds column width `2` | `src/trees/confusion_tree.py:145` | `IndexError` when assigning `k_sub > 2` children to a Ward tree leaf | In `expand_leaf_by_entropy()`, widen `tree.children` to `max(B_max, k_sub)` padded with `-1` before writing child IDs |
| Full-width ResNet-18 on CPU in unit tests can be slow if many steps are run | `src/training/e2e_trainer.py` | Slow unit test execution | Parameterize `Tier2ResNet18(width_multiplier=...)` (default `1.0` for full ResNet-18 `[64, 128, 256, 512]`, `0.25` in fast unit tests) |

---

## Tech Decisions

| Decision | Choice | Rationale |
| -------- | ------ | --------- |
| Virtual class ID encoding for leaf sub-clusters | `sub_cls_id = (sub_idx + 1) * 1000 + base_class_id` | Keeps `leaf_to_class_id` unique across all leaves (satisfying `FastHierarchyTree.validate_tree()`), preserves `.save()`/`.load()` compatibility, and allows O(1) recovery of `base_class_id = sub_cls_id % 1000` |
| In-place `FastHierarchyTree` mutation on leaf expansion | Reassign contiguous NumPy attributes on the existing `FastHierarchyTree` instance | Ensures existing references held by `MCTSHNMAgent`, `DualStreamBatchSampler`, and `MCTSDiagnosticsTracker` immediately see the expanded tree without object replacement |
