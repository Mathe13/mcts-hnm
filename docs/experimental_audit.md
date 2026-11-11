# Experimental Audit and Historical Inventory

**Status**: Active  
**Last Updated**: 2026-11-11  
**Scope**: Governance and Immutability of Historical Scientific Artifacts (`roadmap_mcts_hnm_proxima_run_v2.md` Phase A, Item A1)

---

## 1. Executive Summary

This document establishes the official experimental audit and inventory for the `mcts-hnm` repository, freezing historical campaign artifacts and defining provenance requirements for all subsequent experimental runs.

Per **Roadmap v2 Item A1**, historical experimental runs and configurations are permanently frozen and immutable. No new experiment may overwrite, alter, or silently combine results across distinct campaigns or execution modes (`SMOKE` vs `REAL`).

---

## 2. Directory Layout and Immutability Boundary

The repository enforces strict separation across experimental directories:

```
results/
├── legacy/     # [IMMUTABLE] Frozen historical run artifacts from prior campaigns
├── real/       # Full-scale scientific benchmarks executed on verified embeddings/datasets
│   ├── runs/   # Granular individual run records indexed by unique run_id
│   └── paper/  # Final publication figures and LaTeX tables generated from REAL runs
└── smoke/      # Fast verification test artifacts (watermarked, never for publication)
    ├── runs/
    └── paper/

embeddings/
├── real/       # Real ResNet-18 feature tensors extracted from official CIFAR-100
├── smoke/      # Synthetic test fixtures for unit and integration testing
└── trees/      # Serialized hierarchy trees (Semantic and Ward/Average Confusion)
```

### Immutability Invariants

1. **`results/legacy/` is Read-Only**: Code execution routines SHALL NOT write, overwrite, or mutate any artifact inside `results/legacy/`. Any attempt raises `PermissionError`.
2. **Mode Confinement**: Execution under `ExecutionMode.SMOKE` writes strictly to `results/smoke/`; `ExecutionMode.REAL` writes strictly to `results/real/`. Mixing paths raises `ValueError`.
3. **No Synthetic Fallback**: In `ExecutionMode.REAL`, missing inputs raise `FileNotFoundError`. Synthetic prototypes or random tensors are strictly forbidden in `REAL` mode.

---

## 3. Embedding and Tree Artifact Hashes

Verified SHA-256 digests for cached feature representations and tree hierarchies:

| Artifact Path | Format | Role | SHA-256 Digest |
| ------------- | ------ | ---- | -------------- |
| `embeddings/trees/confusion_tree_scenario2.npz` | NPZ | Ward Confusion Tree (Scenario 2) | `7f5fb3cf1be6123fdb8be728d00b0ece4f8b131db033fc1d66f9e710aa0e0b9c` |
| `embeddings/trees/semantic_tree_scenario2.npz` | NPZ | Semantic Superclass Tree (Scenario 2) | `03b320be7c1a5961fa3fa201ee142c9808d99882cf1b81edab120a5dc1cfa4fc` |
| `embeddings/smoke/cifar100_resnet18.pt` | PyTorch | Synthetic Smoke Embedding Bundle | `0bdf7fb4332962216c548b61c5a2bd5e355447487fd81083461847204fa94cc9` |

---

## 4. Experimental Provenance Requirements

Every experimental run executing in the repository SHALL generate a provenance manifest recording:

1. **`run_id`**: Canonical identifier: `{scenario}_cls{target_class}_r{ratio}_{tree}_{strategy}_s{seed}_{commit}`
2. **`git_commit`**: Full 40-character git commit hash at execution time.
3. **`timestamp_utc`**: ISO-8601 UTC execution timestamp.
4. **`hardware`**: CPU architecture, GPU model (if present), and total memory.
5. **`dependencies`**: Versions of `torch`, `numpy`, `scipy`, `scikit-learn`, and Python runtime.
6. **`resolved_config`**: Exact hyperparameters used (seed, batch sizes, learning rate, $c$, $\alpha$, steps).
7. **`input_hashes`**: SHA-256 digests of embeddings and split indices.

---

## 5. Audit Compliance Checklist

- [x] Historical directories inventoried and immutability boundary defined.
- [x] Embedding and tree checksums recorded.
- [x] Run manifest schema standardized.
- [x] Zero silent mixing between execution campaigns.
