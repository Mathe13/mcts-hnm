# Tier-2 End-to-End Vision - Checks

Profile: light
Plan: `.specs/features/10-tier2-e2e-real-cifar/plan.md`

## Intent

10 checks in 3 slices · 3 one-way doors · 0 open

## Checks

### S1 - Execution Mode Enforcement and Real CIFAR-100 Image Loading · 3 files · 25 KB · ~8k

**C1** - `load_cifar100_raw_images` in REAL mode with local files missing and download=False raises `FileNotFoundError` (S1, AC 1)
Proof: `pytest tests/test_tier2_e2e_real.py::test_load_cifar100_real_missing_raises`

**C2** - `load_cifar100_raw_images` in SMOKE mode returns deterministic synthetic 32x32 images (S1, AC 2)
Proof: `pytest tests/test_tier2_e2e_real.py::test_load_cifar100_smoke_synthetic_images`

**C3** - Notebook 04 initial cell declares `EXECUTION_MODE = ExecutionMode.REAL` and outputs banner (S1, AC 3)
Proof: `pytest tests/test_tier2_e2e_real.py::test_notebook_04_structure_and_banner`

**C4** - Notebook 04 in SMOKE mode writes results to `results/smoke/` and checkpoints to `checkpoints/smoke/` (S1, AC 4)
Proof: `pytest tests/test_tier2_e2e_real.py::test_notebook_04_smoke_execution`

### S2 - Tier-2 E2E Vision Pipeline and Checkpoint Persistence · 3 files · 30 KB · ~10k

**C5** - `run_e2e_experiment` writes `.meta.json` sidecar alongside exported JSON results (S2, AC 1)
Proof: `pytest tests/test_tier2_e2e_real.py::test_run_e2e_experiment_provenance_metadata`

**C6** - `train_e2e_scenario1` saves model checkpoint to `checkpoints/{mode}/` (S2, AC 2)
Proof: `pytest tests/test_tier2_e2e_real.py::test_train_e2e_scenario1_checkpoint_saving`

**C7** - `train_e2e_scenario2` saves model checkpoint to `checkpoints/{mode}/` (S2, AC 2)
Proof: `pytest tests/test_tier2_e2e_real.py::test_train_e2e_scenario2_checkpoint_saving`

**C8** - `run_e2e_experiment` records per-seed metrics and paired statistical tests for multi-seed runs (S2, AC 3)
Proof: `pytest tests/test_tier2_e2e_real.py::test_run_e2e_experiment_multi_seed_and_stats`

### S3 - Dynamic Leaf Expansion and Tree Growth Auditing · 3 files · 25 KB · ~8k

**C9** - Dynamic leaf expansion in `train_scenario1` increases leaf count when visits exceed threshold (S3, AC 1)
Proof: `pytest tests/test_tier2_e2e_real.py::test_dynamic_leaf_expansion_growth`

**C10** - Dynamically expanded tree preserves structural invariants and passes `validate_tree()` (S3, AC 2)
Proof: `pytest tests/test_tier2_e2e_real.py::test_dynamically_expanded_tree_invariants`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| `load_cifar100_raw_images()` statuses (3) | 200 C2 · 400 C1 · 422 C1 | - |
| `run_e2e_experiment()` statuses (3) | 200 C5 · 400 C8 · 422 C8 | - |
| `train_e2e_scenario1()` statuses (2) | 200 C6 · 400 C6 | - |
| `train_e2e_scenario2()` statuses (1) | 200 C7 | - |
| dynamic leaf expansion statuses (2) | growth C9 · invariant C10 | - |

- Claims naming a status code, route or response shape: C1, C2, C5, C6, C7, C8 - each has a proof that crosses the boundary
- No other check claims more than the single case its proof exercises

## Swept

- validation: C1, C3
- failure modes: C1
- idempotency: C5
- authorization: n/a - local research experimentation without user auth roles
- concurrency: C8
- data lifecycle: C6, C7
- dependency failure: C1
- state transitions: C9, C10
- observability: C3, C4, C5, C9

## Handoff

- S1-S3 = 80 KB total across dataset loading, training, expansion, and tests; estimate ~26k tokens, well under the 150k budget - one builder
