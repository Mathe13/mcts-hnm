# MCTS-HNM — Roadmap for Fixes, Experimental Standardization, and Reproducibility

**Project:** `Mathe13/mcts-hnm`  
**Objective:** Fix identified issues across experimental notebooks, ensure strict isolation between smoke tests and real experiments, and establish a reproducible pipeline for generating scientific results.

## 1. General Agent Instructions

Act as a Senior Machine Learning Engineer specializing in scientific experimentation, PyTorch, MCTS, reproducibility, and automated testing.

Implement fixes incrementally, preserving the existing architecture wherever possible.

### Mandatory Rules

1. Do not remove smoke tests.
2. Do not mix synthetic and real data within the same notebook, regardless of the cell.
3. Do not allow automatic fallback from real data to synthetic data.
4. Every smoke test run must produce an explicit and easily identifiable warning banner in the output.
5. The execution mode must be defined once, prior to any data loading or generation.
6. No notebook should change its mode during execution.
7. Synthetic results must never feed into figures, tables, or scientific conclusions as if they were real results.
8. Caches, checkpoints, and results must be isolated by mode.
9. Preserve historical results until verifying that the migration is correct.
10. Maintain compatibility with remote execution via JupyterHub, CUDA, and CPU.
11. Do not silently alter the semantics of existing algorithms or metrics.
12. Do not invent metrics, replace missing results with constants, or backfill unexecuted experiments.
13. Implement unit and integration tests for relevant changes.
14. Do not run full REAL experiments automatically during development. Use smoke tests and small tests to validate the implementation.

Before modifying any files, inspect the current version of the repository and identify existing APIs, structures, and conventions. Paths mentioned in this document are suggestions and should be adapted to the actual architecture.

---

# Phase 0 — Audit and Baseline

**Priority: P0**

Before modifying the implementation:

- Identify all synthetic data generation (`torch.randn`, `np.random`, fixtures, artificial prototypes).
- Identify silent fallbacks.
- Identify relative paths dependent on the working execution directory.
- Locate all read and write points for results.
- Locate configurations for seeds, number of steps, and dataset sizes.
- Inspect the contracts of `run_benchmark`, `run_e2e_experiment`, `generate_all_paper_artifacts`, and helper functions.
- Identify which experiments have evidence of real execution.

Preserve existing Tier-1 CSVs and JSONs without overwriting them.

**Deliverable:** Short audit report containing file, function, issue, risk, and planned fix.

# Phase 1 — Global Execution Contract

**Priority: P0 — Blocker**

## 1.1 Create Centralized Configuration

Create a module, preferably:

`src/utils/experiment_config.py`

Implement an enum or equivalent:

```python
class ExecutionMode(str, Enum):
    SMOKE = "smoke"
    REAL = "real"
```

Create a validated configuration containing, at minimum:

- `execution_mode`
- `seed` or `seeds`
- `dataset_name`
- `dataset_source`
- `dataset_fingerprint`, when available
- `num_steps`
- `eval_every`
- `output_dir`
- `cache_dir`
- `device`
- `config_version`

All notebooks must load this same configuration, with experiment-specific parameters where necessary.

The REAL mode should be the recommended default. It must never automatically switch to SMOKE due to missing files, GPU failure, out-of-memory errors, or unavailable dependencies.

## 1.2 Make Mode Explicit in Notebooks

The first executable cell must contain the mode selection or load a pre-defined configuration.

Example of desired interface:

```python
EXECUTION_MODE = ExecutionMode.REAL
```

The configuration must be propagated to functions responsible for data, training, evaluation, and export.

### Fundamental Rule

`SMOKE` denotes technical pipeline validation. `REAL` denotes experimental execution with real data and genuinely computed metrics.

SMOKE may use synthetic data, downsized models, and fewer steps, but only in a completely isolated environment.

REAL must not consume synthetic fixtures, SMOKE artifacts, or fabricated results.

## 1.3 Implement Mandatory Warning Banner

Create a reusable function:

```python
print_execution_banner(config, notebook_name)
```

Every notebook must print the mode at startup.

Mandatory example in SMOKE:

```text
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! SMOKE TEST — SYNTHETIC/TEST EXECUTION !!!
!!! NOT VALID FOR SCIENTIFIC REPORTING    !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Notebook: 04_tier2_e2e
Mode: SMOKE
Output directory: results/smoke/
```

In REAL:

```text
==============================================
EXPERIMENT MODE: REAL
Dataset: CIFAR-100
Synthetic fallback: DISABLED
Scientific results: ENABLED
==============================================
```

The SMOKE warning must also appear when a smoke test routine is initiated directly, outside of notebooks.

Banner output should be centralized to avoid inconsistencies, without producing one message per training batch.

## 1.4 Isolate Data and Artifacts

Implement separate directories:

```text
embeddings/
    real/
    smoke/

results/
    real/
    smoke/
    legacy/

checkpoints/
    real/
    smoke/

paper/
    figures/
    tables/
```

Migration must respect existing paths and avoid overwriting earlier results.

Every generated experimental file must record its origin and mode, directly within the file or in auxiliary metadata.

Minimum fields:

```json
{
  "execution_mode": "real",
  "dataset": "CIFAR-100",
  "data_source": "cifar100_resnet18.pt",
  "seed": 42,
  "synthetic_data": false,
  "num_steps": 100,
  "git_commit": "...",
  "timestamp_utc": "..."
}
```

The `synthetic_data` field must reflect the true provenance of the data, not merely a declared configuration.

In case of conflict between metadata and the requested mode, halt execution.

**Acceptance Criteria:**

- All notebooks feature explicit mode selection.
- All print the proper warning banner.
- No silent fallback remains.
- SMOKE cannot write to REAL results.
- REAL cannot load SMOKE caches.
- Mode changes during execution are rejected.

---

# Phase 2 — Fix Notebooks 00, 01, and 02

**Priority: P0**

## Notebook 00 — Master / Orchestration

Replace local controls such as `FAST_MODE` with the global contract.

### SMOKE

- Use explicitly identified synthetic data or test fixtures.
- Execute reduced end-to-end runs.
- Verify tree construction, sampling, training, evaluation, and export.
- Log SMOKE warning.
- Do not access REAL data or results.

### REAL

- Exclusively load CIFAR-100 and its real artifacts.
- Use trees constructed from real data.
- Execute declared experimental configurations.
- Fail explicitly if any real dependency is missing.

Avoid duplicating logic from other notebooks. Notebook 00 should preferably orchestrate existing modules.

## Notebook 01 — Data & Embeddings

### SMOKE

- Generate a small, deterministic synthetic dataset.
- Validate extraction, serialization, and loading of embeddings.
- Save exclusively to the SMOKE cache.
- Record metadata identifying data as synthetic.

### REAL

- Load real CIFAR-100.
- Use the specified backbone and preprocessing.
- Preserve train, validation, and test splits.
- Produce real embeddings with validated dimensions and counts.
- Record backbone, weights, transforms, and dataset fingerprint.

Absence of CIFAR-100 must raise an error, never fall back.

## Notebook 02 — Tree Construction

### SMOKE

- Build both the semantic tree and confusion tree over a consistent synthetic dataset.
- Exclusively use SMOKE embeddings and labels.
- Generate visualizations and `.npz` files tagged as SMOKE.

### REAL

- Exclusively load real embeddings from notebook 01.
- Build semantic and confusion-based hierarchies using real data.
- Preserve separation between training and validation.
- Serialize trees with provenance metadata.

### Methodological Review

Investigate the use of Ward linkage with distances derived from the confusion matrix.

The current implementation should be compared against mathematically sound alternatives:

1. Ward applied to Euclidean class-representative vectors.
2. Average linkage applied directly to the confusion dissimilarity matrix.

Do not replace the historical implementation without preserving its results and documenting the methodological shift.

Create tests for:

- Number of nodes and leaves.
- Instance counts.
- Index integrity.
- Serialization and deserialization.
- Compatibility between hierarchy and dataset.
- Absence of leakage across splits.
- Absence of mode mixing.

**Acceptance Criterion:** Both modes execute the entire notebook, generating valid trees in isolated directories.

---

# Phase 3 — Fix Notebook 03: Tier-1 Benchmarks

**Priority: P1**

Existing results using CIFAR-100 embeddings must be preserved.

## 3.1 Eliminate Synthetic Fallback

Remove behavior that generates artificial prototypes when `cifar100_resnet18.pt` is missing, when in REAL mode.

Mandatory behavior:

- REAL: load validated real cache or raise an error.
- SMOKE: load or create SMOKE cache only.
- Never switch modes automatically.

## 3.2 Configure Benchmarks

Centralize:

- Negative sampling strategies.
- Hierarchy types.
- Seeds.
- Number of steps.
- Evaluation frequency.
- Imbalance ratios.
- Target class.
- MCTS hyperparameters.

Preserve existing metrics and experiment names where possible.

Reduced step counts may be retained in SMOKE. In REAL, use an explicit experimental configuration, without evaluating the validity of an experiment solely by its step count.

## 3.3 Verify Scenario 1

Audit the meaning and implementation of the 1:50 imbalance ratio.

Distinguish:

- Number of available positives.
- Number of available negatives.
- Effective ratio per minibatch.
- Effective cumulative ratio during training.
- Class distribution during evaluation.

Log these values in the results.

Verify computational budget equivalence between Uniform, OHEM, Strict GiantHunter, and Enhanced MCTS.

## 3.4 Investigate Scenario 2

Previous results indicated significantly worse multiclass performance for MCTS compared to Uniform.

Add instrumentation for:

- Visit distribution per node and class.
- Sampling frequency.
- Effective minibatch distribution.
- Unvisited classes.
- Per-class recall.
- Gradients and loss progression.
- Class coverage per strategy.
- MCTS update frequency.

Verify implementation, metrics, and fairness of comparison before modifying the algorithm.

Do not tune hyperparameters exclusively on the test set.

## 3.5 Complete Announced Experiments

Implement or explicitly mark as pending the sensitivity analysis for:

- Exploration parameter `c`.
- Momentum parameter `alpha`.

Avoid presenting experiments merely described in the notebook as if they had been executed.

## 3.6 Export

Export individual results per seed and aggregated summaries.

Files must include configuration, mode, embedding provenance, metrics, and experiment identification.

**Acceptance Criteria:**

- The entire notebook functions in both modes.
- REAL exclusively uses real embeddings.
- SMOKE is explicitly identified.
- Historical results are preserved.
- Statistics can be reproduced from individual results.
- No invented metrics or synthetic fill-ins exist.

---

# Phase 4 — Fix Notebook 04: Tier-2 End-to-End

**Priority: P1 — Essential Experimental Fix**

The previous version uses `torch.randn` to create images. Preserve this test exclusively in SMOKE mode.

## 4.1 SMOKE Mode

Maintain a small, fast test covering:

- Dynamic leaf expansion.
- End-to-end training.
- Uniform and MCTS strategies.
- Tree updating, where applicable.
- Serialization of metrics and checkpoints.

The data used must be fully synthetic and identified as such.

## 4.2 REAL Mode

Implement loading of real CIFAR-100 images.

Requirements:

- Consistent train, validation, and test splits.
- Appropriate transforms for ResNet-18.
- Training with real images.
- Classes and indices consistent with hierarchies.
- Comparison between methods using the same splits and seeds.
- Comparable model, initialization, and computational budget.
- Persistence of metrics and checkpoints.
- Prohibition against using the same split for training and testing.

## 4.3 Dynamic Expansion

Evaluate leaf expansion under controlled conditions.

Compare:

- MCTS without dynamic expansion.
- MCTS with dynamic expansion.

Record:

- Initial and final number of leaves.
- Expansion frequency.
- Expansion criteria.
- Instance distribution per leaf.
- Additional training and sampling overheads.

If an experiment depends on a previously constructed static hierarchy, validate its provenance prior to loading.

**Acceptance Criterion:** SMOKE runs exclusively with synthetic data; REAL runs exclusively with real images and independent splits.

---

# Phase 5 — Fix Notebook 05: Paper Artifacts

**Priority: P0 for scientific result generation**

## 5.1 Integrate Actual Results

Currently, benchmarks produce files such as:

```text
tier1_axis1_samplers.json
tier1_axis2_tree_ablation.json
tier1_scenario2_curriculum.json
```

Adapt the artifact generator to directly consume the actual formats produced by benchmarks, using explicit parsers and validations.

Do not fabricate an intermediate summary with arbitrary metrics.

## 5.2 Eliminate Silent Synthesis

In REAL:

- Prohibit fallback to synthetic distributions.
- Prohibit hardcoded metrics.
- Prohibit fabrication of missing seeds.
- Abort generation when a mandatory experiment is missing.
- Report the missing file or field.

In SMOKE:

- Allow generation of demonstrative figures from SMOKE results.
- Use dedicated, labeled synthetic files.
- Add visible smoke test watermarks/indicators to generated figures and tables, such as `SMOKE TEST — NOT FOR PUBLICATION`.
- Never write these results to final figure directories.

## 5.3 Paper Provenance

Every figure and table must be traceable to source files and their respective configurations.

Generate a manifest, for example:

```text
paper/artifact_manifest.json
```

The manifest must identify:

- Artifact name.
- Source files.
- Execution mode.
- Experiments and seeds included.
- Generation timestamp.
- Commit used.
- Input file hashes.

## 5.4 Fix Paths

Resolve all paths relative to the project root.

Eliminate dependency on Jupyter's current working directory for locating:

- `paper/main.tex`
- `paper_metadata.yaml`
- `results/`
- `paper/figures/`
- `paper/tables/`

## 5.5 LaTeX Compilation

Verify existence of `main.tex`, compilers, and dependencies.

In REAL, do not report success if compilation fails.

In SMOKE, permit technical tests of generation and compilation while keeping outputs isolated.

**Acceptance Criteria:**

- REAL figures derive exclusively from REAL results.
- No synthetic file enters the real paper.
- Missing files produce clear errors.
- Existing Tier-1 results are correctly ingested after migration.
- Every figure has auditable provenance.

---

# Phase 6 — Automated Testing

**Priority: P0/P1**

Create unit and integration tests, leveraging the existing testing framework in the repository.

## 6.1 Execution Mode Tests

Verify:

- REAL without dataset raises an exception.
- SMOKE creates only SMOKE data and artifacts.
- REAL rejects SMOKE cache.
- SMOKE rejects REAL cache.
- Mode does not change across cells or modules.
- Every SMOKE run produces an explicit warning.
- No silent fallback occurs.

## 6.2 Provenance Tests

Verify:

- Mandatory metadata.
- Dataset identity.
- Split consistency.
- Input hashes where applicable.
- Compatibility between tree, embeddings, and labels.
- Consistency between configuration and exported results.

## 6.3 Algorithm Tests

Implement targeted tests for:

- MCTS reward calculation.
- Node statistics updates.
- Selection and expansion.
- Hierarchy construction.
- Negative sampling strategies.
- Metric calculation.
- Serialization and recovery of results.

## 6.4 End-to-End Tests

Execute all notebooks in SMOKE mode in a controlled environment, preferably via `nbclient` or an equivalent tool.

Run integration tests for the REAL path using a small real fixture explicitly labeled as an integration test.

This reduced test must not be published as a full REAL experiment. Its configuration and artifacts must remain segregated from definitive scientific results.

Add a test confirming that notebook 05 rejects SMOKE results during REAL generation.

---

# Phase 7 — Documentation and Reproducibility

**Priority: P2**

Update the README with:

- Definition of SMOKE and REAL.
- Dataset requirements.
- Environment setup.
- Notebook execution order.
- Directory layout.
- How to run smoke tests.
- How to run real experiments.
- How to reproduce tables and figures.
- Result versioning rules.
- Known experimental limitations.

Create a reproducibility guide, preferably:

`docs/experimental_reproducibility.md`

The guide must identify completed experiments, pending experiments, and historical results lacking sufficient provenance.

---

# Mandatory Implementation Order

1. **Phase 0:** Audit and historical result preservation.
2. **Phase 1:** Global contract, modes, and isolation.
3. **Phase 2:** Notebooks 00, 01, and 02.
4. **Phase 3:** Notebook 03 and Tier-1 diagnostics.
5. **Phase 4:** Notebook 04 and real Tier-2.
6. **Phase 5:** Notebook 05 and artifact integration.
7. **Phase 6:** Full automated validation.
8. **Phase 7:** Documentation and reproduction instructions.

Tests must be added progressively, not solely in Phase 6.

Do not start definitive figure generation before completing real result integration and provenance validation.

# Definition of Done

The implementation is considered complete only when:

- [x] All six notebooks support both modes without mixing.
- [x] All notebooks clearly print the execution mode.
- [x] No silent synthetic fallback remains.
- [x] REAL and SMOKE directories are isolated.
- [x] Data, trees, caches, models, and results have consistent provenance.
- [x] Notebook 03 consumes real embeddings in REAL mode.
- [x] Notebook 04 consumes real images in REAL mode.
- [x] Notebook 05 uses actual results in REAL mode.
- [x] Missing data and results produce clear errors.
- [x] Historical experiments are preserved.
- [x] Unit and integration tests pass.
- [x] All six notebooks complete execution in SMOKE mode.
- [x] The REAL path is validated by integration tests with real data.
- [x] Reproducibility documentation is updated.

**Important:** Completing the implementation does not require running all full REAL benchmarks, but it requires demonstrating that the REAL execution path is correct. Full scientific runs represent a subsequent step with their own validation criteria.

# Agent Hand-off Instructions

Upon completing each phase, provide:

1. Modified files.
2. Description of changes.
3. Important architectural decisions.
4. Implemented and executed tests.
5. Test results.
6. Still-pending issues.
7. Potential incompatibilities with historical results.

Avoid changes unrelated to the roadmap.

Do not present hypothetical validations as executed tests.

Do not alter scientific results solely to make tests pass.

Preserve the ability to reproduce previous experiments and clearly distinguish historical results from results obtained following the fixes.
