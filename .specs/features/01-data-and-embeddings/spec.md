# F1: Project Scaffold, CIFAR-100 Loader & Embedding Cache Specification

**Approved**: 2026-11-08 - user: "prossiga"

## Problem Statement

Evaluating MCTS-based hard negative mining across multiple random seeds and hyperparameter configurations on raw images is computationally prohibitive without a standardized, deterministic feature cache and project environment. Furthermore, PyTorch's default `torchvision.datasets.CIFAR100` class only exposes the 100 fine-grained labels directly without providing a programmatic 20-superclass taxonomy lookup or deterministic One-vs-Rest imbalanced splitters. Feature `F1` establishes the `uv`-managed Python package (`pyproject.toml`), deterministic seed control (`src/utils/seed.py`), the CIFAR-100 20-superclass $\leftrightarrow$ 100-fine-class taxonomy (`src/data/cifar100_taxonomy.py`), a frozen ResNet-18 embedding extractor with disk cache verification (`src/data/extract_embeddings.py`), imbalanced Scenario 1 & Scenario 2 splitters (`src/data/splits.py`), and the versioned orchestration notebook (`notebooks/01_data_and_embeddings.ipynb`) for remote JupyterHub execution.

## Goals

- [ ] Configure a reproducible Python 3.10+ package (`pyproject.toml`) managed by `uv` with all scientific, deep learning, testing (`pytest`), and Jupyter (`ipykernel`, `nbformat`) dependencies (`AD-003`, `AD-008`).
- [ ] Enforce deterministic random seeding across Python `random`, `numpy`, and `torch` (CPU/CUDA/cuDNN) via `src/utils/seed.py`.
- [ ] Provide a zero-I/O-overhead CIFAR-100 taxonomy module (`src/data/cifar100_taxonomy.py`) mapping all 100 fine classes (`0..99`) to their 20 native superclasses (`0..19`) and human-readable names.
- [ ] Extract and cache deterministic `[N, 512]` feature embeddings from a frozen ImageNet-pretrained ResNet-18 backbone (`src/data/extract_embeddings.py`) across stratified train (`45,000`), validation (`5,000`), and test (`10,000`) splits with cache checksum/shape verification (`AD-001`).
- [ ] Generate deterministic index splits (`src/data/splits.py`) for Scenario 1 (Imbalanced Binary One-vs-Rest with positive subsample ratios $\rho \in \{0.10, 0.02, 0.01\}$ relative to per-class negative count, plus explicit target counts) and Scenario 2 (100-way Multi-class Curriculum) (`AD-004`).
- [ ] Deliver a clean, versioned Jupyter notebook (`notebooks/01_data_and_embeddings.ipynb`) that orchestrates CIFAR-100 download, embedding extraction, cache verification, and split visualization end-to-end on a remote JupyterHub (`AD-008`).

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
| ------- | ------ |
| Semantic and Ward Confusion tree construction (`src/trees/`) | Covered in Feature `F2` (`02-dual-hierarchy-trees`). |
| MCTS UCB1 traversal, momentum backpropagation, and Dual-Stream sampler (`src/mcts/`) | Covered in Feature `F3` (`03-mcts-hnm-agent`). |
| Downstream MLP/CNN training loops and OHEM baselines (`src/models/`, `src/training/`) | Covered in Features `F4` and `F5`. |
| End-to-end raw image data augmentation during CNN fine-tuning | Deferred to Tier-2 pipeline in Feature `F5` (`05-extensions-and-e2e`). |

---

## Assumptions & Open Questions

Every ambiguity is resolved or recorded here.

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| Stratified validation split from CIFAR-100 train set | 90% Train (`45,000` images, `450`/class) / 10% Validation (`5,000` images, `50`/class) stratified by fine label; `10,000` official Test set untouched | Provides a clean 5,000-sample validation split for baseline confusion matrix estimation in `F2` without leaking test data | y |
| ResNet-18 feature extraction preprocessing | Bilinear upsampling to $224 \times 224$ with standard ImageNet mean/std normalization before the frozen `ResNet18_Weights.IMAGENET1K_V1` trunk (`fc = Identity`) | Matches ResNet-18's pretrained receptive field and produces well-separated 512-d representations without modifying backbone architecture | y |
| Embedding storage format & cache reuse | Single PyTorch dictionary archive (`embeddings/cifar100_resnet18.pt`) storing `train`, `val`, `test` dicts (`embeddings: float32 [N, 512]`, `fine_labels: int64 [N]`, `coarse_labels: int64 [N]`, `metadata: dict`) | Fast single-file memory load on JupyterHub with built-in shape, dtype, and seed metadata validation | y |
| Scenario 1 positive class subsampling (`imbalance_ratio`) | For a target fine class $y^+$ (`450` train samples), subsample $N_{\text{pos}} = \max(1, \lfloor \text{pos\_fraction} \times 450 \rfloor)$ positives while retaining all $99 \times 450 = 44,550$ negative samples across the 99 negative classes (supporting `1:10` $\rightarrow 45$ pos, `1:50` $\rightarrow 9$ pos, `1:100` $\rightarrow 5$ pos per negative class size, or explicit `n_positives`) | Preserves all 99 negative classes in the MCTS search tree while controlling rare-positive scarcity deterministically | y |
| Default Scenario 1 target classes | `DEFAULT_SCENARIO1_TARGETS = ("beaver", "oak_tree", "pickup_truck", "rocket", "shark")` (fine IDs `4, 52, 58, 69, 73` across 5 distinct superclasses) | Settled in `00-epic-roadmap/spec.md` for diverse visual domains and reproducible cross-target benchmarks | y |
| Remaining implicit-requirement dimensions (auth, rate limits, TTL/expiry, concurrency) | N/A for local/JupyterHub offline dataset preparation and tensor caching | Single-process deterministic scientific pipeline with no network auth or multi-user concurrent state | y |

**Open questions:** none - all resolved and recorded above.

---

## User Stories

### P1: Project Environment, Seed Control & CIFAR-100 Taxonomy (`ISS-01`, `ISS-02a`) ⭐ MVP

**User Story**: As an ML engineer running experiments locally or on JupyterHub, I want a standardized `pyproject.toml`, a deterministic seed utility (`src/utils/seed.py`), and a verified CIFAR-100 taxonomy map (`src/data/cifar100_taxonomy.py`) so that all random operations and coarse-to-fine class lookups are 100% reproducible.

**Why P1**: Foundational package structure and taxonomy tables are required before any dataset loading, tree construction, or MCTS traversal can execute.

**Acceptance Criteria**:

1. The system SHALL define package metadata, Python $\ge 3.10$ compatibility, and dependencies (`torch`, `torchvision`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `ipykernel`, `nbformat`, `pytest`) in `pyproject.toml` with `src/` configured as the importable package root.
2. WHEN `set_global_seed(seed)` is called with a non-negative integer `seed` THEN the system SHALL seed Python `random`, `os.environ["PYTHONHASHSEED"]`, `numpy.random`, `torch.manual_seed`, and `torch.cuda.manual_seed_all`, and enable `torch.backends.cudnn.deterministic = True` with `torch.backends.cudnn.benchmark = False`.
3. IF `set_global_seed(seed)` is called with a negative integer or non-integer value THEN the system SHALL raise a `ValueError` describing the invalid seed.
4. WHEN `get_cifar100_taxonomy()` is called THEN the system SHALL return a taxonomy mapping where every fine class index in `0..99` maps to exactly one superclass index in `0..19`, and each of the 20 superclasses contains exactly 5 fine classes matching the official CIFAR-100 specification.
5. WHEN `fine_to_coarse_tensor(fine_labels)` is called with a 1D tensor or array of fine labels in `0..99` THEN the system SHALL return the corresponding coarse superclass labels in `0..19` with identical shape and device.

**Independent Test**: Run `pytest tests/test_seed_and_taxonomy.py` to verify that repeated random draws after `set_global_seed(42)` are bit-identical and that all 20 superclasses partition the 100 fine classes into 20 disjoint sets of 5 classes.

---

### P1: Frozen ResNet-18 Embedding Extractor & Cache Manager (`ISS-02b`) ⭐ MVP

**User Story**: As a researcher executing Tier-1 MCTS ablations, I want a programmatic embedding extractor (`src/data/extract_embeddings.py`) that downloads CIFAR-100 (or accepts a dataset/backbone hook), splits the training set into stratified train/val partitions, and caches `[N, 512]` feature tensors to disk so that 60,000 MCTS steps can run in minutes without redundant CNN forward passes.

**Why P1**: Tier-1 embedding caching (`AD-001`) is the primary enabler for multi-seed RL ablation throughput.

**Acceptance Criteria**:

1. WHEN `create_stratified_train_val_indices(labels, val_fraction=0.1, seed=42)` is called on the 50,000 CIFAR-100 training labels THEN the system SHALL return disjoint `train_indices` (`45,000` samples, `450` per fine class) and `val_indices` (`5,000` samples, `50` per fine class) without class distribution skew.
2. WHEN `extract_cifar100_embeddings(data_dir, cache_path, batch_size, seed, overwrite=False)` is called and `cache_path` does not exist THEN the system SHALL extract `float32` feature tensors of shape `[N, 512]` along with `int64` `fine_labels` and `coarse_labels` for `train`, `val`, and `test` splits and atomically save the bundle to `cache_path`.
3. WHILE a valid embedding bundle already exists at `cache_path` and `overwrite=False`, WHEN `extract_cifar100_embeddings()` or `load_embedding_cache(cache_path)` is called THEN the system SHALL validate the cached keys, tensor dimensions (`D == 512`), and absence of `NaN`/`Inf` values, and return the cached bundle without re-running feature extraction.
4. IF a cached embedding file at `cache_path` is corrupted, missing required split keys (`train`, `val`, `test`), or contains `NaN`/`Inf` entries THEN the system SHALL raise a `ValueError` identifying the failed integrity check.

**Independent Test**: Run `pytest tests/test_extract_embeddings.py` using a lightweight deterministic backbone/dataset fixture to verify stratified split counts, `[N, 512]` output shapes, cache hit reuse on second invocation, and `ValueError` on corrupted cache payloads.

---

### P1: Imbalanced Scenario Splitters & Notebook 01 Orchestration (`ISS-03`) ⭐ MVP

**User Story**: As an experimenter on a remote JupyterHub, I want deterministic Scenario 1 (One-vs-Rest imbalanced binary) and Scenario 2 (100-way multi-class) index splitters (`src/data/splits.py`) and a versioned notebook (`notebooks/01_data_and_embeddings.ipynb`) so that I can prepare and inspect all benchmark data splits with a single "Run All".

**Why P1**: Connects the cached feature tensors to the two experimental regimes required by `AD-004` and satisfies the notebook orchestration standard (`AD-008`).

**Acceptance Criteria**:

1. WHEN `build_scenario1_split(fine_labels, target_class, pos_fraction, seed)` is called for a valid `target_class` in `0..99` and `pos_fraction` in `(0, 1]` THEN the system SHALL return a `Scenario1Split` containing deterministically subsampled positive indices of size $\max(1, \lfloor \text{pos\_fraction} \times N_{\text{target}} \rfloor)$, binary target labels (`1` for `target_class`, `0` for all 99 negative classes), and a `negative_class_to_indices` dictionary mapping each of the 99 negative classes ($c \neq \text{target\_class}$) to its full list of sample indices.
2. IF `build_scenario1_split()` is called with `target_class` outside `0..99` or `pos_fraction <= 0` or `pos_fraction > 1` THEN the system SHALL raise a `ValueError`.
3. WHEN `build_scenario2_split(fine_labels)` is called THEN the system SHALL return a `Scenario2Split` mapping all 100 fine classes (`0..99`) to their corresponding sample index arrays with zero dropped classes.
4. WHEN `notebooks/01_data_and_embeddings.ipynb` is validated or executed THEN the notebook SHALL contain valid nbformat v4 structure importing exclusivamente from `src.utils.seed` and `src.data.*` (adhering to `AD-008`) to run CIFAR-100 download/extraction, verify cache shapes, build Scenario 1 (`1:10`, `1:50`, `1:100`) and Scenario 2 splits, and plot class/imbalance distributions.

**Independent Test**: Run `pytest tests/test_splits_and_notebook.py` to verify exact positive/negative index counts across $\rho \in \{1:10, 1:50, 1:100\}$, zero overlap between positive and negative index pools, and valid notebook structure/imports for `notebooks/01_data_and_embeddings.ipynb`.

---

## Edge Cases

- IF `fine_to_coarse_tensor()` receives a label outside `0..99` THEN the system SHALL raise a `ValueError` indicating out-of-range CIFAR-100 fine labels.
- IF `create_stratified_train_val_indices()` is given a `val_fraction` outside `(0.0, 1.0)` THEN the system SHALL raise a `ValueError`.
- WHEN `pos_fraction * N_target < 1.0` in `build_scenario1_split()` THEN the system SHALL clamp the number of positive instances to `1` so the positive training pool is never empty.
- IF `extract_cifar100_embeddings()` is interrupted or fails before saving completes THEN the system SHALL write to a temporary file first and rename atomically so a partial cache file is never left at `cache_path`.

---

## Requirement Traceability

Each requirement maps to a unique ID for tracking across design, tasks, and verification.

| Requirement ID | Story | Phase | Status |
| -------------- | ----- | ----- | ------ |
| DATA-01 | P1: Project Environment, Seed Control & CIFAR-100 Taxonomy (`ISS-01`, `ISS-02a`) | Execute | Verified |
| DATA-02 | P1: Frozen ResNet-18 Embedding Extractor & Cache Manager (`ISS-02b`) | Execute | Verified |
| DATA-03 | P1: Imbalanced Scenario Splitters & Notebook 01 Orchestration (`ISS-03`) | Execute | Verified |

**Coverage:** 3 total requirements mapped to `EPIC-01`.

---

## Differences from the source

Every departure from the original *GiantHunter* paper (`btaf239.pdf`) and official repository (`FuchuanQu/GiantHunter`) is linked to an active architectural decision in `.specs/STATE.md`.

| Source shows | Product shows | Confirmed |
| ------------ | ------------- | --------- |
| GiantHunter preprocesses viral contigs via Prodigal, DIAMOND, and MCL into Protein Cluster token frequencies | Extracts 512-d visual embeddings from CIFAR-100 via a frozen ImageNet-pretrained ResNet-18 backbone and caches them to disk | y - AD-001 |
| GiantHunter maps NCBI Taxonomy (Family $\rightarrow$ Genus $\rightarrow$ Species) | Maps CIFAR-100 taxonomy (20 coarse superclasses $\rightarrow$ 100 fine classes) | y - AD-002 |
| GiantHunter prepares only a single binary phage-vs-NCLDV dataset | Provides splitters for both Scenario 1 (One-vs-Rest imbalanced binary across 5 target classes and 3 imbalance ratios) and Scenario 2 (100-way multi-class curriculum) | y - AD-004 |
| GiantHunter executes preprocessing via shell `subprocess.run` inside a CLI script | Exposes modular Python APIs (`src/data/*`) orchestrated by a versioned Jupyter notebook (`notebooks/01_data_and_embeddings.ipynb`) for remote JupyterHub | y - AD-008 |

## Sources

- `.specs/features/00-epic-roadmap/spec.md` (`EPIC-01`) - parent epic specification and acceptance criteria for Feature `F1`.
- `ROADMAP.md` (`Feature F1`, `ISS-01`–`ISS-03`, `T1.1`–`T1.6`) - planned issue and task decomposition.
- `../btaf239.pdf` (*Qu et al., Bioinformatics 2025*) - reference for imbalanced binary training setup.

---

## Success Criteria

- [ ] `uv run pytest` executes all unit tests for `F1` (`tests/test_seed_and_taxonomy.py`, `tests/test_extract_embeddings.py`, `tests/test_splits_and_notebook.py`) with 100% pass rate.
- [ ] All 20 CIFAR-100 superclasses and 100 fine classes are bijectively verified and mapped in `src/data/cifar100_taxonomy.py`.
- [ ] Stratified train/val/test embedding extraction and cache validation run deterministically with `seed=42`.
- [ ] `notebooks/01_data_and_embeddings.ipynb` is versioned, valid, and invokes `src/` APIs cleanly for remote JupyterHub execution.
