# F6: Automated Paper Artifacts, Master Notebook & LaTeX Manuscript Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, the walk, check_proofs, the Verifier and the screens sub-agent).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Design**: `.specs/features/06-paper-and-artifacts/design.md`  
**Status**: Ready  
**Approved**: 2026-11-09 - user: "aprovado"  
**Build**: `.venv/bin/python -m compileall -q src tests`  
**Run unit**: `.venv/bin/pytest -v`  

---

## Test Coverage Matrix

> Generated from codebase, project guidelines, and spec - confirm before Execute. Guidelines found: `AGENTS.md`, `ROADMAP.md`, `ARCHITECTURE.md`.

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Metadata & Configuration (`src/utils/paper_artifacts.py`) | unit | `load_paper_metadata()` parsing `paper_metadata.yaml` with title, authors, PPGC UFRGS affiliation, and fallback handling | `tests/test_paper_artifacts.py` | `.venv/bin/pytest -v tests/test_paper_artifacts.py` |
| Figure Generator (`src/utils/paper_artifacts.py`) | unit | Convergence curves, tree visitation heatmaps, hyperparameter sensitivity plots exported as vector PDF and PNG | `tests/test_paper_artifacts.py` | `.venv/bin/pytest -v tests/test_paper_artifacts.py` |
| Table Generator (`src/utils/paper_artifacts.py`) | unit | Scenario 1, Scenario 2, and ablation study exported as clean LaTeX `booktabs` `.tex` tables with paired statistical test $p$-values | `tests/test_paper_artifacts.py` | `.venv/bin/pytest -v tests/test_paper_artifacts.py` |
| Results Loader & Synthesizer (`src/utils/paper_artifacts.py`) | unit | Dual-mode parsing of real `results/*.json` and deterministic synthetic 5-seed benchmark generation | `tests/test_paper_artifacts.py` | `.venv/bin/pytest -v tests/test_paper_artifacts.py` |
| LaTeX Template & Manuscript (`paper/`) | integration | `PRIMEarxiv.sty`, `references.bib`, and `main.tex` syntax validation, complete section coverage, and citation resolution | `tests/test_paper_artifacts.py` | `.venv/bin/pytest -v tests/test_paper_artifacts.py` |
| Manuscript Compilation (`paper/main.pdf`) | integration | Successful headless compilation via `pdflatex` producing valid `paper/main.pdf` | `tests/test_paper_artifacts.py` | `.venv/bin/pytest -v tests/test_paper_artifacts.py` |
| Notebook Orchestration (`notebooks/`) | integration | End-to-end execution of `05_paper_artifacts.ipynb` and `00_master_end_to_end.ipynb` via `nbclient` without errors | `tests/test_paper_artifacts.py` | `.venv/bin/pytest -v tests/test_paper_artifacts.py` |

## Gate Check Commands

> Generated from codebase - confirm before Execute.

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | After tasks with unit tests only | `.venv/bin/pytest -v` |
| Full | After tasks with integration tests | `.venv/bin/pytest -v` |
| Build | After phase completion or config/entity-only tasks | `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v` |

---

## Execution Plan

Phases are ordered and run sequentially - each phase completes before the next begins, and tasks within a phase execute in order.

### Phase 1: Metadata, Dual-Mode Results & Publication Plots (`ISS-15`, `AD-009`)

Create `paper_metadata.yaml`, implement metadata loader, dual-mode benchmark data loader/synthesizer, and vector PDF/PNG figure generators in `src/utils/paper_artifacts.py`.

```
T1 → T2
```

### Phase 2: LaTeX Tables & Artifact Orchestrator (`ISS-15`)

Implement LaTeX booktabs table generators (`tab_scenario1_imbalanced.tex`, `tab_scenario2_multiclass.tex`, `tab_ablation_study.tex`) and high-level `generate_all_paper_artifacts()` in `src/utils/paper_artifacts.py`.

```
T3
```

### Phase 3: LaTeX Template, Manuscript & PDF Compilation (`ISS-16`, `ISS-17`, `ISS-18`, `AD-007`)

Establish `paper/PRIMEarxiv.sty` and `paper/references.bib`, draft complete scientific manuscript `paper/main.tex`, and implement compilation test verifying `paper/main.pdf`.

```
T4 → T5
```

### Phase 4: Versioned Notebooks 05 & 00 Orchestration (`ISS-15`, `AD-008`)

Create versioned interactive artifact notebook `notebooks/05_paper_artifacts.ipynb` and master pipeline orchestrator `notebooks/00_master_end_to_end.ipynb` with `FAST_MODE` toggle.

```
T6
```

---

## Task Breakdown

### T1: Create `paper_metadata.yaml` & Implement Metadata Loader in `src/utils/paper_artifacts.py`

**What**: Create root `paper_metadata.yaml` with PPGC UFRGS metadata, implement `load_paper_metadata()` in `src/utils/paper_artifacts.py`, and create initial test file `tests/test_paper_artifacts.py`.
**Where**: `src/utils/paper_artifacts.py`
**Depends on**: None
**Reuses**: `src/utils/__init__.py`
**Requirement**: PAP-01

**Done when**:

- [x] `paper_metadata.yaml` exists at project root specifying title, authors, PPGC UFRGS affiliation, and keywords
- [x] `load_paper_metadata()` parses `paper_metadata.yaml` into a structured dictionary and falls back gracefully to PPGC UFRGS defaults if missing or invalid
- [x] Unit tests in `tests/test_paper_artifacts.py` verify metadata loading and fallback behavior
- [x] Gate check passes: `.venv/bin/pytest -v`

**Tests**: unit
**Gate**: quick
**Commit**: `feat(paper): create root paper metadata and loader in paper_artifacts`

---

### T2: Implement Dual-Mode Results Ingestion & Publication Plots in `src/utils/paper_artifacts.py`

**What**: Implement `load_or_synthesize_benchmark_results()`, `generate_convergence_plot()`, `generate_tree_visitation_heatmap()`, and `generate_hyperparameter_sensitivity_plot()` in `src/utils/paper_artifacts.py` exporting vector PDF and PNG figures to `paper/figures/`.
**Where**: `src/utils/paper_artifacts.py`
**Depends on**: T1
**Reuses**: `src/utils/seed.py`
**Requirement**: PAP-01

**Done when**:

- [x] `load_or_synthesize_benchmark_results()` parses real `results/*.json` or deterministically synthesizes complete 5-seed benchmark metric distributions matching Scenario 1, Scenario 2, and ablation schemas
- [x] `generate_convergence_plot()` exports `fig_convergence_curves.pdf` and `.png` displaying F1/AUPRC and loss convergence across training steps
- [x] `generate_tree_visitation_heatmap()` exports `fig_tree_visitation_heatmap.pdf` and `.png` illustrating UCB1 exploration-to-exploitation transitions across superclasses
- [x] `generate_hyperparameter_sensitivity_plot()` exports `fig_hyperparameter_sensitivity.pdf` and `.png` mapping F1/AUPRC across exploration constant $c$ and momentum $\alpha$
- [x] Unit tests verify file creation, PDF/PNG formats, non-empty file sizes, and deterministic generation
- [x] Gate check passes: `.venv/bin/pytest -v`

**Tests**: unit
**Gate**: quick
**Commit**: `feat(paper): implement dual-mode benchmark data loader and publication figure generator`

---

### T3: Implement LaTeX Booktabs Table Generator & Artifact Pipeline in `src/utils/paper_artifacts.py`

**What**: Implement `generate_scenario1_table()`, `generate_scenario2_table()`, `generate_ablation_table()`, and `generate_all_paper_artifacts()` in `src/utils/paper_artifacts.py`.
**Where**: `src/utils/paper_artifacts.py`
**Depends on**: T2
**Reuses**: `src/utils/statistics.py`
**Requirement**: PAP-01

**Done when**:

- [x] `generate_scenario1_table()` exports `tab_scenario1_imbalanced.tex` with clean `booktabs` format, mean $\pm$ std metrics, and paired test $p$-values
- [x] `generate_scenario2_table()` exports `tab_scenario2_multiclass.tex` for 100-way multi-class curriculum evaluation
- [x] `generate_ablation_table()` exports `tab_ablation_study.tex` covering tree geometry, $Q$-normalization, and dynamic expansion
- [x] `generate_all_paper_artifacts()` orchestrates all figures and tables into `paper/figures/` and `paper/tables/`, creating directories automatically if absent
- [x] Unit tests verify valid LaTeX table syntax (`\begin{tabular}`, `\toprule`, `\bottomrule`) and correct file generation
- [x] Gate check passes: `.venv/bin/pytest -v`

**Tests**: unit
**Gate**: quick
**Commit**: `feat(paper): implement latex booktabs table generator and paper artifact pipeline`

---

### T4: Set Up LaTeX Preprint Template & Draft Scientific Manuscript in `paper/`

**What**: Set up `paper/PRIMEarxiv.sty`, `paper/references.bib`, and draft complete scientific manuscript in `paper/main.tex` in English, incorporating generated figures and tables.
**Where**: `paper/main.tex`
**Depends on**: T3
**Reuses**: `paper_metadata.yaml`
**Requirement**: PAP-02

**Done when**:

- [x] `paper/PRIMEarxiv.sty` provides arXiv / Prime AI preprint styling
- [x] `paper/references.bib` contains BibTeX entries for GiantHunter, Sutton & Barto, UCB1, OHEM, ResNet, CIFAR-100, and Ward clustering
- [x] `paper/main.tex` contains complete manuscript sections: `Abstract`, `Introduction`, `Related Work`, `Methodology`, `Experimental Setup`, `Results`, `Discussion & Limitations`, and `Conclusion`, including tables and figures
- [x] Integration test verifies that every `\cite{...}`, `\input{tables/...}`, and `\includegraphics{figures/...}` in `main.tex` resolves to existing files/entries
- [x] Gate check passes: `.venv/bin/pytest -v`

**Tests**: integration
**Gate**: full
**Commit**: `feat(paper): create latex template, bibliography, and complete scientific manuscript`

---

### T5: Implement Manuscript Compilation & Compilation Verification in `src/utils/paper_artifacts.py`

**What**: Implement `compile_paper()` in `src/utils/paper_artifacts.py` and integration tests compiling `paper/main.pdf` via `pdflatex` with 0 fatal errors.
**Where**: `src/utils/paper_artifacts.py`
**Depends on**: T4
**Reuses**: `paper/main.tex`
**Requirement**: PAP-04

**Done when**:

- [x] `compile_paper(paper_dir="paper")` executes `pdflatex` and `bibtex` headless inside `paper/`
- [x] `paper/main.pdf` is successfully generated with exit code 0
- [x] Integration test verifies `main.pdf` file presence and non-empty byte size
- [x] Gate check passes: `.venv/bin/pytest -v`

**Tests**: integration
**Gate**: full
**Commit**: `feat(paper): implement manuscript compilation and pdf verification`

---

### T6: Deliver Versioned Notebooks 05 and 00 for Interactive Artifacts & Master Orchestration

**What**: Create versioned interactive notebook `notebooks/05_paper_artifacts.ipynb` and master pipeline notebook `notebooks/00_master_end_to_end.ipynb` with `FAST_MODE` toggle.
**Where**: `notebooks/00_master_end_to_end.ipynb`
**Depends on**: T5
**Reuses**: `src/data/`, `src/trees/`, `src/mcts/`, `src/training/`, `src/utils/`
**Requirement**: PAP-03

**Done when**:

- [x] `notebooks/05_paper_artifacts.ipynb` executes `generate_all_paper_artifacts()` and renders inline previews of all publication figures and LaTeX tables
- [x] `notebooks/00_master_end_to_end.ipynb` provides top-to-bottom master orchestration (CIFAR-100 download check $\rightarrow$ embeddings $\rightarrow$ trees $\rightarrow$ Tier-1 smoke benchmark $\rightarrow$ Tier-2 E2E smoke train $\rightarrow$ paper artifacts) running under 2 minutes in `FAST_MODE = True`
- [x] Integration test executes both notebooks via `nbclient` confirming error-free execution
- [x] Gate check passes: `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v`

**Tests**: integration
**Gate**: build
**Commit**: `feat(notebooks): create versioned paper artifacts notebook 05 and master orchestrator notebook 00`

---

## Checks

**C1** - `load_paper_metadata()` parses `paper_metadata.yaml` with title, authors, and PPGC UFRGS affiliation, falling back gracefully to defaults if missing (PAP-01, AC 1)  
Proof: run unit | tests/test_paper_artifacts.py::test_load_paper_metadata PASSED

**C2** - `generate_convergence_plot()`, `generate_tree_visitation_heatmap()`, and `generate_hyperparameter_sensitivity_plot()` export vector PDF and PNG figures under `paper/figures/` (PAP-01, AC 2)  
Proof: run unit | tests/test_paper_artifacts.py::test_publication_figures_generation PASSED

**C3** - `generate_scenario1_table()`, `generate_scenario2_table()`, and `generate_ablation_table()` export formatted LaTeX `booktabs` tables with statistical test $p$-values under `paper/tables/` (PAP-01, AC 3)  
Proof: run unit | tests/test_paper_artifacts.py::test_latex_tables_generation PASSED

**C4** - `load_or_synthesize_benchmark_results()` parses real `results/*.json` or deterministically synthesizes complete 5-seed benchmark metric distributions (PAP-01, AC 4)  
Proof: run unit | tests/test_paper_artifacts.py::test_benchmark_results_loader_and_synthesizer PASSED

**C5** - `generate_all_paper_artifacts()` creates `paper/figures/` and `paper/tables/` directories automatically if absent and populates all artifact files (PAP-01, AC 5)  
Proof: run unit | tests/test_paper_artifacts.py::test_generate_all_paper_artifacts_directory_creation PASSED

**C6** - `paper/PRIMEarxiv.sty`, `paper/references.bib`, and `paper/main.tex` contain all academic sections and all `\cite`, `\input`, and `\includegraphics` commands resolve to valid files and entries (PAP-02, AC 1–3)  
Proof: run unit | tests/test_paper_artifacts.py::test_manuscript_syntax_and_citations PASSED

**C7** - `notebooks/05_paper_artifacts.ipynb` executes end-to-end without errors and renders inline figure and table previews (PAP-03, AC 1)  
Proof: run unit | tests/test_paper_artifacts.py::test_notebook_05_execution PASSED

**C8** - `notebooks/00_master_end_to_end.ipynb` executes top-to-bottom in `FAST_MODE = True` without errors in under 2 minutes (PAP-03, AC 2)  
Proof: run unit | tests/test_paper_artifacts.py::test_notebook_00_execution PASSED

**C9** - `compile_paper()` compiles `paper/main.tex` into a valid `paper/main.pdf` document with exit code 0 (PAP-04, AC 1–2)  
Proof: run unit | tests/test_paper_artifacts.py::test_compile_paper_pdf PASSED

---

## Handoff

The feature's live state, written as it happens - a compaction gives no warning.

**Settled mid-build:** none  
**Dispatched:** none  
**Walked:** none  

---

## Phase Execution Map

Visual representation of task ordering. Phases run in sequence, and tasks within a phase run in order:

```
Phase 1 → Phase 2 → Phase 3 → Phase 4

T1 → T2 → T3 → T4 → T5 → T6
```

---

## Task Granularity Check

| Task | Scope | Status |
| ---- | ----- | ------ |
| T1: Create `paper_metadata.yaml` & Implement Metadata Loader in `src/utils/paper_artifacts.py` | 1 config file + 1 module (`paper_metadata.yaml`, `src/utils/paper_artifacts.py`) | ✅ Granular |
| T2: Implement Dual-Mode Results Ingestion & Publication Plots in `src/utils/paper_artifacts.py` | 1 module (`src/utils/paper_artifacts.py`) | ✅ Granular |
| T3: Implement LaTeX Booktabs Table Generator & Artifact Pipeline in `src/utils/paper_artifacts.py` | 1 module (`src/utils/paper_artifacts.py`) | ✅ Granular |
| T4: Set Up LaTeX Preprint Template & Draft Scientific Manuscript in `paper/` | 1 template + 1 bib + 1 tex file (`paper/`) | ✅ Granular |
| T5: Implement Manuscript Compilation & Compilation Verification in `src/utils/paper_artifacts.py` | 1 compilation function + 1 test (`src/utils/paper_artifacts.py`) | ✅ Granular |
| T6: Deliver Versioned Notebooks 05 and 00 for Interactive Artifacts & Master Orchestration | 2 notebooks (`notebooks/05_*.ipynb`, `notebooks/00_*.ipynb`) | ✅ Granular |

---

## Diagram-Definition Cross-Check

| Task | Depends On (task body) | Diagram Shows | Status |
| ---- | ---------------------- | ------------- | ------ |
| T1 | None | Phase 1 start (`T1`) | ✅ Match |
| T2 | T1 | `T1 → T2` | ✅ Match |
| T3 | T2 | Phase 1 (`T2`) → Phase 2 (`T3`) | ✅ Match |
| T4 | T3 | Phase 2 (`T3`) → Phase 3 (`T4`) | ✅ Match |
| T5 | T4 | `T4 → T5` | ✅ Match |
| T6 | T5 | Phase 3 (`T5`) → Phase 4 (`T6`) | ✅ Match |

---

## Test Co-location Validation

| Task | Code Layer Created/Modified | Matrix Requires | Task Says | Status |
| ---- | --------------------------- | --------------- | --------- | ------ |
| T1 | Metadata & Configuration (`src/utils/paper_artifacts.py`) | unit | unit | ✅ OK |
| T2 | Figure Generator (`src/utils/paper_artifacts.py`) | unit | unit | ✅ OK |
| T3 | Table Generator (`src/utils/paper_artifacts.py`) | unit | unit | ✅ OK |
| T4 | LaTeX Template & Manuscript (`paper/`) | integration | integration | ✅ OK |
| T5 | Manuscript Compilation (`paper/main.pdf`) | integration | integration | ✅ OK |
| T6 | Notebook Orchestration (`notebooks/`) | integration | integration | ✅ OK |
