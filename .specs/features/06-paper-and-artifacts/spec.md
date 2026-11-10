# F6: Automated Paper Artifacts, Master Notebook & LaTeX Manuscript Specification

**Approved**: 2026-11-09 - user: "aprovado"

## Problem Statement

The P1 MVP and P2 algorithmic extensions (`F1`–`F5`) establish modular components for CIFAR-100 embedding extraction, dual hierarchy tree construction, MCTS hard negative mining, multi-seed Tier-1 benchmarking, dynamic entropy leaf expansion, and Tier-2 end-to-end CNN training (`AD-001`–`AD-006`, `AD-008`). However, disseminating these findings as an academic deliverable requires bridging experimental results with publication-grade figures, statistical tables, versioned notebook orchestration, and a complete scientific manuscript (`AD-007`). Feature `F6` delivers an automated artifact generation pipeline (`src/utils/paper_artifacts.py`), interactive visualization and master execution notebooks (`notebooks/05_paper_artifacts.ipynb` and `notebooks/00_master_end_to_end.ipynb`), and a complete LaTeX manuscript in the arXiv/Prime AI template (`paper/main.tex`, `paper/PRIMEarxiv.sty`, `paper/references.bib`) configured dynamically via a root `paper_metadata.yaml` file (`AD-009`) and compiled cleanly to `paper/main.pdf`.

## Goals

- [ ] Implement `src/utils/paper_artifacts.py` supporting `paper_metadata.yaml` loading, vector figure generation (convergence curves, tree visitation heatmaps, hyperparameter sensitivity plots in `paper/figures/`), and LaTeX booktabs table generation (Scenario 1, Scenario 2, and ablation summaries in `paper/tables/`) with dual-mode support for both real `results/*.json` and deterministic fallback data (`ISS-15`, `AD-007`, `AD-009`).
- [ ] Deliver versioned notebook `notebooks/05_paper_artifacts.ipynb` rendering inline previews of all publication figures and LaTeX tables while writing vector PDF and `.tex` artifacts to disk (`ISS-15`, `AD-008`).
- [ ] Deliver versioned master notebook `notebooks/00_master_end_to_end.ipynb` providing a single-click top-to-bottom pipeline (data download $\rightarrow$ embeddings $\rightarrow$ trees $\rightarrow$ benchmarks $\rightarrow$ paper artifacts) with a `FAST_MODE = True` default smoke-test toggle (`ISS-15`, `AD-008`).
- [ ] Establish the complete LaTeX paper template in `paper/` (`main.tex`, `PRIMEarxiv.sty`, `references.bib`) with full scientific sections (`Abstract`, `Introduction`, `Related Work`, `Methodology`, `Experimental Setup`, `Results`, `Discussion & Limitations`, `Conclusion`) dynamically populated with metadata from `paper_metadata.yaml` (PPGC UFRGS) (`ISS-16`, `ISS-17`, `AD-009`).
- [ ] Verify clean compilation of `paper/main.pdf` via `pdflatex` with zero broken references, missing citations, or unresolved figure links (`ISS-18`).

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
| ------- | ------ |
| Interactive web dashboards or Streamlit portals | All visualizations must be publication-grade static PDF/PNG figures and LaTeX booktabs tables (`AD-007`). |
| Proprietary or cloud-only Overleaf compilation APIs | All compilation is local and headless via standard `pdflatex` / `bibtex` (`AD-007`). |
| External bioinformatics pipelines or metagenomic contig evaluations | Excluded by epic design; experimental validation focuses on CIFAR-100 hierarchical classification (`AD-001`, `AD-002`). |

---

## Assumptions & Open Questions

Every ambiguity is resolved or recorded here.

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| Root paper metadata configuration (`ISS-16`, `AD-009`) | Maintain `paper_metadata.yaml` at project root storing `title`, list of `authors` (with names, affiliations, emails), `institution` ("PPGC - Universidade Federal do Rio Grande do Sul (UFRGS)"), and `keywords`; read by `src/utils/paper_artifacts.py` | Decided by user: allows multiple authors to be updated subsequently without editing LaTeX boilerplates directly | y |
| Dual-mode experimental data ingestion (`ISS-15`) | `generate_all_paper_artifacts()` checks `results/` for benchmark JSON files; if present, it parses actual multi-seed logs; if absent or partial, it generates deterministic synthetic benchmark data matching the 5-seed schema | Allows immediate local publication artifact generation and test verification even before multi-hour remote JupyterHub runs finish | y |
| Master notebook execution mode (`ISS-15`, `AD-008`) | `notebooks/00_master_end_to_end.ipynb` includes `FAST_MODE = True` toggle by default executing small-scale smoke steps in ~1 minute, with a one-line switch to `FAST_MODE = False` for full remote GPU execution | Ensures seamless testability in CI and quick local validation while preserving full experiment reproducibility | y |
| Publication figure formats & styling (`ISS-15`) | Export vector PDF (`.pdf`) for LaTeX compilation and raster PNG (`.png`, 300 DPI) for inline notebook previews and README embedding, using Matplotlib/Seaborn with publication fonts and colorblind-safe palettes | Standard academic publishing practice for high-resolution vector printing and digital rendering | y |
| LaTeX manuscript template & style (`ISS-16`, `AD-007`) | Use `PRIMEarxiv.sty` (arXiv / Prime AI preprint style) with standard packages (`amsmath`, `amssymb`, `booktabs`, `graphicx`, `microtype`, `hyperref`, `natbib`) | Matches standard arXiv preprint conventions and requested course deliverable | y |
| Scientific bibliography & citations (`ISS-16`) | Comprehensive BibTeX database in `paper/references.bib` citing GiantHunter (Qu et al., 2025), Sutton & Barto (2018), Auer et al. (UCB1, 2002), Shrivastava et al. (OHEM, 2016), He et al. (ResNet, 2016), Krizhevsky (CIFAR-100, 2009), Ward (1963), and related RL/curriculum works | Ensures academic rigor with complete, valid citations for every method compared | y |
| Remaining implicit-requirement dimensions (auth, rate limits, TTL/expiry, external network calls, concurrency) | N/A for local scientific artifact generation, Jupyter notebook execution, and LaTeX compilation | Self-contained local tools with deterministic filesystem persistence | y |

**Open questions:** none - all resolved and recorded above.

---

## User Stories

### P1: Automated Figure & Table Generator in `src/utils/paper_artifacts.py` (`ISS-15`, `AD-008`, `AD-009`) ⭐ MVP

**User Story**: As a master's student preparing the final seminar deliverable, I want a Python module (`src/utils/paper_artifacts.py`) that loads `paper_metadata.yaml`, aggregates experimental benchmark metrics, and automatically generates publication-quality PDF plots and LaTeX booktabs tables into `paper/figures/` and `paper/tables/` so that all numerical results and graphics in the manuscript are 100% reproducible from code.

**Why P1**: Solves the error-prone task of manually transcribing metrics and creating figures, establishing direct automated traceability from benchmark runs to LaTeX tables and figures (`AD-007`).

**Acceptance Criteria**:

1. WHEN `load_paper_metadata()` is called THEN the system SHALL load and parse `paper_metadata.yaml` from the project root (falling back to default PPGC UFRGS metadata if missing) and return a structured dictionary containing `title`, `authors`, `institution`, and `keywords`.
2. WHEN `generate_all_paper_artifacts()` is executed with an output directory `paper_dir` THEN the system SHALL generate and save vector PDF and PNG plots under `paper/figures/`: (a) `fig_convergence_curves.pdf`, (b) `fig_tree_visitation_heatmap.pdf`, and (c) `fig_hyperparameter_sensitivity.pdf`.
3. WHEN `generate_all_paper_artifacts()` is executed THEN the system SHALL generate and save formatted LaTeX `.tex` booktabs tables under `paper/tables/`: (a) `tab_scenario1_imbalanced.tex` (One-vs-Rest comparing Uniform, OHEM, Strict GiantHunter, and Enhanced MCTS-HNM across imbalance ratios with paired test $p$-values), (b) `tab_scenario2_multiclass.tex` (100-way multi-class curriculum), and (c) `tab_ablation_study.tex` (tree geometry and MCTS equation ablations).
4. WHERE experimental JSON logs in `results/` are absent or incomplete, the system SHALL synthesize deterministic, schema-compliant benchmark results so that artifact generation runs without errors.
5. IF `paper_dir` does not exist THEN the system SHALL automatically create `paper_dir`, `paper_dir/figures`, and `paper_dir/tables`.

**Independent Test**: Run `.venv/bin/pytest -v tests/test_paper_artifacts.py -k "artifacts or metadata"` to verify metadata loading, PDF/PNG figure export, and valid LaTeX booktabs table syntax.

---

### P1: LaTeX Template & Scientific Manuscript Drafting (`ISS-16`, `ISS-17`, `AD-007`, `AD-009`) ⭐ MVP

**User Story**: As an academic reviewer reading the project submission, I want a complete, rigorous scientific manuscript in `paper/main.tex` structured with `Abstract`, `Introduction`, `Related Work`, `Methodology`, `Experimental Setup`, `Results`, `Discussion & Limitations`, and `Conclusion` in the arXiv / Prime AI template citing all relevant literature so that the research contributions of MCTS-HNM are fully articulated and substantiated.

**Why P1**: Represents the required scientific paper deliverable of the master's reinforcement learning seminar (`AD-007`).

**Acceptance Criteria**:

1. The system SHALL provide `paper/PRIMEarxiv.sty`, `paper/references.bib`, and `paper/main.tex` with title, authors, and affiliations synchronized with `paper_metadata.yaml`.
2. The manuscript in `paper/main.tex` SHALL contain all standard academic sections: `Abstract`, `Introduction`, `Related Work`, `Methodology`, `Experimental Setup`, `Results`, `Discussion & Limitations`, and `Conclusion`, incorporating `\input{tables/...}` and `\includegraphics{figures/...}` without hardcoded broken references.
3. Every citation in `paper/main.tex` SHALL resolve to a valid entry in `paper/references.bib` (including GiantHunter, Sutton & Barto, Auer UCB1, Shrivastava OHEM, He ResNet, Krizhevsky CIFAR-100, and Ward clustering).

**Independent Test**: Run `.venv/bin/pytest -v tests/test_paper_artifacts.py -k "manuscript_syntax"` to parse `paper/main.tex` and verify section presence, citation resolution, and table/figure file linkages.

---

### P2: Versioned Notebooks 05 and 00 for Artifacts & Master Orchestration (`ISS-15`, `AD-008`)

**User Story**: As a researcher running on a remote JupyterHub, I want `notebooks/05_paper_artifacts.ipynb` for interactive plot/table inspection and `notebooks/00_master_end_to_end.ipynb` for single-click execution of the entire project pipeline (with a `FAST_MODE` toggle) so that the entire project can be run top-to-bottom from a Jupyter interface.

**Why P2**: Fulfills the JupyterHub remote orchestration requirement (`AD-008`) for paper generation and end-to-end reproducibility.

**Acceptance Criteria**:

1. WHEN `notebooks/05_paper_artifacts.ipynb` is executed THEN the notebook SHALL call `generate_all_paper_artifacts()`, render inline Matplotlib figures and LaTeX table previews, and confirm that all artifact files exist on disk.
2. WHEN `notebooks/00_master_end_to_end.ipynb` is executed with `FAST_MODE = True` THEN the notebook SHALL execute all stages (setup & data check $\rightarrow$ tree construction $\rightarrow$ Tier-1 smoke benchmark $\rightarrow$ Tier-2 E2E smoke train $\rightarrow$ paper artifact generation) within 2 minutes with zero errors.

**Independent Test**: Run `.venv/bin/pytest -v tests/test_paper_artifacts.py -k "notebook"` to execute `notebooks/05_paper_artifacts.ipynb` and `notebooks/00_master_end_to_end.ipynb` via `nbclient` and confirm error-free execution.

---

### P2: Manuscript Compilation & Verification (`ISS-18`)

**User Story**: As a student submitting the paper PDF, I want `paper/main.pdf` compiled cleanly via `pdflatex` / `latexmk` so that the final PDF is ready for submission and archive upload.

**Why P2**: Validates that the LaTeX source compiles without errors, fatal warnings, or missing dependencies (`ISS-18`).

**Acceptance Criteria**:

1. WHEN `pdflatex` or `latexmk` is executed on `paper/main.tex` THEN the system SHALL produce a valid `paper/main.pdf` document.
2. The compilation SHALL exit with return code 0 and contain zero LaTeX fatal errors or missing bibliography entries.

**Independent Test**: Run `pdflatex -interaction=nonstopmode main.tex` inside `paper/` and verify that `paper/main.pdf` is created and `test_paper_artifacts.py` passes.

---

## Edge Cases

- IF `results/` contains no benchmark JSON files THEN `generate_all_paper_artifacts()` SHALL automatically use synthetic 5-seed benchmark distributions and log an informative warning rather than crashing.
- IF `paper_metadata.yaml` is missing or invalid YAML THEN `load_paper_metadata()` SHALL fall back to hardcoded default values (PPGC UFRGS) and log a warning.
- IF a figure title or LaTeX table column name contains special characters (`_`, `%`, `&`) THEN `src/utils/paper_artifacts.py` SHALL properly escape them in generated `.tex` files.
- IF `FAST_MODE` is disabled in `notebooks/00_master_end_to_end.ipynb`, the notebook SHALL execute full training and benchmark loops using user-specified GPU/CPU devices.

---

## Requirement Traceability

Each requirement maps to a unique ID for tracking across design, tasks, and verification.

| Requirement ID | Story | Phase | Status |
| -------------- | ----- | ----- | ------ |
| PAP-01 | P1: Automated Figure & Table Generator in `src/utils/paper_artifacts.py` (`ISS-15`) | Specify | Pending |
| PAP-02 | P1: LaTeX Template & Scientific Manuscript Drafting (`ISS-16`, `ISS-17`) | Specify | Pending |
| PAP-03 | P2: Versioned Notebooks 05 and 00 for Artifacts & Master Orchestration (`ISS-15`) | Specify | Pending |
| PAP-04 | P2: Manuscript Compilation & Verification (`ISS-18`) | Specify | Pending |

**Coverage:** 4 total requirements mapped to `EPIC-06`.

---

## Differences from the source

Every departure from the original *GiantHunter* paper (`btaf239.pdf`) is linked to an active architectural decision in `.specs/STATE.md`.

| Source shows | Product shows | Confirmed |
| ------------ | ------------- | --------- |
| GiantHunter only provides manual plotting code and no automated LaTeX table/figure exporter | Provides modular `src/utils/paper_artifacts.py` generating booktabs tables and vector PDF figures directly into `paper/` | y - AD-007 |
| GiantHunter does not provide Jupyter orchestration notebooks | Provides `notebooks/05_paper_artifacts.ipynb` and `notebooks/00_master_end_to_end.ipynb` for single-click remote JupyterHub execution | y - AD-008 |
| GiantHunter has fixed hardcoded author and title strings | Configures paper title, authors, affiliation (PPGC UFRGS), and metadata via root `paper_metadata.yaml` | y - AD-009 |

## Sources

- `.specs/features/00-epic-roadmap/spec.md` (`EPIC-06`) - parent epic specification for paper artifacts and manuscript.
- `ROADMAP.md` (`Feature F6`, `ISS-15`–`ISS-18`, `T6.1`–`T6.5`) - roadmap issues and tasks.
- `ARCHITECTURE.md` (`Section 5: Two-Tier Execution & Notebook Orchestration`) - notebook orchestration guidelines.
- `.specs/STATE.md` (`AD-001` through `AD-009`) - active architectural decisions governing project structure and metadata.

---

## Success Criteria

- [ ] `.venv/bin/pytest -v tests/test_paper_artifacts.py` passes 100% of unit and integration checks covering `PAP-01`, `PAP-02`, `PAP-03`, and `PAP-04`.
- [ ] `paper/main.pdf` is compiled and verified with zero missing citations or broken table/figure linkages.
