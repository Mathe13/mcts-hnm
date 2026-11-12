# F27: P2 — Expansão Analítica: Curvas Precision-Recall e Análise em 4 Quadrantes Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, the walk, check_proofs, the Verifier and the screens sub-agent).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Design**: `.specs/features/27-p2-expansao-figuras-pr-e-4quadrantes/design.md`  
**Status**: Ready  
**Approved**: 2026-11-12 - user: "prosseguir"  
**Build**: `.venv/bin/python -m compileall -q src tests`  
**Run unit**: `.venv/bin/pytest -v tests/test_p2_pr_and_quadrant_figures.py`  

---

## Test Coverage Matrix

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Precision-Recall Curves Generator (`src/utils/paper_artifacts.py`) | unit | 100-point uniform interpolation, +/- 1 std variance bands, tau_val marker, dual export | `tests/test_p2_pr_and_quadrant_figures.py` | `.venv/bin/pytest -v tests/test_p2_pr_and_quadrant_figures.py` |
| 4-Quadrant Plot Generator (`src/utils/paper_artifacts.py`) | unit | Grouped bar layout for Q1-Q4 across methods, error bars, dual export | `tests/test_p2_pr_and_quadrant_figures.py` | `.venv/bin/pytest -v tests/test_p2_pr_and_quadrant_figures.py` |
| Publication Pipeline Orchestration (`src/utils/paper_artifacts.py`) | integration | `generate_all_paper_artifacts()` integration, manifest presence, mode isolation | `tests/test_p2_pr_and_quadrant_figures.py` | `.venv/bin/pytest -v tests/test_p2_pr_and_quadrant_figures.py` |
| Notebook 05 Integration (`notebooks/05_paper_artifacts.ipynb`) | integration | Execution and visualization cells for FIG-06 and FIG-07 in SMOKE mode | `tests/test_p2_pr_and_quadrant_figures.py` | `.venv/bin/pytest -v tests/test_p2_pr_and_quadrant_figures.py -k "test_notebook_05_integration_p2"` |
| LaTeX Manuscript Integration (`paper/main.tex`) | integration | Environments, labels `fig:precision_recall_curves`, `fig:evaluation_4quadrants` | `tests/test_p2_pr_and_quadrant_figures.py` | `.venv/bin/pytest -v tests/test_p2_pr_and_quadrant_figures.py -k "test_latex_figure_blocks_p2"` |

---

## Gate Check Commands

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | After tasks with unit tests only | `.venv/bin/pytest -v tests/test_p2_pr_and_quadrant_figures.py` |
| Full | After tasks with integration tests | `.venv/bin/pytest -v tests/test_p2_pr_and_quadrant_figures.py tests/test_paper_artifacts.py` |
| Build | After phase completion or config/entity-only tasks | `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_p2_pr_and_quadrant_figures.py` |

---

## Execution Plan

Phases are ordered and run sequentially - each phase completes before the next begins, and tasks within a phase execute in order.

### Phase 1: Implementação dos Geradores de Figuras da Fase 3 (FIG-06, FIG-07)

Implementa `generate_precision_recall_curves_plot()` e `generate_quadrant_evaluation_plot()` em `src/utils/paper_artifacts.py` com exportação dupla (.pdf e .png 300 DPI) e isolamento SMOKE/REAL.

```
T1
```

### Phase 2: Suíte Automatizada de Testes de Figuras P2 (FIG-06, FIG-07, INT-02)

Cria `tests/test_p2_pr_and_quadrant_figures.py` cobrindo interpolação contínua de curvas PR, agrupamento em 4 quadrantes, integridade dos formatos e modo SMOKE.

```
T2
```

### Phase 3: Integração no Orquestrador Global de Artefatos (INT-02)

Conecta `generate_precision_recall_curves_plot` e `generate_quadrant_evaluation_plot` em `generate_all_paper_artifacts()` em `src/utils/paper_artifacts.py`.

```
T3
```

### Phase 4: Integração Interativa no Notebook 05 (INT-02)

Atualiza `notebooks/05_paper_artifacts.ipynb` para renderizar FIG-06 e FIG-07 no fluxo interativo do notebook.

```
T4
```

### Phase 5: Incorporação das Figuras no Manuscrito LaTeX (INT-02)

Atualiza `paper/main.tex` adicionando os ambientes `\begin{figure}` para FIG-06 e FIG-07 com legendas e labels acadêmicos.

```
T5
```

---

## Task Breakdown

### T1: Implementar geradores de FIG-06 e FIG-07 em paper_artifacts.py

**What**: Implementar `generate_precision_recall_curves_plot()` e `generate_quadrant_evaluation_plot()` em `src/utils/paper_artifacts.py` com interpolação contínua em grade de recall, agrupamento nos 4 quadrantes de avaliação, bandas de variância, exportação dupla em .pdf e .png (300 DPI) e watermarks em SMOKE.  
**Where**: `src/utils/paper_artifacts.py`  
**Depends on**: none  
**Reuses**: `src/training/evaluation.py`, `src/utils/statistics.py`  
**Requirement**: FIG-06, FIG-07  

**Done when**:

- [x] `generate_precision_recall_curves_plot()` interpola curvas PR contínuas em 100 pontos de recall com bandas $\pm 1\ \text{std}$ e ponto operacional $\tau_{\text{val}}$
- [x] `generate_quadrant_evaluation_plot()` traça barras agrupadas em 4 quadrantes (Q1 a Q4) por método com barras de erro $\pm 1\ \text{std}$
- [x] Ambas as funções exportam simultaneamente arquivos `.pdf` e `.png` (300 DPI)
- [x] Modo SMOKE insere tarja de aviso preliminar sem poluir modo REAL
- [x] Gate check passes: `.venv/bin/python -m compileall -q src tests`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `feat(paper): implement precision-recall curves and 4-quadrant evaluation plot generators`  

---

### T2: Criar suíte de testes automatizados para as figuras P2

**What**: Criar `tests/test_p2_pr_and_quadrant_figures.py` validando os geradores de FIG-06 e FIG-07, envelopes monotônicos de precisão, layout de 4 quadrantes, integridade de arquivos e modo SMOKE.  
**Where**: `tests/test_p2_pr_and_quadrant_figures.py`  
**Depends on**: T1  
**Reuses**: fixtures de teste de `paper_artifacts`  
**Requirement**: FIG-06, FIG-07, INT-02  

**Done when**:

- [x] Teste valida que FIG-06 interpola curvas PR contínuas e assinala $\tau_{\text{val}}$
- [x] Teste valida que FIG-07 plota os 4 quadrantes por método com barras de erro
- [x] Teste valida isolamento entre modos SMOKE e REAL
- [x] Gate check passes: `.venv/bin/pytest -v tests/test_p2_pr_and_quadrant_figures.py`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `test(paper): add regression test suite for pr curves and 4-quadrant figures`  

---

### T3: Integrar geradores P2 no orquestrador global de artefatos

**What**: Conectar `generate_precision_recall_curves_plot` e `generate_quadrant_evaluation_plot` em `generate_all_paper_artifacts()` em `src/utils/paper_artifacts.py`.  
**Where**: `src/utils/paper_artifacts.py`  
**Depends on**: T1, T2  
**Reuses**: pipeline de geração global de artefatos  
**Requirement**: INT-02  

**Done when**:

- [x] `generate_all_paper_artifacts()` executa as duas novas figuras e inclui seus caminhos no manifesto
- [x] Gate check passes: `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_p2_pr_and_quadrant_figures.py`  

**Tests**: integration  
**Gate**: full  
**Commit**: `feat(paper): integrate pr curves and quadrant plots into global artifacts manifest`  

---

### T4: Integrar visualização das figuras P2 no Notebook 05

**What**: Atualizar `notebooks/05_paper_artifacts.ipynb` para incluir células executáveis de demonstração e visualização interativa de FIG-06 e FIG-07.  
**Where**: `notebooks/05_paper_artifacts.ipynb`  
**Depends on**: T3  
**Reuses**: fluxo existente do notebook 05  
**Requirement**: INT-02  

**Done when**:

- [x] Notebook 05 contém células demonstrando a geração e exibição de FIG-06 e FIG-07
- [x] Gate check passes: `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_p2_pr_and_quadrant_figures.py -k "test_notebook_05_integration_p2"`  

**Tests**: integration  
**Gate**: quick  
**Commit**: `feat(notebooks): add pr curves and 4-quadrant visualization cells to notebook 05`  

---

### T5: Incorporar ambientes de figura correspondentes no manuscrito LaTeX

**What**: Atualizar `paper/main.tex` adicionando blocos de figura `\begin{figure}` com legendas e labels para FIG-06 e FIG-07 nas seções pertinentes.  
**Where**: `paper/main.tex`  
**Depends on**: T3  
**Reuses**: estrutura existente do template LaTeX  
**Requirement**: INT-02  

**Done when**:

- [x] `paper/main.tex` contém bloco figure para `fig_precision_recall_curves` (`fig:precision_recall_curves`)
- [x] `paper/main.tex` contém bloco figure para `fig_evaluation_4quadrants_comparison` (`fig:evaluation_4quadrants`)
- [x] Gate check passes: `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_p2_pr_and_quadrant_figures.py`  

**Tests**: integration  
**Gate**: build  
**Commit**: `docs(paper): add latex figure environments for pr curves and 4-quadrant evaluation`  

---

## Checks

**C1** - `generate_precision_recall_curves_plot()` computes continuous PR curves across 100 recall points with variance bands and tau_val marker, exporting dual PDF/PNG (FIG-06, AC 1, AC 2, AC 3, AC 4, AC 5)  
Proof: run unit | tests/test_p2_pr_and_quadrant_figures.py::test_precision_recall_curves_plot_generation PASSED  

**C2** - `generate_quadrant_evaluation_plot()` renders grouped 4-quadrant comparison across methods with error bars and exports dual PDF/PNG (FIG-07, AC 1, AC 2, AC 3, AC 4)  
Proof: run unit | tests/test_p2_pr_and_quadrant_figures.py::test_quadrant_evaluation_plot_generation PASSED  

**C3** - `generate_all_paper_artifacts()` orchestrates all P2 plots and includes their paths in the generated manifest (INT-02, AC 1)  
Proof: run unit | tests/test_p2_pr_and_quadrant_figures.py::test_generate_all_artifacts_includes_p2_figures PASSED  

**C4** - Notebook 05 integrates execution cells for FIG-06 and FIG-07 in SMOKE mode (INT-02, AC 3)  
Proof: run unit | tests/test_p2_pr_and_quadrant_figures.py::test_notebook_05_integration_p2 PASSED  

**C5** - LaTeX manuscript `paper/main.tex` integrates figure environments for FIG-06 and FIG-07 with proper labels and captions (INT-02, AC 2)  
Proof: run unit | tests/test_p2_pr_and_quadrant_figures.py::test_latex_figure_blocks_p2 PASSED  

---

## Handoff

- **Dispatched:** none
