# F25: P0 — Expansão Analítica: Fronteira de Pareto e Dinâmica de Entropia/Cobertura Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, the walk, check_proofs, the Verifier and the screens sub-agent).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Design**: `.specs/features/25-p0-expansao-figuras-pareto-e-entropia/design.md`  
**Status**: Ready  
**Approved**: 2026-11-11 - user: "aprovado"  
**Build**: `.venv/bin/python -m compileall -q src tests`  
**Run unit**: `.venv/bin/pytest -v tests/test_p0_pareto_entropy_figures.py`  

---

## Test Coverage Matrix

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Pareto Plot Generator (`src/utils/paper_artifacts.py`) | unit | Coordinates calculation, Pareto frontier identification, dual PDF/PNG export, SMOKE watermark | `tests/test_p0_pareto_entropy_figures.py` | `.venv/bin/pytest -v tests/test_p0_pareto_entropy_figures.py` |
| Entropy & Coverage Plot Generator (`src/utils/paper_artifacts.py`) | unit | Dual subplot generation, normalized entropy curves, coverage dynamics, dual export | `tests/test_p0_pareto_entropy_figures.py` | `.venv/bin/pytest -v tests/test_p0_pareto_entropy_figures.py` |
| Publication Pipeline Orchestration (`src/utils/paper_artifacts.py`) | integration | `generate_all_paper_artifacts()` integration, manifest paths presence, mode isolation | `tests/test_p0_pareto_entropy_figures.py` | `.venv/bin/pytest -v tests/test_p0_pareto_entropy_figures.py` |
| LaTeX Manuscript Integration (`paper/main.tex`) | integration | LaTeX figure environments existence, labels, captions | `tests/test_p0_pareto_entropy_figures.py` | `.venv/bin/pytest -v tests/test_p0_pareto_entropy_figures.py -k "test_latex_figure_blocks"` |
| Notebook 05 Integration (`notebooks/05_paper_artifacts.ipynb`) | integration | Execution and visualization cells for FIG-01 and FIG-02 in SMOKE mode | `tests/test_p0_pareto_entropy_figures.py` | `.venv/bin/pytest -v tests/test_p0_pareto_entropy_figures.py -k "test_notebook_05_integration"` |

---

## Gate Check Commands

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | After tasks with unit tests only | `.venv/bin/pytest -v tests/test_p0_pareto_entropy_figures.py` |
| Full | After tasks with integration tests | `.venv/bin/pytest -v tests/test_p0_pareto_entropy_figures.py tests/test_paper_artifacts.py` |
| Build | After phase completion or config/entity-only tasks | `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_p0_pareto_entropy_figures.py` |

---

## Execution Plan

Phases are ordered and run sequentially - each phase completes before the next begins, and tasks within a phase execute in order.

### Phase 1: Implementação dos Geradores de Figuras Analíticas (FIG-01, FIG-02)

Implementa `generate_pareto_cost_performance_plot()` e `generate_coverage_entropy_dynamics_plot()` em `src/utils/paper_artifacts.py`, garantindo exportação vetorial PDF e raster PNG 300 DPI, suporte a barras de erro e isolamento estrito de modo com marca d'água em SMOKE.

```
T1
```

### Phase 2: Suíte Automatizada de Testes de Figuras Analíticas (FIG-01, FIG-02, FIG-03)

Cria `tests/test_p0_pareto_entropy_figures.py` cobrindo o cálculo da fronteira de Pareto, plots temporais de entropia/cobertura, integridade de arquivos e modos de execução.

```
T2
```

### Phase 3: Integração no Orquestrador Global de Artefatos (FIG-03)

Integra a chamada dos novos geradores em `generate_all_paper_artifacts()` em `src/utils/paper_artifacts.py`.

```
T3
```

### Phase 4: Integração Interativa no Notebook 05 (FIG-03)

Atualiza o `notebooks/05_paper_artifacts.ipynb` para renderizar as novas figuras no fluxo interativo do notebook.

```
T4
```

### Phase 5: Incorporação das Figuras no Manuscrito LaTeX (FIG-03)

Atualiza `paper/main.tex` adicionando os ambientes `\begin{figure}` para FIG-01 e FIG-02 com legendas e labels acadêmicos.

```
T5
```

---

## Task Breakdown

### T1: Implementar geradores de FIG-01 (Pareto) e FIG-02 (Entropia/Cobertura)

**What**: Adicionar `generate_pareto_cost_performance_plot()` e `generate_coverage_entropy_dynamics_plot()` em `src/utils/paper_artifacts.py` com cálculo de fronteira de Pareto não-dominada, barras de dispersão $\pm 1\ \text{std}$, traçado de curvas de entropia e cobertura em formato duplo (.pdf e .png 300 DPI) com watermark em SMOKE mode.  
**Where**: `src/utils/paper_artifacts.py`  
**Depends on**: none  
**Reuses**: `src/utils/experiment_config.py`, matplotlib styling existents  
**Requirement**: FIG-01, FIG-02  

**Done when**:

- [x] `generate_pareto_cost_performance_plot()` calcula médias, stds e linha de Pareto conectando estratégias não-dominadas
- [x] `generate_coverage_entropy_dynamics_plot()` renderiza painel duplo com evolução de entropia $H(t)/H_{\max}$ e cobertura cumulativa de classes
- [x] Ambas as funções exportam simultaneamente arquivos `.pdf` e `.png` (300 DPI)
- [x] Modo SMOKE insere tarja de aviso preliminar sem poluir modo REAL
- [x] Gate check passes: `.venv/bin/python -m compileall -q src tests`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `feat(paper): implement pareto cost-performance and entropy dynamics plot generators`  

---

### T2: Criar suíte de testes automatizados para as novas figuras

**What**: Criar `tests/test_p0_pareto_entropy_figures.py` validando os geradores de Pareto e dinâmica de entropia, integridade dos arquivos gerados, cálculo de pontos não-dominados e marcas d'água em SMOKE mode.  
**Where**: `tests/test_p0_pareto_entropy_figures.py`  
**Depends on**: T1  
**Reuses**: fixtures de teste de `paper_artifacts`  
**Requirement**: FIG-01, FIG-02, FIG-03  

**Done when**:

- [x] Teste valida que FIG-01 gera .pdf e .png com fronteira de Pareto identificada
- [x] Teste valida que FIG-02 gera painel duplo com curvas de entropia e métricas de cobertura
- [x] Teste valida isolamento entre modos SMOKE e REAL
- [x] Gate check passes: `.venv/bin/pytest -v tests/test_p0_pareto_entropy_figures.py`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `test(paper): add test suite for pareto frontier and entropy dynamics plots`  

---

### T3: Integrar novos geradores em generate_all_paper_artifacts

**What**: Conectar `generate_pareto_cost_performance_plot` e `generate_coverage_entropy_dynamics_plot` dentro de `generate_all_paper_artifacts()` em `src/utils/paper_artifacts.py`.  
**Where**: `src/utils/paper_artifacts.py`  
**Depends on**: T1, T2  
**Reuses**: pipeline de geração global de artefatos  
**Requirement**: FIG-03  

**Done when**:

- [x] `generate_all_paper_artifacts()` executa ambos os geradores e inclui os caminhos em seu manifesto
- [x] Gate check passes: `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_p0_pareto_entropy_figures.py`  

**Tests**: integration  
**Gate**: full  
**Commit**: `feat(paper): integrate pareto and entropy plots in global artifacts manifest`  

---

### T4: Integrar visualização das novas figuras no Notebook 05

**What**: Atualizar `notebooks/05_paper_artifacts.ipynb` para incluir células executáveis de demonstração e plot interativo de FIG-01 e FIG-02.  
**Where**: `notebooks/05_paper_artifacts.ipynb`  
**Depends on**: T3  
**Reuses**: fluxo existente do notebook 05  
**Requirement**: FIG-03  

**Done when**:

- [x] Notebook 05 possui células demonstrando a geração e visualização das duas novas figuras
- [x] Gate check passes: `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_p0_pareto_entropy_figures.py -k "test_notebook_05_integration"`  

**Tests**: integration  
**Gate**: quick  
**Commit**: `feat(notebooks): add pareto and entropy dynamics visualization cells to notebook 05`  

---

### T5: Incorporar ambientes de figura correspondentes no manuscrito LaTeX

**What**: Atualizar `paper/main.tex` adicionando blocos de figura `\begin{figure}` com captions e labels acadêmicos para FIG-01 e FIG-02 nas seções pertinentes.  
**Where**: `paper/main.tex`  
**Depends on**: T3  
**Reuses**: estrutura existente do template LaTeX  
**Requirement**: FIG-03  

**Done when**:

- [x] `paper/main.tex` contém bloco figure para `fig_pareto_cost_vs_accuracy` com label `fig:pareto_cost`
- [x] `paper/main.tex` contém bloco figure para `fig_coverage_entropy_dynamics` com label `fig:coverage_entropy`
- [x] Teste de integridade de referências LaTeX passa sem erro
- [x] Gate check passes: `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_p0_pareto_entropy_figures.py`  

**Tests**: integration  
**Gate**: build  
**Commit**: `docs(paper): add latex figure environments for pareto frontier and entropy dynamics`  

---

## Checks

**C1** - `generate_pareto_cost_performance_plot()` identifies non-dominated Pareto frontier and exports dual PDF/PNG (FIG-01, AC 1, AC 2, AC 3, AC 4, AC 5)  
Proof: run unit | tests/test_p0_pareto_entropy_figures.py::test_pareto_cost_performance_plot_generation PASSED  

**C2** - `generate_coverage_entropy_dynamics_plot()` renders normalized entropy and cumulative coverage dual panel (FIG-02, AC 1, AC 2, AC 3, AC 4, AC 5)  
Proof: run unit | tests/test_p0_pareto_entropy_figures.py::test_coverage_entropy_dynamics_plot_generation PASSED  

**C3** - `generate_all_paper_artifacts()` orchestrates both plots and includes their paths in the generated manifest (FIG-03, AC 1)  
Proof: run unit | tests/test_p0_pareto_entropy_figures.py::test_generate_all_artifacts_includes_new_figures PASSED  

**C4** - LaTeX manuscript `paper/main.tex` integrates figure environments with proper labels and captions (FIG-03, AC 2)  
Proof: run unit | tests/test_p0_pareto_entropy_figures.py::test_latex_figure_blocks PASSED  

**C5** - Notebook 05 integrates execution cells for FIG-01 and FIG-02 in SMOKE mode (FIG-03, AC 3)  
Proof: run unit | tests/test_p0_pareto_entropy_figures.py::test_notebook_05_integration PASSED  

---

## Handoff

- **Dispatched:** none
