# F26: P1 — Expansão Analítica: Geometria de Árvores e Interação Método x Desbalanceamento Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, the walk, check_proofs, the Verifier and the screens sub-agent).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Design**: `.specs/features/26-p1-expansao-figuras-arvore-e-interacao/design.md`  
**Status**: Ready  
**Approved**: 2026-11-12 - user: "sim"  
**Build**: `.venv/bin/python -m compileall -q src tests`  
**Run unit**: `.venv/bin/pytest -v tests/test_p1_expansion_figures.py`  

---

## Test Coverage Matrix

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Class Exposure vs Recall Generator (`src/utils/paper_artifacts.py`) | unit | Relative visit frequencies, delta recall, Spearman correlation annotation, dual export | `tests/test_p1_expansion_figures.py` | `.venv/bin/pytest -v tests/test_p1_expansion_figures.py` |
| Ward vs Semantic Dendrogram Generator (`src/utils/paper_artifacts.py`) | unit | Ward distance condensation, hierarchical linkage, 20-superclass coloring, dual export | `tests/test_p1_expansion_figures.py` | `.venv/bin/pytest -v tests/test_p1_expansion_figures.py` |
| Interaction Ratio Generator (`src/utils/paper_artifacts.py`) | unit | Dual-panel multi-ratio curves, 95% CI bands, interaction effect annotation, dual export | `tests/test_p1_expansion_figures.py` | `.venv/bin/pytest -v tests/test_p1_expansion_figures.py` |
| Publication Pipeline Orchestration (`src/utils/paper_artifacts.py`) | integration | `generate_all_paper_artifacts()` integration, manifest presence, mode isolation | `tests/test_p1_expansion_figures.py` | `.venv/bin/pytest -v tests/test_p1_expansion_figures.py` |
| Notebook 05 Integration (`notebooks/05_paper_artifacts.ipynb`) | integration | Cells for FIG-03, FIG-04, and FIG-05 visualization in SMOKE mode | `tests/test_p1_expansion_figures.py` | `.venv/bin/pytest -v tests/test_p1_expansion_figures.py -k "test_notebook_05_integration_p1"` |
| LaTeX Manuscript Integration (`paper/main.tex`) | integration | Environments, labels `fig:class_exposure_recall`, `fig:ward_dendrogram`, `fig:interaction_ratio` | `tests/test_p1_expansion_figures.py` | `.venv/bin/pytest -v tests/test_p1_expansion_figures.py -k "test_latex_figure_blocks_p1"` |

---

## Gate Check Commands

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | After tasks with unit tests only | `.venv/bin/pytest -v tests/test_p1_expansion_figures.py` |
| Full | After tasks with integration tests | `.venv/bin/pytest -v tests/test_p1_expansion_figures.py tests/test_paper_artifacts.py` |
| Build | After phase completion or config/entity-only tasks | `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_p1_expansion_figures.py` |

---

## Execution Plan

Phases are ordered and run sequentially - each phase completes before the next begins, and tasks within a phase execute in order.

### Phase 1: Implementação dos Novos Geradores da Fase 2 (FIG-03, FIG-04, FIG-05)

Adiciona `generate_class_exposure_vs_delta_recall_plot()`, `generate_ward_vs_semantic_dendrogram_plot()` e `generate_interaction_ratio_plot()` em `src/utils/paper_artifacts.py` com exportação dupla (.pdf e .png 300 DPI) e isolamento SMOKE/REAL.

```
T1
```

### Phase 2: Suíte Automatizada de Testes de Figuras P1 (FIG-03, FIG-04, FIG-05, INT-01)

Cria `tests/test_p1_expansion_figures.py` cobrindo o cálculo de correlação de Spearman, dendrograma hierárquico, curvas de interação com bandas IC 95% e integridade dos arquivos.

```
T2
```

### Phase 3: Integração no Orquestrador Global de Artefatos (INT-01)

Conecta a execução das três novas figuras dentro de `generate_all_paper_artifacts()` em `src/utils/paper_artifacts.py`.

```
T3
```

### Phase 4: Integração Interativa no Notebook 05 (INT-01)

Atualiza o `notebooks/05_paper_artifacts.ipynb` para renderizar FIG-03, FIG-04 e FIG-05 no fluxo interativo do notebook.

```
T4
```

### Phase 5: Incorporação das Figuras no Manuscrito LaTeX (INT-01)

Atualiza `paper/main.tex` adicionando os ambientes `\begin{figure}` para FIG-03, FIG-04 e FIG-05 com legendas e labels acadêmicos.

```
T5
```

---

## Task Breakdown

### T1: Implementar geradores de FIG-03, FIG-04 e FIG-05 em paper_artifacts.py

**What**: Implementar `generate_class_exposure_vs_delta_recall_plot()`, `generate_ward_vs_semantic_dendrogram_plot()` e `generate_interaction_ratio_plot()` em `src/utils/paper_artifacts.py` com cálculos estatísticos, estilização acadêmica, exportação em .pdf e .png (300 DPI) e watermarks em SMOKE.  
**Where**: `src/utils/paper_artifacts.py`  
**Depends on**: none  
**Reuses**: `src/training/exposure.py`, `src/utils/statistics.py`, `src/data/cifar100.py`  
**Requirement**: FIG-03, FIG-04, FIG-05  

**Done when**:

- [x] `generate_class_exposure_vs_delta_recall_plot()` calcula frequências de visita, $\Delta \text{Recall}$ e anota Spearman $\rho_s$ e $p$-valor
- [x] `generate_ward_vs_semantic_dendrogram_plot()` gera o dendrograma de Ward com folhas e ramos categorizados pelas 20 superclasses ontológicas
- [x] `generate_interaction_ratio_plot()` gera painel duplo com curvas de F1 e AUPRC vs. $\rho$, bandas de IC 95% e anotação do efeito de interação
- [x] Todas as funções exportam simultaneamente arquivos `.pdf` e `.png` (300 DPI)
- [x] Gate check passes: `.venv/bin/python -m compileall -q src tests`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `feat(paper): implement exposure-recall, dendrogram and interaction plot generators`  

---

### T2: Criar suíte de testes automatizados para as figuras P1

**What**: Criar `tests/test_p1_expansion_figures.py` validando os geradores de FIG-03, FIG-04 e FIG-05, integridade dos formatos, valores estatísticos de correlação e interação, e modo SMOKE.  
**Where**: `tests/test_p1_expansion_figures.py`  
**Depends on**: T1  
**Reuses**: fixtures de teste de `paper_artifacts`  
**Requirement**: FIG-03, FIG-04, FIG-05, INT-01  

**Done when**:

- [ ] Teste valida que FIG-03 plota correlação e exporta .pdf e .png
- [ ] Teste valida que FIG-04 reconstrói o dendrograma de Ward e colore por superclasse
- [ ] Teste valida que FIG-05 traça curvas com bandas de confiança e calcula a interação
- [ ] Gate check passes: `.venv/bin/pytest -v tests/test_p1_expansion_figures.py`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `test(paper): add regression test suite for p1 expansion figures`  

---

### T3: Integrar geradores P1 no orquestrador global de artefatos

**What**: Conectar `generate_class_exposure_vs_delta_recall_plot`, `generate_ward_vs_semantic_dendrogram_plot` e `generate_interaction_ratio_plot` em `generate_all_paper_artifacts()` em `src/utils/paper_artifacts.py`.  
**Where**: `src/utils/paper_artifacts.py`  
**Depends on**: T1, T2  
**Reuses**: pipeline de geração global de artefatos  
**Requirement**: INT-01  

**Done when**:

- [ ] `generate_all_paper_artifacts()` executa as três novas figuras e inclui seus caminhos no manifesto
- [ ] Gate check passes: `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_p1_expansion_figures.py`  

**Tests**: integration  
**Gate**: full  
**Commit**: `feat(paper): integrate p1 figures into global artifacts manifest`  

---

### T4: Integrar visualização das figuras P1 no Notebook 05

**What**: Atualizar `notebooks/05_paper_artifacts.ipynb` para incluir células executáveis de demonstração e visualização interativa de FIG-03, FIG-04 e FIG-05.  
**Where**: `notebooks/05_paper_artifacts.ipynb`  
**Depends on**: T3  
**Reuses**: fluxo existente do notebook 05  
**Requirement**: INT-01  

**Done when**:

- [ ] Notebook 05 contém células demonstrando a geração e exibição das três figuras da Fase 2
- [ ] Gate check passes: `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_p1_expansion_figures.py -k "test_notebook_05_integration_p1"`  

**Tests**: integration  
**Gate**: quick  
**Commit**: `feat(notebooks): add p1 figures visualization cells to notebook 05`  

---

### T5: Incorporar ambientes de figura correspondentes no manuscrito LaTeX

**What**: Atualizar `paper/main.tex` adicionando blocos de figura `\begin{figure}` com legendas e labels para FIG-03, FIG-04 e FIG-05 nas seções pertinentes.  
**Where**: `paper/main.tex`  
**Depends on**: T3  
**Reuses**: estrutura existente do template LaTeX  
**Requirement**: INT-01  

**Done when**:

- [ ] `paper/main.tex` contém bloco figure para `fig_class_exposure_vs_delta_recall` (`fig:class_exposure_recall`)
- [ ] `paper/main.tex` contém bloco figure para `fig_ward_vs_semantic_dendrogram` (`fig:ward_dendrogram`)
- [ ] `paper/main.tex` contém bloco figure para `fig_interaction_ratio_methods` (`fig:interaction_ratio`)
- [ ] Gate check passes: `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_p1_expansion_figures.py`  

**Tests**: integration  
**Gate**: build  
**Commit**: `docs(paper): add latex figure environments for p1 expansion figures`  

---

## Checks

**C1** - `generate_class_exposure_vs_delta_recall_plot()` computes class relative exposure, delta recall, Spearman correlation, and exports dual PDF/PNG (FIG-03, AC 1, AC 2, AC 3, AC 4, AC 5)  
Proof: run unit | tests/test_p1_expansion_figures.py::test_class_exposure_vs_delta_recall_plot_generation PASSED  

**C2** - `generate_ward_vs_semantic_dendrogram_plot()` reconstructs Ward dendrogram with 20-superclass coloring and exports dual PDF/PNG (FIG-04, AC 1, AC 2, AC 3, AC 4, AC 5)  
Proof: run unit | tests/test_p1_expansion_figures.py::test_ward_vs_semantic_dendrogram_plot_generation PASSED  

**C3** - `generate_interaction_ratio_plot()` renders dual-panel curves with 95% CI bands and interaction effect annotation (FIG-05, AC 1, AC 2, AC 3, AC 4, AC 5)  
Proof: run unit | tests/test_p1_expansion_figures.py::test_interaction_ratio_plot_generation PASSED  

**C4** - `generate_all_paper_artifacts()` orchestrates all P1 plots and includes their paths in the generated manifest (INT-01, AC 1)  
Proof: run unit | tests/test_p1_expansion_figures.py::test_generate_all_artifacts_includes_p1_figures PASSED  

**C5** - Notebook 05 integrates execution cells for FIG-03, FIG-04 and FIG-05 in SMOKE mode (INT-01, AC 3)  
Proof: run unit | tests/test_p1_expansion_figures.py::test_notebook_05_integration_p1 PASSED  

**C6** - LaTeX manuscript `paper/main.tex` integrates figure environments for all P1 figures with proper labels and captions (INT-01, AC 2)  
Proof: run unit | tests/test_p1_expansion_figures.py::test_latex_figure_blocks_p1 PASSED  

---

## Handoff

- **Dispatched:** none
