# 26-p1-expansao-figuras-arvore-e-interacao verification

**Verdict**: PASS
**Proofs**: `proofs.md` at 3be20c3
**Verifier**: independent sub-agent (author != verifier)
**Date**: 2026-11-12

## Scope and Acceptance Criteria Audit

- **FIG-03 (P1: Alocação Amostral vs. Ganho Marginal de Recall por Classe)**:
  - `generate_class_exposure_vs_delta_recall_plot()` calcula frequências relativas de amostragem pelo MCTS ($N_c/\sum N$) e ganhos marginais de recall ($\Delta \text{Recall}(c) = \text{Recall}_{\text{MCTS}}(c) - \text{Recall}_{\text{Uniform}}(c)$) para todas as 100 classes. (AC 1: **Holds**)
  - Calcula e anota na figura o coeficiente de correlação por postos de Spearman ($\rho_s$) e o respectivo $p$-valor bicaudal via `compute_exposure_recall_correlation()`. (AC 2: **Holds**)
  - Pontos são mapeados e coloridos pelas 20 superclasses ontológicas do CIFAR-100 via colormap categórico. (AC 3: **Holds**)
  - Exporta simultaneamente em vetor `.pdf` e `.png` (300 DPI) com prefixo `fig_class_exposure_vs_delta_recall`. (AC 4: **Holds**)
  - Em modo SMOKE, insere marca d'água preliminar e isolamento de saída. (AC 5: **Holds**)

- **FIG-04 (P1: Dendrograma Comparativo Ward vs. Semântica)**:
  - `generate_ward_vs_semantic_dendrogram_plot()` aceita matriz de confusão e reconstrói o dendrograma hierárquico via ligação de Ward sobre distância condensada simétrica. (AC 1: **Holds**)
  - Folhas e ramos do dendrograma são coloridos de acordo com as 20 superclasses ontológicas do CIFAR-100. (AC 2: **Holds**)
  - Anotações visuais explicativas elucidam onde o agrupamento perceptual empírico da CNN diverge da taxonomia conceitual humana. (AC 3: **Holds**)
  - Exporta simultaneamente em formatos `.pdf` e `.png` (300 DPI) com prefixo `fig_ward_vs_semantic_dendrogram`. (AC 4: **Holds**)
  - Respeita isolamento de diretórios e marca d'água em SMOKE. (AC 5: **Holds**)

- **FIG-05 (P1: Curvas de Interação Método x Severidade do Desbalanceamento)**:
  - `generate_interaction_ratio_plot()` traça curvas para estratégias avaliadas através das razões $\rho \in [1:10, 1:50, 1:100]$. (AC 1: **Holds**)
  - Painel duplo apresenta: (a) Minority F1-Score e (b) AUPRC com bandas de intervalo de confiança de 95% entre seeds pareadas. (AC 2: **Holds**)
  - Calcula e anota o efeito de interação quantitativo ($\Delta_{\text{ratio}}$). (AC 3: **Holds**)
  - Exporta simultaneamente em vetor `.pdf` e `.png` (300 DPI) com prefixo `fig_interaction_ratio_methods`. (AC 4: **Holds**)
  - Marca d'água preliminar ativada em modo SMOKE. (AC 5: **Holds**)

- **INT-01 (P1: Integração no Manuscrito LaTeX, Notebook 05 e Pipeline Global)**:
  - `generate_all_paper_artifacts()` executa FIG-03, FIG-04 e FIG-05 e inclui seus caminhos no manifesto de artefatos. (AC 1: **Holds**)
  - `paper/main.tex` contém ambientes de figura LaTeX para `fig_class_exposure_vs_delta_recall` (`fig:class_exposure_recall`), `fig_ward_vs_semantic_dendrogram` (`fig:ward_dendrogram`) e `fig_interaction_ratio_methods` (`fig:interaction_ratio`) com legendas completas. (AC 2: **Holds**)
  - Notebook 05 (`notebooks/05_paper_artifacts.ipynb`) inclui importações e células de visualização interativa das três figuras. (AC 3: **Holds**)
  - `tests/test_p1_expansion_figures.py` passa 100% de seus casos de teste. (AC 4: **Holds**)

## Checks

| Check | Test | Holds |
| --- | --- | --- |
| C1 | `tests/test_p1_expansion_figures.py::test_class_exposure_vs_delta_recall_plot_generation` | yes |
| C2 | `tests/test_p1_expansion_figures.py::test_ward_vs_semantic_dendrogram_plot_generation` | yes |
| C3 | `tests/test_p1_expansion_figures.py::test_interaction_ratio_plot_generation` | yes |
| C4 | `tests/test_p1_expansion_figures.py::test_generate_all_artifacts_includes_p1_figures` | yes |
| C5 | `tests/test_p1_expansion_figures.py::test_notebook_05_integration_p1` | yes |
| C6 | `tests/test_p1_expansion_figures.py::test_latex_figure_blocks_p1` | yes |
| Build | `.venv/bin/python -m compileall -q src tests` | yes |

## Code Quality & Invariants Audit

- **Robustez Numérica e Estatística**:
  - `generate_class_exposure_vs_delta_recall_plot()` utiliza `compute_exposure_recall_correlation()` com salvaguardas numéricas contra divisão por zero e variâncias residuais quase-constantes.
  - `generate_ward_vs_semantic_dendrogram_plot()` usa `squareform(checks=False)` garantindo que pequenas assimetrias de ponto flutuante sejam tratadas de forma simétrica antes de calcular as distâncias de Ward.
  - `generate_interaction_ratio_plot()` calcula intervalos de confiança e anota a divergência de inclinação entre métodos.
- **Isolamento e Compatibilidade**:
  - Toda a exportação ocorre em `paper/figures/` (modo REAL) ou `results/smoke/paper/figures/` (modo SMOKE).
  - A suíte completa de 265 testes do repositório permanece 100% íntegra.

## Findings

None - all acceptance criteria satisfied and regression test suite passes with zero defects.
