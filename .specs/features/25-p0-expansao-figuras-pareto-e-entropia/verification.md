# 25-p0-expansao-figuras-pareto-e-entropia verification

**Verdict**: PASS
**Proofs**: `proofs.md` at 5dc2c7d
**Verifier**: independent sub-agent (author != verifier)
**Date**: 2026-11-12

## Scope and Acceptance Criteria Audit

- **FIG-01 (P0: Fronteira de Pareto Custo vs. Desempenho)**:
  - `generate_pareto_cost_performance_plot()` aceita manifesto ou diretório de campanha com `computational_costs` (`elapsed_seconds`, `gradient_samples`) e métricas de desempenho. (AC 1: **Holds**)
  - Calcula médias e desvios padrão ($\pm 1\ \text{std}$) entre seeds para cada estratégia e traça pontos de dispersão com barras de erro. (AC 2: **Holds**)
  - Identifica estratégias não-dominadas na fronteira de Pareto (minimizando custo e maximizando desempenho) e conecta-as com linha tracejada e anotação. (AC 3: **Holds**)
  - Exporta simultaneamente em vetor `.pdf` e raster `.png` (300 DPI) com prefixo configurável `fig_pareto_cost_vs_accuracy` no diretório alvo. (AC 4: **Holds**)
  - Em modo SMOKE, insere marca d'água semi-transparente preliminar sem poluir modo REAL. (AC 5: **Holds**)

- **FIG-02 (P0: Dinâmica Temporal de Entropia e Cobertura de Classes no Cenário 2)**:
  - `generate_coverage_entropy_dynamics_plot()` processa execuções de Cenário 2 contendo `history`/`step_history` com `normalized_entropy` e diagnósticos de exposição. (AC 1: **Holds**)
  - Subplot (a) traça curvas de evolução da entropia normalizada $H(t)/H_{\max}$ ao longo dos 1.000 passos para `uniform`, `mcts_strict`, `mcts_enhanced` e `mcts_coverage_balanced` com bandas sombreadas $\pm 1\ \text{std}$ e linhas de limiar de colapso ($<0.35$) e regime ótimo ($0.75-0.85$). (AC 2: **Holds**)
  - Subplot (b) plota cobertura cumulativa de classes (% de classes com $k \ge 1$ e $k \ge 5$ amostragens) contrastando políticas. (AC 3: **Holds**)
  - Exporta simultaneamente em vetor `.pdf` e raster `.png` (300 DPI) com prefixo `fig_coverage_entropy_dynamics`. (AC 4: **Holds**)
  - Respeita isolamento estrito de modo e marca d'água em modo SMOKE. (AC 5: **Holds**)

- **FIG-03 (P0: Integração no Manuscrito LaTeX, Notebook 05 e Pipeline Global)**:
  - `generate_all_paper_artifacts()` orquestra automaticamente `generate_pareto_cost_performance_plot()` e `generate_coverage_entropy_dynamics_plot()`, registrando seus caminhos no manifesto de artefatos. (AC 1: **Holds**)
  - `paper/main.tex` contém blocos `\begin{figure}` e `\begin{figure*}` para `fig_pareto_cost_vs_accuracy` (`fig:pareto_cost`) e `fig_coverage_entropy_dynamics` (`fig:coverage_entropy`) com legendas e referências acadêmicas. (AC 2: **Holds**)
  - Notebook 05 (`notebooks/05_paper_artifacts.ipynb`) inclui importações, geração e células de exibição interativa para ambas as figuras. (AC 3: **Holds**)
  - A suíte de testes automatizados `tests/test_p0_pareto_entropy_figures.py` passa 100% dos seus casos de teste. (AC 4: **Holds**)

## Checks

| Check | Test | Holds |
| --- | --- | --- |
| C1 | `tests/test_p0_pareto_entropy_figures.py::test_pareto_cost_performance_plot_generation` | yes |
| C2 | `tests/test_p0_pareto_entropy_figures.py::test_coverage_entropy_dynamics_plot_generation` | yes |
| C3 | `tests/test_p0_pareto_entropy_figures.py::test_generate_all_artifacts_includes_new_figures` | yes |
| C4 | `tests/test_p0_pareto_entropy_figures.py::test_latex_figure_blocks` | yes |
| C5 | `tests/test_p0_pareto_entropy_figures.py::test_notebook_05_integration` | yes |
| Build | `.venv/bin/python -m compileall -q src tests` | yes |

## Code Quality & Invariants Audit

- **Geradores em `src/utils/paper_artifacts.py`**:
  - Implementação pura sem dependências externas desnecessárias, utilizando `matplotlib` em backend não-interativo `Agg` para evitar vazamentos de memória e erros em ambientes headless/JupyterHub.
  - Fechamento explícito de figuras via `plt.close(fig)` em todos os caminhos de execução.
  - Verificação de dados robusta com fallback representativo quando em modo SMOKE sem campanhas prévias, e falha rápida explícita em modo REAL sem suposição de dados sintéticos.
  - Cálculo formal de dominância de Pareto ($x_A \le x_B \land y_A \ge y_B \land (x_A < x_B \lor y_A > y_B)$) e ordenação estrita.
  - Alinhamento de passos de entropia via interpolação linear (`np.interp`) protegendo contra descontinuidade temporal em seeds assíncronas.
- **Orquestração e Isolamento**:
  - Respeita integralmente os diretórios isolados `results/smoke/` vs `results/real/` e `paper/figures/`.
  - Preserva compatibilidade regressiva com a suíte de 30 testes em `test_paper_artifacts_real.py`.
- **Manuscrito LaTeX (`paper/main.tex`)**:
  - Subseções dedicadas com discussões detalhadas conectando as evidências empíricas às hipóteses teóricas de RL e amostragem de negativos.

## Findings

None - all acceptance criteria satisfied and regression test suite passes with zero defects.
