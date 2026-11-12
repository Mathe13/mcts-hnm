# F25: P0 — Expansão Analítica: Fronteira de Pareto e Dinâmica de Entropia/Cobertura Design

## Architecture Overview

Esta feature expande o módulo de artefatos científicos `src/utils/paper_artifacts.py` implementando duas figuras de alta relevância metodológica derivadas da instrumentação de `computational_costs` e `exposure_diagnostics`:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Campaign Runs / Manifest                        │
│             (computational_costs, step_history, exposure)              │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
                    ▼                                ▼
┌──────────────────────────────────────┐ ┌───────────────────────────────┐
│ generate_pareto_cost_performance_plot│ │generate_coverage_entropy_plot │
│   - Cost: elapsed_seconds/samples    │ │ - Subplot A: Normalized H(t)  │
│   - Y-Axis: Minority / Macro-F1      │ │ - Subplot B: Coverage k>=1/5  │
│   - Error bars (std over seeds)      │ │ - Curves for 4 strategies     │
│   - Non-dominated Pareto frontier    │ │ - Dual output: .pdf + .png    │
└───────────────────┬──────────────────┘ └───────────────┬───────────────┘
                    │                                    │
                    └─────────────────┬──────────────────┘
                                      ▼
                        generate_all_paper_artifacts()
                                      ▼
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
     paper/main.tex (LaTeX)                   notebooks/05_paper_artifacts.ipynb
```

## Component Details

### 1. `generate_pareto_cost_performance_plot` (`src/utils/paper_artifacts.py`)
- **Inputs**: Manifest ou diretório de campanha com runs de Cenário 1 ou Cenário 2.
- **Eixo X**: Custo computacional médio (tempo em segundos ou gradient samples calculados a partir de `payload["computational_costs"]`).
- **Eixo Y**: Métrica de desempenho média (`minority_f1` para Cenário 1 ou `macro_f1` / `accuracy` para Cenário 2).
- **Dispersão**: Barras de erro $\pm 1\ \text{std}$ em ambos os eixos quando múltiplas sementes estiverem disponíveis.
- **Linha de Pareto**: Algoritmo de identificação de estratégias não-dominadas (onde nenhuma outra estratégia possui simultaneamente menor custo e maior pontuação). Conecta os pontos dominantes em ordem crescente de custo.
- **Estilos**: Marcadores e cores distintas para `uniform` (cinza), `ohem` (vermelho), `mcts_strict` (laranja), `mcts_coverage_balanced` (roxo) e `mcts_enhanced` (verde).
- **Modos**:
  - SMOKE: overlay `SMOKE MODE - PRELIMINARY DATA`, saída em `results/smoke/paper/figures/`.
  - REAL: saída estrita em `paper/figures/`.

### 2. `generate_coverage_entropy_dynamics_plot` (`src/utils/paper_artifacts.py`)
- **Inputs**: Execuções de Cenário 2 contendo `history` / `step_history` com `normalized_entropy` e diagnósticos de cobertura.
- **Subplot (a) - Entropia de Visitação**:
  - Traça $H_{\text{norm}}(t)$ ao longo dos passos para cada estratégia avaliada.
  - Média com banda sombreada de variância ($\pm 1\ \text{std}$) entre seeds.
  - Linhas de referência horizontal para patamares esperados.
- **Subplot (b) - Cobertura Cumulativa de Classes**:
  - Evolução do percentual de classes cobertas ($k \ge 1$ e $k \ge 5$) ao longo do treinamento, ou barras comparativas finais de cobertura.
- **Exportação**: Gera simultaneamente `fig_coverage_entropy_dynamics.pdf` e `fig_coverage_entropy_dynamics.png` (300 DPI).

### 3. Integração com LaTeX (`paper/main.tex`)
- Inclusão de figuras flutuantes:
  - `fig:pareto_cost`: Fronteira de Pareto na Seção de Eficiência Computacional.
  - `fig:coverage_entropy`: Dinâmica de Entropia e Cobertura na Seção de Dinâmica do Cenário 2.

### 4. Integração no Notebook 05 e Suíte de Testes
- Adição de células de invocação no `notebooks/05_paper_artifacts.ipynb`.
- Suíte `tests/test_p0_pareto_entropy_figures.py` com testes unitários e de integração em modo SMOKE.
