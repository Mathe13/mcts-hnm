# F27: P2 — Expansão Analítica: Curvas Precision-Recall e Análise em 4 Quadrantes Design

## Architecture Overview

Esta feature implementa as duas figuras finais de transparência metodológica e caracterização contínua de desempenho do roadmap analítico:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Campaign Runs / Artifacts                       │
│           (PR curve points, 4-quadrant calibration metrics)            │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
                    ▼                                ▼
┌──────────────────────────────────────┐ ┌───────────────────────────────┐
│generate_precision_recall_curves_plot │ │generate_quadrant_evaluation_  │
│  - 100-point uniform recall grid     │ │plot                           │
│  - Mean & shaded +/- 1 std bands     │ │ - 4 Quadrants: Q1, Q2, Q3, Q4 │
│  - Operating point at tau_val        │ │ - Grouped bars + error bars   │
│  - Dual: .pdf + .png                 │ │ - Dual: .pdf + .png           │
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

### 1. `generate_precision_recall_curves_plot` (`src/utils/paper_artifacts.py`)
- **Inputs**: Run records ou dados de validação/teste contendo predições contínuas (probabilidades e rótulos verdadeiros) ou curvas PR discretas (`precision`, `recall`).
- **Cálculo**:
  - Grade linear de recall: $R_{\text{grid}} = \text{np.linspace}(0.0, 1.0, 100)$.
  - Para cada seed e estratégia, calcula envelope de precisão monotonicamente decrescente:
    $P_{\text{interp}}(r) = \max_{r' \ge r} P(r')$.
  - Agregação inter-sementes: calcula média $\bar{P}(r)$ e desvio padrão $\sigma_P(r)$.
  - Localização do ponto operacional: ponto $(R_{\tau_{\text{val}}}, P_{\tau_{\text{val}}})$ resultante do limiar ótimo $\tau_{\text{val}}$ calibrado na validação.
- **Visualização**: Linhas sólidas com bandas sombreadas ($\pm 1\ \text{std}$) para `Uniform`, `OHEM`, `MCTS Strict` e `Enhanced MCTS-HNM`, com marcadores circulares destacados no ponto de $\tau_{\text{val}}$.

### 2. `generate_quadrant_evaluation_plot` (`src/utils/paper_artifacts.py`)
- **Inputs**: Dicionário de resultados em 4 quadrantes (`tier1_axis1_samplers.json` ou `four_quadrants` de `evaluate_4quadrants()`).
- **Quadrantes**:
  - Q1: `last_raw` (Checkpoint Final, $\tau = 0.5$)
  - Q2: `last_calibrated` (Checkpoint Final, $\tau = \tau_{\text{val}}$)
  - Q3: `best_val_raw` (Melhor Checkpoint Validação, $\tau = 0.5$)
  - Q4: `best_val_calibrated` (Melhor Checkpoint Validação, $\tau = \tau_{\text{val}}$)
- **Visualização**: Gráfico de barras agrupadas em 4 quadrantes por método com barras de erro $\pm 1\ \text{std}$ entre sementes, permitindo constatar imediatamente o ganho incremental de seleção de checkpoint e de calibração de limiar.

### 3. Integração Global e Manuscrito
- Adição das funções ao orquestrador `generate_all_paper_artifacts()`.
- Seções correspondentes em `paper/main.tex`:
  - `fig:precision_recall_curves`: Seção de Desempenho Contínuo Precision-Recall.
  - `fig:evaluation_4quadrants`: Seção de Transparência e Avaliação em 4 Quadrantes.
