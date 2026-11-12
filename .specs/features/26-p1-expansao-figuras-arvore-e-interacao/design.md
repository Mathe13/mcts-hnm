# F26: P1 — Expansão Analítica: Geometria de Árvores e Interação Método x Desbalanceamento Design

## Architecture Overview

Esta feature introduz as três figuras da Fase 2 (P1) do roadmap científico, explorando o aprendizado por currículo ativo (FIG-03), a taxonomia perceptual versus humana (FIG-04) e a robustez a desbalanceamento extremo (FIG-05):

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Campaign Runs / Artifacts                       │
│    (confusion_matrices, class_exposures, multi-ratio evaluation)       │
└───────────────────┬───────────────────┬───────────────────┬────────────┘
                    │                   │                   │
                    ▼                   ▼                   ▼
┌─────────────────────────┐ ┌──────────────────────┐ ┌───────────────────┐
│generate_class_exposure_ │ │generate_ward_vs_     │ │generate_          │
│vs_delta_recall_plot     │ │semantic_dendrogram_  │ │interaction_ratio_ │
│  - Scatter: N(c) vs dRec│ │plot                  │ │plot               │
│  - Spearman rho & p-val │ │ - Ward linkage       │ │ - F1 & AUPRC vs r │
│  - Dual: .pdf + .png    │ │ - 20 Superclass cols │ │ - 95% CI bands    │
└───────────────────┬─────┘ └───────────┬──────────┘ └───────────┬───────┘
                    │                   │                        │
                    └───────────────────┼────────────────────────┘
                                        ▼
                          generate_all_paper_artifacts()
                                        ▼
                ┌───────────────────────┴───────────────────────┐
                ▼                                               ▼
       paper/main.tex (LaTeX)                   notebooks/05_paper_artifacts.ipynb
```

## Component Details

### 1. `generate_class_exposure_vs_delta_recall_plot` (`src/utils/paper_artifacts.py`)
- **Inputs**: Manifest ou payloads de runs do Cenário 2 contendo `exposure_diagnostics` (`class_exposure_counts`) e métricas por classe de validação/teste (`per_class_recalls`).
- **Cálculo**:
  - Para cada classe $c \in \{0, \dots, 99\}$:
    - Frequência relativa de amostragem pelo MCTS: $f_{\text{MCTS}}(c) = N_c / \sum_k N_k$.
    - Ganho marginal de recall: $\Delta \text{Recall}(c) = \text{Recall}_{\text{MCTS}}(c) - \text{Recall}_{\text{Uniform}}(c)$.
  - Estatística inferencial: `compute_exposure_recall_correlation()` calcula o coeficiente de Spearman $\rho_s$ e $p$-valor bicaudal.
- **Visualização**: Scatter plot com reta de regressão ajustada, marcadores coloridos pela superclasse ontológica da classe, caixa de anotação com $\rho_s$ e $p$.

### 2. `generate_ward_vs_semantic_dendrogram_plot` (`src/utils/paper_artifacts.py`)
- **Inputs**: Matriz de confusão de validação ou payload de árvore de confusão (`embeddings/{mode}/trees/confusion_tree_ward.pt` ou `tier1_axis2_tree_ablation.json`).
- **Cálculo**:
  - Matriz de distância condensada simétrica: $D_{i,j} = 1 - \frac{C_{i,j} + C_{j,i}}{2 \cdot \max(C) + \epsilon}$.
  - Matriz de ligação hierárquica via `scipy.cluster.hierarchy.linkage(condensed_dist, method="ward")`.
- **Visualização**:
  - `scipy.cluster.hierarchy.dendrogram` com folhas nomeadas pelas classes finas.
  - Paleta de 20 cores distintas para as superclasses do CIFAR-100 mapeadas aos rótulos das folhas.
  - Anotações gráficas destacando clusters onde classes de diferentes superclasses se aproximaram perceptualmente.

### 3. `generate_interaction_ratio_plot` (`src/utils/paper_artifacts.py`)
- **Inputs**: Resultados multi-razão do Cenário 1 (`tier1_axis1_samplers.json` com `imbalance_ratios=("1:10", "1:50", "1:100")`).
- **Cálculo**:
  - Agrupamento por estratégia e razão de desbalanceamento $\rho$.
  - Média e intervalo de confiança de 95% ($t_{\text{crit}} \times \text{std} / \sqrt{N}$) por razão.
  - Anotação da estatística de interação: $\text{Interação}(1:10, 1:100) = \Delta_{\text{MCTS}}(1:100) - \Delta_{\text{MCTS}}(1:10)$.
- **Visualização**: Painel duplo:
  - Subplot (a): Minority F1-Score vs. $\rho$ (escala semi-log ou categórica com linhas e marcadores).
  - Subplot (b): AUPRC vs. $\rho$.

### 4. Integração Global e LaTeX
- Incorporação em `generate_all_paper_artifacts()`.
- Seções correspondentes em `paper/main.tex`:
  - `fig:class_exposure_recall`: Seção de Análise de Currículo Ativo.
  - `fig:ward_dendrogram`: Seção de Topologia e Geometria da Árvore.
  - `fig:interaction_ratio`: Seção de Sensibilidade ao Desbalanceamento.
