# F23: P1 — Ablações de Mecanismo (Strict vs Enhanced) e Comparação Estatística das Hierarquias Design

## System Architecture Overview

Esta feature resolve os dois objetivos P1 prioritários do roadmap metodológico:
1. Isolar rigorosamente os fatores do grid fatorial $2 \times 2$ de mecanismos MCTS com controles de custo computacional.
2. Realizar a comparação inferencial pareada seed a seed entre as hierarquias Semântica e Ward no Axis 2, eliminando conclusões baseadas em médias sem controle de dispersão amostral.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                           MCTS-HNM P1 Evaluation                              │
└───────────────────────┬───────────────────────────────┬───────────────────────┘
                        │                               │
                        ▼                               ▼
    ┌───────────────────────────────────────┐   ┌───────────────────────────────┐
    │     Axis 2: Hierarchy Evaluation      │   │  2x2 Factorial Mechanism Grid │
    │     (Semantic vs Ward Confusion)      │   │   (Strict vs Enhanced Abl.)   │
    └───────────────────┬───────────────────┘   └───────────────┬───────────────┘
                        │                                       │
                        ▼                                       ▼
        ┌───────────────────────────────┐       ┌───────────────────────────────┐
        │  Paired Inferential Testing   │       │  Factorial Effects Isolation  │
        │  • Seed-by-seed paired deltas │       │  • Main effect: Q-norm        │
        │  • 95% Confidence Intervals   │       │  • Main effect: Dual-Stream   │
        │  • Cohen's d effect size      │       │  • Interaction term           │
        │  • Student's t + Wilcoxon     │       │  • Computational cost metrics │
        │  • Tree topological metrics   │       └───────────────┬───────────────┘
        └───────────────┬───────────────┘                       │
                        │                                       │
                        └───────────────────┬───────────────────┘
                                            │
                                            ▼
                        ┌───────────────────────────────────────┐
                        │   Unified Persistence & Reports       │
                        │   • tier1_axis2_tree_ablation.json    │
                        │   • tier1_mechanism_ablation.json     │
                        │   • Notebook 03 comparative tables    │
                        └───────────────────────────────────────┘
```

---

## Detailed Component Design

### 1. Decomposição Fatorial $2 \times 2$ e Métricas de Custo (`src/utils/statistics.py` e `src/training/benchmark_runner.py`)

- **Grid Fatorial**:
  - $S_{00}$ = `mcts_strict`: Q bruto, 100% MCTS
  - $S_{10}$ = `mcts_qnorm_only`: Q Min-Max normalizado, 100% MCTS
  - $S_{01}$ = `mcts_dualstream_only`: Q bruto, 50% MCTS / 50% Uniforme
  - $S_{11}$ = `mcts_enhanced`: Q Min-Max normalizado, 50% MCTS / 50% Uniforme

- **Função `compute_2x2_factorial_effects(scores_dict: dict[str, list[float]]) -> dict[str, Any]`**:
  - Para cada semente pareada $s \in \{1, \dots, N\}$:
    - $\text{ME}_{Q}(s) = \frac{1}{2} [(S_{10}(s) + S_{11}(s)) - (S_{00}(s) + S_{01}(s))]$
    - $\text{ME}_{\text{DS}}(s) = \frac{1}{2} [(S_{01}(s) + S_{11}(s)) - (S_{00}(s) + S_{10}(s))]$
    - $\text{Interação}(s) = (S_{11}(s) - S_{10}(s)) - (S_{01}(s) - S_{00}(s))$
  - Calcula médias, desvios-padrão, intervalos de confiança e testes estatísticos para $\text{ME}_{Q}$, $\text{ME}_{\text{DS}}$ e $\text{Interação}$.

- **Controles de Custo**:
  - `total_updates`: número de passos de treino executados.
  - `gradient_samples`: total de amostras efetivamente retropropagadas pelo modelo.
  - `mining_forwards`: forward passes de avaliação adicionais utilizados por amostradores (ex.: OHEM).
  - `elapsed_seconds`: tempo real de parede da execução.
  - Os custos são extraídos de `res["computational_costs"]` e incorporados em `final_metrics` e nas tabelas CSV.

### 2. Comparação Inferencial Pareada de Hierarquias no Axis 2 (`src/training/benchmark_runner.py`)

- **Emparelhamento Seed a Seed**:
  - Para cada semente compartilhada entre `semantic` e `confusion_ward` sob o mesmo baseline (`mcts_enhanced`):
    - $\Delta_{\text{AUPRC}}(s) = \text{AUPRC}_{\text{semantic}}(s) - \text{AUPRC}_{\text{confusion\_ward}}(s)$
    - $\Delta_{\text{F1}}(s) = \text{F1}_{\text{semantic}}(s) - \text{F1}_{\text{confusion\_ward}}(s)$
  - Se $N \ge 2$, calcula:
    - `mean_diff`, `std_diff`, `se_diff`, `ci_95_lower`, `ci_95_upper`
    - `cohens_d`: tamanho de efeito padronizado
    - `t_stat`, `t_pvalue`: teste t de Student pareado
    - `wilcoxon_stat`, `wilcoxon_pvalue`: teste não paramétrico de Wilcoxon

### 3. Diagnósticos Estruturais e Topológicos da Árvore (`src/trees/base_tree.py`)

- **Função `compute_tree_topology_diagnostics(tree: FastHierarchyTree) -> dict[str, Any]`**:
  - `max_depth`: profundidade máxima da árvore (raiz = 0).
  - `mean_leaf_depth`: profundidade média ponderada das folhas.
  - `num_leaves`: contagem de nós folha ativos.
  - `num_internal_nodes`: contagem de nós internos não-folha.
  - `total_nodes`: total de nós ($V$).
  - `branching_factor_mean`: média do número de filhos por nó interno.
  - `branching_factor_max`: número máximo de filhos de um nó interno.

### 4. Integração no Notebook 03 (`notebooks/03_tier1_mcts_benchmarks.ipynb`)

- **Seção 3 (Axis 2: Tree Hierarchy Ablation)**:
  - Executa as hierarquias `semantic` e `confusion_ward`.
  - Exibe a tabela inferencial pareada com deltas, IC 95%, Cohen's d e p-valores.
  - Emite o parecer estatístico formal: indica se a diferença entre Semantic e Ward possui significância estatística ($p < 0.05$) ou se a hipótese nula de equivalência não pode ser rejeitada.
- **Seção de Ablação Fatorial $2 \times 2$**:
  - Exibe a matriz de efeitos $2 \times 2$ com $\text{ME}_{Q}$, $\text{ME}_{\text{DS}}$ e $\text{Interação}$, além do custo computacional comparativo.

---

## Traceability to Requirements

| Requirement | Implementation Component | Verification Test |
| ----------- | ------------------------ | ----------------- |
| `ABL-01` | `src/utils/statistics.py` + `src/training/benchmark_runner.py` | `test_factorial_2x2_effects_and_costs` |
| `HIER-01` | `src/trees/base_tree.py` + `src/training/benchmark_runner.py` | `test_axis2_paired_hierarchy_comparison_and_topology` |
| `NOTE-01` | `notebooks/03_tier1_mcts_benchmarks.ipynb` | `test_notebook_03_p1_tables_and_execution` |
