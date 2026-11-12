# F24: P2 — Expansão do Estudo: Múltiplas Classes-Alvo e Razões Pareadas Design

## System Architecture Overview

Esta feature expande o escopo metodológico do Scenario 1 (One-vs-Rest) para suportar a avaliação sistemática sob múltiplas razões de desbalanceamento e múltiplas classes-alvo, preservando o rigor do emparelhamento de sementes e splits, a rastreabilidade de proveniência e a compatibilidade dos artefatos.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                           MCTS-HNM P2 Study Expansion                         │
└───────────────────────┬───────────────────────────────┬───────────────────────┘
                        │                               │
                        ▼                               ▼
    ┌───────────────────────────────────────┐   ┌───────────────────────────────┐
    │     Multi-Ratio Grid Evaluation       │   │  Multi-Target Class Extension │
    │     (1:10, 1:50, 1:100)               │   │  (classes 0, 1, 2, ...)       │
    └───────────────────┬───────────────────┘   └───────────────┬───────────────┘
                        │                                       │
                        ▼                                       ▼
        ┌───────────────────────────────┐       ┌───────────────────────────────┐
        │ Method x Ratio Interactions   │       │ Inter-Class Generalization    │
        │ • Paired delta at r1 vs r2    │       │ • Per-class metrics & seeds   │
        │ • 95% Confidence Intervals    │       │ • Cross-class mean & std      │
        │ • Student's t hypothesis test │       │ • Class-stratified audit      │
        └───────────────┬───────────────┘       └───────────────┬───────────────┘
                        │                                       │
                        └───────────────────┬───────────────────┘
                                            │
                                            ▼
                        ┌───────────────────────────────────────┐
                        │   Unified Persistence & Notebooks     │
                        │   • tier1_axis1_samplers.json & .csv  │
                        │   • ratio_interactions & class_summary│
                        │   • Notebook 03 (Section 2)           │
                        │   • Notebook 05 (Artifact generation) │
                        └───────────────────────────────────────┘
```

---

## Detailed Component Design

### 1. Suporte a Múltiplas Classes-Alvo em `run_benchmark()` (`EXP-02`)

- **Interface**:
  - Novo parâmetro em `run_benchmark`: `target_classes: tuple[int, ...] | None = None`.
  - Se `target_classes is None`, mantém retrocompatibilidade usando `classes_to_run = (target_class_scenario1,)`.
  - Se fornecido (ex.: `target_classes=(0, 1, 2)`), itera sobre cada classe-alvo.
- **Isolamento e Rastreamento**:
  - `BenchmarkRunResult` ganha o campo `target_class: int | None = None`.
  - `csv_row` ganha a coluna `"target_class"` posicionada logo após `"scenario"`.
  - A partição de treino via `build_scenario1_split()` recebe `target_class=tc`, garantindo que positivos e negativos pertençam à classe avaliada.
  - O rótulo binário de teste é computado como `binary_test_lbl = (test_lbl_np == tc).astype(np.int64)`.
  - O resumo agregado no JSON inclui `classes_evaluated: list[int]` e métricas desagregadas por classe e consolidadas inter-classes.

### 2. Avaliação Multi-Razão e Interação Método x Razão (`EXP-01`)

- **Execução Multi-Razão**:
  - `imbalance_ratios` pode ser `("1:10", "1:50", "1:100")`.
  - Para cada razão $\rho \in \text{imbalance\_ratios}$, o subsampling positivo utiliza a fração canônica mapeada:
    - `"1:10"` -> `pos_fraction = 0.10` (~45 positivos)
    - `"1:50"` -> `pos_fraction = 0.02` (~9 positivos)
    - `"1:100"` -> `pos_fraction = 0.01` (~4-5 positivos)
- **Cálculo da Interação Método $\times$ Razão**:
  - Quando `len(imbalance_ratios) > 1` e sob estratégias compartilhadas (ex.: `mcts_enhanced` vs `uniform`):
    - Para pares de razões $(r_1, r_2)$ (ex.: `1:50` vs `1:10`, `1:50` vs `1:100`):
      - Invoca `compute_method_by_ratio_interaction(scores_r1_target, scores_r1_base, scores_r2_target, scores_r2_base)` de `src/utils/statistics.py`.
      - Exporta `mean_interaction`, `ci_95`, `cohens_d`, `t_pvalue` para cada par de razões em `res["ratio_interactions"]`.
  - `tier1_axis1_samplers.json` e `.csv` persistem a coluna e campo `imbalance_ratio` para cada linha e o dicionário `ratio_interactions` no bloco agregado.

### 3. Integração com Notebook 03 e Artefatos do Artigo (`EXP-03`)

- **Notebook 03 (Seção 2 - Axis 1)**:
  - Documenta e expõe os parâmetros `IMBALANCE_RATIOS_AXIS1` e `TARGET_CLASSES_AXIS1`.
  - Exibe a tabela comparativa agrupada por estratégia e razão.
  - Se `ratio_interactions` estiver presente, exibe a tabela de efeitos de interação método $\times$ razão.
- **Compatibilidade dos Loaders (`src/paper/loaders.py`)**:
  - Assegura que `load_real_benchmark_results()` leia os arquivos `tier1_axis1_samplers.json` com múltiplas razões e/ou múltiplas classes sem violar validações de esquema.
- **Notebook 05**:
  - Assegura que geradores de tabelas e gráficos lidem com múltiplas razões (gerando linhas/curvas por razão quando aplicável).

---

## Traceability to Requirements

| Requirement | Implementation Component | Verification Test |
| ----------- | ------------------------ | ----------------- |
| `EXP-01` | `src/training/benchmark_runner.py` + `src/utils/statistics.py` | `test_multi_ratio_execution_and_interaction` |
| `EXP-02` | `src/training/benchmark_runner.py` | `test_multi_target_class_execution_and_aggregation` |
| `EXP-03` | `notebooks/03_tier1_mcts_benchmarks.ipynb` + `src/paper/loaders.py` | `test_notebook_03_and_paper_artifacts_multi_ratio` |
