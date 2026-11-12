# F22: P0 — Investigação da Cobertura no Scenario 2 e Variantes de Amostragem Design

## System Architecture Overview

Esta feature investiga e resolve a discrepância causal no Scenario 2 (100-way Multi-Class Curriculum), implementando o controle `uniform_grouped`, as ablações de amostragem e a nova política corretiva `mcts_coverage_balanced`, acompanhadas de diagnósticos quantitativos por classe e testes pareados.

```
                          ┌──────────────────────────────────────────────┐
                          │          Scenario 2 Multi-Class Grid         │
                          │            (100 classes CIFAR-100)           │
                          └──────────────────────┬───────────────────────┘
                                                 │
          ┌──────────────────────────────────────┼──────────────────────────────────────┐
          │                                      │                                      │
          ▼                                      ▼                                      ▼
┌──────────────────┐                  ┌──────────────────┐                  ┌──────────────────┐
│ Uniform Baseline │                  │ Uniform Grouped  │                  │  MCTS Enhanced   │
│  (64 uniform)    │                  │  (32 focal rand  │                  │ (32 focal MCTS   │
│                  │                  │   + 32 uniform)  │                  │   + 32 uniform)  │
└─────────┬────────┘                  └─────────┬────────┘                  └─────────┬────────┘
          │                                      │                                      │
          └──────────────────────────────────────┼──────────────────────────────────────┘
                                                 │
                                                 ▼
                              ┌──────────────────────────────────────┐
                              │  MCTS Coverage Balanced (Corretiva)  │
                              │     (16 focal MCTS + 48 uniform)     │
                              │    (Fração focal atenuada em 25%)    │
                              └──────────────────┬───────────────────┘
                                                 │
                                                 ▼
                              ┌──────────────────────────────────────┐
                              │     Per-Class Quantitative Telemetry │
                              ├──────────────────────────────────────┤
                              │ • Normalized Entropy H_norm          │
                              │ • Class Coverage C_cov               │
                              │ • Unvisited Classes N_unvisited      │
                              │ • Worst-10 Class Recall              │
                              │ • Exposure-Recall Spearman Corr.     │
                              └──────────────────┬───────────────────┘
                                                 │
                                                 ▼
                              ┌──────────────────────────────────────┐
                              │  Paired Inferential Causal Report    │
                              │   (Delta vs Uniform, Cohen's d,      │
                              │    Notebook 03 Comparative Table)    │
                              └──────────────────────────────────────┘
```

---

## Detailed Component Design

### 1. Política Corretiva `mcts_coverage_balanced` (`src/training/trainer.py`)

- **Estratégia**:
  - `SUPPORTED_STRATEGIES` é estendida com `"mcts_coverage_balanced"`.
  - Em `train_scenario2()`, quando `strategy == "mcts_coverage_balanced"`:
    - O batch size é particionado com `dual_stream_ratio = 0.25`:
      $K_{\text{mcts}} = \lfloor 64 \times 0.25 \rfloor = 16$ instâncias da classe minerada pelo MCTS.
      $K_{\text{uniform}} = 64 - 16 = 48$ instâncias sorteadas uniformemente pelo dataset.
    - Isso atenua a homogeneidade artificial do batch em 50% em comparação ao `mcts_enhanced` ($32/32$), mitigando o viés do gradiente nas 100 classes enquanto preserva o sinal de mineração hierárquico com Q-normalização Min-Max e exploração UCB.
  - A estratégia histórica `mcts_enhanced` permanece estritamente inalterada com `dual_stream_ratio = 0.5` ($32/32$).

### 2. Suporte Completo de Estratégias em `run_benchmark(scenario="scenario2")` (`src/training/benchmark_runner.py`)

- Suporte completo às 7 estratégias:
  1. `uniform`: baseline uniforme padrão (batch 64 uniformes).
  2. `uniform_grouped`: controle de homogeneidade (32 de classe uniforme aleatória + 32 uniformes).
  3. `mcts_strict`: MCTS sem Q-norm (Q bruto) e lote 100% MCTS.
  4. `mcts_qnorm_only`: MCTS com Q-norm Min-Max e lote 100% MCTS.
  5. `mcts_dualstream_only`: MCTS com Q bruto e lote dual-stream 50/50.
  6. `mcts_enhanced`: MCTS Enhanced histórico (Q-norm Min-Max + dual-stream 50/50).
  7. `mcts_coverage_balanced`: MCTS com Q-norm Min-Max e dual-stream balanceado 25/75.

### 3. Diagnósticos Quantitativos e Correlação Exposição–Recall (`src/training/benchmark_runner.py`)

Para cada execução do Scenario 2, o payload registra:
- `top1_accuracy`, `macro_f1`, `balanced_accuracy`, `worst_10_class_recall`
- `exposure_entropy`: $H_{\text{norm}} = -\frac{1}{\ln K} \sum_{k=1}^K p_k \ln p_k$
- `class_coverage`: fração de classes com exposição $> 0$
- `unvisited_classes`: contagem de classes com exposição $= 0$
- `exposure_recall_correlation`: correlação de postos de Spearman entre as contagens de amostragem por classe e os recalls de teste correspondentes.
- Análise pareada seed a seed contra o baseline `uniform`: $\Delta_{\text{acc}}$, $\Delta_{\text{macro\_f1}}$, $\Delta_{\text{worst10}}$.

### 4. Integração no Notebook 03 (`notebooks/03_tier1_mcts_benchmarks.ipynb`)

- A Seção 4 (Scenario 2) executa o conjunto de estratégias:
  `strategies = ("uniform", "uniform_grouped", "mcts_enhanced", "mcts_coverage_balanced")`
- Emite tabela comparativa rica:
  | Strategy | Top-1 Acc | Macro-F1 | Worst-10 Recall | Entropy $H_{\text{norm}}$ | Unvisited | Corr(Exp, Rec) |
- Imprime o parecer causal:
  - Se `uniform_grouped` $\approx$ `uniform`, a perda decorre da política de busca do MCTS.
  - Se `uniform_grouped` $\approx$ `mcts_enhanced`, a perda decorre da saturação por classe nos minibatches.
  - Avalia o ganho proporcionado por `mcts_coverage_balanced`.

---

## Traceability to Requirements

| Requirement | Implementation Component | Verification Target |
| ----------- | ------------------------ | ------------------- |
| `S2-01` | `src/training/benchmark_runner.py` & `trainer.py` | Executa todas as 7 variantes no Scenario 2 |
| `S2-02` | `src/training/trainer.py` (`mcts_coverage_balanced`) | Minibatch 16/48 com Q-norm Min-Max |
| `S2-03` | `src/training/benchmark_runner.py` | Métricas $H_{\text{norm}}$, $C_{\text{cov}}$, $N_{\text{unvisited}}$ e Spearman corr |
| `S2-04` | `notebooks/03_tier1_mcts_benchmarks.ipynb` | Tabela comparativa e diagnóstico causal |
