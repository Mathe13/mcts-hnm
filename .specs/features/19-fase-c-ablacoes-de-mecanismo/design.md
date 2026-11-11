# F19: Fase C — Ablações de Mecanismo com Variantes Imutáveis Design

## System Architecture Overview

A Fase C expande o sistema de amostragem, os seletores de currículo e a análise topológica de hierarquias para realizar as ablações científicas exigidas pelo roadmap (`roadmap_mcts_hnm_proxima_run_v2.md`).

```
                    ┌────────────────────────────────────────────────────────┐
                    │                   Downstream Trainer                   │
                    │         (Scenario 1 Binary / Scenario 2 Multi)         │
                    └───────────────────────────┬────────────────────────────┘
                                                │
         ┌──────────────────────────────────────┼──────────────────────────────────────┐
         │                                      │                                      │
         ▼                                      ▼                                      ▼
┌──────────────────┐                  ┌──────────────────┐                  ┌──────────────────┐
│  Samplers Layer  │                  │  Selectors Layer │                  │ Structural Layer │
├──────────────────┤                  ├──────────────────┤                  ├──────────────────┤
│• UniformSampler  │                  │• MCTSHNMAgent    │                  │• HierarchyMetrics│
│• UniformGrouped  │                  │  (Strict/Enhanced│                  │  (Depth, Branch) │
│• DualStream      │                  │   QNorm/DualStr) │                  │• RandomTree      │
│  (0.0 vs 0.5)    │                  │• GreedyLossEMA   │                  │  (Topology match)│
│• BatchDiversity  │                  │• FlatUCB (NoTree)│                  │• UPGMA Tree      │
└──────────────────┘                  └──────────────────┘                  └──────────────────┘
         │                                      │                                      │
         └──────────────────────────────────────┼──────────────────────────────────────┘
                                                │
                                                ▼
                               ┌──────────────────────────────────┐
                               │         Gate C Verifier          │
                               │  (Isolates variables, validates  │
                               │   telemetry, classifies causes)  │
                               └──────────────────────────────────┘
```

---

## Detailed Component Design

### 1. Telemetria de Diversidade de Minibatch (`src/training/batch_diversity.py`)

Mede quantitativamente a concentração de rótulos por gradiente em cada passo:

- **Fórmula de Entropia de Lote**:
  $$H_{\text{batch}} = -\sum_{c \in \mathcal{C}_{\text{batch}}} \hat{p}_c \ln(\hat{p}_c)$$
  onde $\hat{p}_c = \frac{\text{count}(c)}{B}$ é a proporção empírica da classe $c$ no minibatch de tamanho $B$.
- **Contagem de Classes Únicas**:
  $$N_{\text{unique}} = |\mathcal{C}_{\text{batch}}|$$
- **Função Principal**:
  ```python
  def compute_batch_diversity(
      labels: np.ndarray | torch.Tensor,
      num_classes: int = 100,
  ) -> dict[str, float]:
      """Returns {'unique_classes': float, 'batch_entropy': float}."""
  ```

### 2. Amostrador Uniform Grouped (`src/mcts/sampler.py`)

Implementa o controle experimental $C1$ para isolar a homogeneidade do lote:

- **Classe**: `UniformGroupedBatchSampler`
- **Comportamento**:
  1. A cada chamada de `sample_batch()`, sorteia uma classe focal $c_{\text{focal}} \sim \text{Uniform}(\{0, \dots, K-1\})$.
  2. Amostra $K_{\text{focal}} = \text{round}((1 - \beta) B)$ instâncias da classe focal $c_{\text{focal}}$ sem reposição (ou com reposição se o pool da classe for menor que $K_{\text{focal}}$).
  3. Amostra $K_{\text{uni}} = B - K_{\text{focal}}$ instâncias uniformemente do pool global de treino.
  4. Retorna uma instância de `DualStreamBatch` compatível com o pipeline de treino existente.

### 3. Grade Fatorial $2 \times 2$ (Q-normalization $\times$ Dual-Stream)

Desacopla formalmente as duas inovações do MCTS-HNM no `trainer.py` e `runner_config.py`:

| Estratégia | `q_norm` | `dual_stream_ratio` ($\beta$) | Composição do Batch |
|---|---|---|---|
| `mcts_strict` | `False` | `0.0` | 100% instâncias da classe selecionada pelo MCTS |
| `mcts_qnorm_only` | `True` | `0.0` | 100% instâncias da classe selecionada pelo MCTS |
| `mcts_dualstream_only` | `False` | `0.5` | 50% classe MCTS + 50% amostragem uniforme |
| `mcts_enhanced` | `True` | `0.5` | 50% classe MCTS + 50% amostragem uniforme |

- **Invariância de Fatores Controlados**:
  - Mesma inicialização de pesos (`set_global_seed(seed)`).
  - Mesma taxa de aprendizado ($1 \times 10^{-3}$), otimizador Adam, momentum $\alpha = 0.3$, constante de exploração $c = 2.0$.
  - Mesmo orçamento de passos ($N = 500$ ou $N = 5000$).

### 4. Controles Simples de Seleção (`src/training/selectors.py`)

- **`GreedyLossEMASelector`**:
  - Mantém vetor $\mathbf{m} \in \mathbb{R}^K$ de perdas médias por classe com fator de amortecimento $\gamma = 0.1$:
    $$m_c^{(t)} = (1 - \gamma) m_c^{(t-1)} + \gamma \bar{\ell}_c^{(t)}$$
  - Seleção gulosa pura: $c^* = \arg\max_c m_c$.
  - Sem bônus de exploração e sem hierarquia em árvore.
- **`FlatUCBSelector`**:
  - Trata as 100 classes como braços independentes de um multi-armed bandit.
  - Para cada classe $c$, mantém contagem de visitas $n_c$ e recompensa média acumulada $Q_c$.
  - Se $\exists c \text{ com } n_c = 0$, seleciona prioritariamente o braço não visitado.
  - Caso contrário, seleciona via UCB1 padrão:
    $$c^* = \arg\max_c \left[ Q_c + c_{\text{ucb}} \sqrt{\frac{\ln(\sum_j n_j)}{n_c}} \right]$$
  - Sem hierarquia em árvore.

### 5. Métricas Estruturais e Baseline de Árvore Aleatória (`src/trees/hierarchy_metrics.py`, `src/trees/random_tree.py`)

- **`compute_tree_structural_metrics(tree: FastHierarchyTree) -> dict[str, float]`**:
  - Calcula:
    - `max_depth`: maior distância da raiz até qualquer folha.
    - `mean_leaf_depth`: profundidade média ponderada das folhas.
    - `total_nodes`: $N$ nós totais.
    - `num_leaves`: $K$ folhas (100 para CIFAR-100).
    - `mean_branching_factor`: média de filhos por nó interno.
- **`build_random_tree(class_to_indices: dict[int, np.ndarray], seed: int = 42) -> FastHierarchyTree`**:
  - Constrói uma hierarquia aleatória particionando as 100 classes em 20 superclasses aleatórias com 5 classes cada, preservando exatamente a topologia da Árvore Semântica do CIFAR-100 para isolar a relevância do agrupamento semântico versus topologia neutra.

### 6. Verificador do Gate C (`src/utils/gate_c_verifier.py`)

- **Função**: `verify_gate_c(ablation_runs: list[dict[str, Any]]) -> dict[str, Any]`
- **Auditorias Executadas**:
  1. **Isolamento de Variáveis**: Confirma que comparações pareadas diferem única e exclusivamente no fator testado.
  2. **Completude Diagnóstica**: Garante presença de `batch_entropy`, `unique_classes`, `step_history` e `exposure_diagnostics`.
  3. **Estabilidade Numérica**: Garante ausência de `NaN` ou `Inf`.
  4. **Classificação Causal**:
     - Calcula a diferença de acurácia $\Delta_{\text{batch}} = \text{Acc}(\text{Uniform}) - \text{Acc}(\text{UniformGrouped})$.
     - Calcula a diferença de política $\Delta_{\text{policy}} = \text{Acc}(\text{UniformGrouped}) - \text{Acc}(\text{Enhanced})$.
     - Emite veredito fundamentado sobre o fator preponderante.
