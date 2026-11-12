# F23: P1 — Ablações de Mecanismo (Strict vs Enhanced) e Comparação Estatística das Hierarquias

**Approved**: 2026-11-11 - user: "sim"

## Problem Statement

A análise dos resultados anteriores revelou duas lacunas metodológicas críticas classificadas como P1:
1. **Ablação Strict vs Enhanced sem isolamento fatorial completo**:
   Historicamente, o `mcts_enhanced` superou o baseline uniforme no Scenario 1, mas diferiu do `mcts_strict` original (GiantHunter) em dois eixos simultâneos: normalização Min-Max de $Q$ (estabilidade contra bounds não-estacionários de recompensa) e lote híbrido Dual-Stream 50/50 (mitigação de esquecimento catastrófico e colapso de gradiente). Sem um grid fatorial $2 \times 2$ controlado (`Strict-equivalente`, `QNorm-only`, `DualStream-only`, `Enhanced-equivalente`) com orçamento e medições explícitas de custo (forwards de mineração, gradient samples, wall-clock time), é impossível atribuir causalmente o ganho de desempenho a cada componente ou identificar interações sinérgicas.
2. **Comparação de hierarquias (Semantic vs Ward) sem significância pareada**:
   No Axis 2 (Scenario 1), as execuções anteriores mostraram flutuações nas médias agregadas entre a Árvore Semântica (AUPRC 0.6848, F1 0.6797) e a Árvore Orientada por Confusão via Ward (AUPRC 0.6749, F1 0.6469). A diferença observada é pequena frente à dispersão amostral por semente. Declarar a superioridade de qualquer uma das hierarquias exige uma comparação estatística estritamente pareada seed a seed ($\Delta_{\text{AUPRC}}$, $\Delta_{\text{F1}}$, intervalo de confiança de 95%, Cohen's d, Student's t pareado e teste de Wilcoxon), além do registro de métricas topológicas das árvores.

Para resolver este P1, é necessário formalizar e executar o grid fatorial $2 \times 2$ de ablação com rastreamento de custos computacionais, implementar a comparação estatística pareada rigorosa no Axis 2 de hierarquias, e atualizar os relatórios do Notebook 03 e testes automatizados.

## Goals

- [ ] Habilitar e padronizar o grid fatorial $2 \times 2$ de ablação de mecanismos (`mcts_strict`, `mcts_qnorm_only`, `mcts_dualstream_only`, `mcts_enhanced`) em `run_benchmark()` tanto no Scenario 1 quanto no Scenario 2, registrando métricas de custo computacional (amostras no gradiente, forwards de seleção, wall-clock time e updates).
- [ ] Implementar a comparação inferencial pareada seed a seed no Axis 2 entre Árvore Semântica e Ward Confusion ($\Delta_{\text{AUPRC}}$, $\Delta_{\text{F1}}$, IC 95%, Cohen's d, t-test e Wilcoxon), exportando `tier1_axis2_tree_ablation.json` com matriz de testes pareados.
- [ ] Registrar métricas estruturais e topológicas das árvores avaliadas (profundidade máxima, profundidade média das folhas, número de nós internos, contagem de folhas, balanceamento).
- [ ] Integrar ao Notebook 03 (Seções 3 e 4) a exibição do grid fatorial $2 \times 2$ com tabela de efeitos principais/interação e a tabela inferencial pareada Semantic vs Ward.
- [ ] Adicionar suíte de testes automatizados de regressão (`tests/test_p1_ablations_and_hierarchies.py`) validando todas as variantes de ablação, pareamento de sementes e consistência dos testes de hipótese.

## Out of Scope

- Execução não-supervisionada de campanhas pesadas em modo REAL com 5 sementes completas de 2000 passos (as execuções reais completas são disparadas sob demanda; os testes automatizados operam em modo rápido / SMOKE com SLA < 60s).
- Alteração da definição dos splits 1:50 e 4-quadrantes do Scenario 1 (congelados na Feature 21).
- Modificação dos baselines históricos existentes.

---

## Assumptions & Open Questions

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| Grid Fatorial de Mecanismos | 4 variantes canônicas: `mcts_strict` (Q bruto, 100% MCTS), `mcts_qnorm_only` (Q-norm, 100% MCTS), `mcts_dualstream_only` (Q bruto, 50% MCTS / 50% Uni), `mcts_enhanced` (Q-norm, 50% MCTS / 50% Uni) | Isola perfeitamente os efeitos principais de Q-norm e Dual-Stream e sua interação | y |
| Métrica Primária Axis 2 | AUPRC para ranking e F1 calibrado (4-quadrantes) para classificação binária | AUPRC é invariante ao limiar de decisão sob desbalanceamento severo | y |
| Testes Pareados Axis 2 | Student's t-test pareado, Wilcoxon signed-rank test, Cohen's d e IC 95% das diferenças pareadas seed a seed | Elimina a variância inter-semente e testa formalmente a equivalência vs superioridade | y |
| Controles de Custo | `total_updates`, `gradient_samples`, `mining_forwards`, `elapsed_seconds` | Permite comparar a eficiência real de cada variante além da acurácia bruta | y |

**Open questions:** none - escopo e metodologia de ablação $2 \times 2$ e testes pareados acordados com o usuário.

---

## User Stories

### P1: Grid Fatorial $2 \times 2$ de Mecanismos e Controles de Custo (`ABL-01`) ⭐ MVP

**User Story**: Como pesquisador de aprendizado por reforço, quero avaliar o grid fatorial $2 \times 2$ completo (`mcts_strict`, `mcts_qnorm_only`, `mcts_dualstream_only`, `mcts_enhanced`) com métricas explícitas de custo computacional em `run_benchmark()` para determinar se o ganho de desempenho decorre da estabilização de Q, da composição do batch, ou de ambos.

**Why P1**: Isola causalmente os dois mecanismos que diferenciam o GiantHunter estrito da versão aprimorada.

**Acceptance Criteria**:

1. WHEN `run_benchmark()` executes with strategies in the $2 \times 2$ factorial grid, THEN the system SHALL train and evaluate each variant under identical initializations, step budgets, and seeds.
2. The system SHALL record computational cost metrics for each run: `total_updates`, `gradient_samples`, `mining_forwards`, and `elapsed_seconds`.
3. The system SHALL compute main effects (marginal effect of Q-norm, marginal effect of Dual-Stream) and their interaction term $\Delta_{\text{interaction}} = (\text{Enhanced} - \text{QNorm}) - (\text{DualStream} - \text{Strict})$.
4. IF any strategy in the grid is executed, THEN the system SHALL export the results to JSON and CSV with `.meta.json` provenance sidecars.

**Independent Test**: Executar `run_benchmark` com as 4 variantes fatoriais em SMOKE mode e verificar cálculo dos efeitos principais e custos computacionais no payload exportado.

---

### P1: Comparação Estatística Pareada de Hierarquias no Axis 2 (`HIER-01`)

**User Story**: Como revisor de artigo científico, quero uma comparação estatística pareada seed a seed entre a Árvore Semântica e a Árvore de Confusão Ward ($\Delta_{\text{AUPRC}}$, $\Delta_{\text{F1}}$, Cohen's d, t-test pareado e Wilcoxon), acompanhada de diagnósticos topológicos, para comprovar rigorosamente se existe diferença estatisticamente significativa entre as árvores.

**Why P1**: Elimina conclusões prematuras baseadas em médias agregadas sem controle de dispersão pareada.

**Acceptance Criteria**:

1. WHEN `run_benchmark()` executes Axis 2 (hierarchy ablation) with `tree_types=("semantic", "confusion_ward")`, THEN the system SHALL compute paired differences seed a seed for identical splits.
2. The system SHALL calculate and report: mean difference, standard error, 95% confidence interval, Cohen's d, Student's t p-value, and Wilcoxon signed-rank p-value for both AUPRC and F1.
3. The system SHALL record topological tree diagnostics for each tree type: `max_depth`, `mean_leaf_depth`, `total_nodes`, `num_leaves`, and branching factors.
4. The system SHALL persist these paired comparisons in `tier1_axis2_tree_ablation.json` and report them in the consolidated summary.

**Independent Test**: Executar Axis 2 com 2 sementes e verificar que a matriz de diferenças pareadas e testes de Wilcoxon/t-test são calculados e persistidos no JSON de saída.

---

### P1: Atualização do Notebook 03 com Tabelas Inferenciais de Ablação e Hierarquias (`NOTE-01`)

**User Story**: Como leitor do notebook experimental, quero visualizar na Seção 3 e na Seção de Ablação tabelas inferenciais consolidadas exibindo o fatorial $2 \times 2$ de mecanismos e os testes pareados Semantic vs Ward para inspecionar visualmente os resultados e diagnósticos.

**Why P1**: Assegura que o documento executável do experimento reflita as evidências estatísticas completas.

**Acceptance Criteria**:

1. Section 3 of Notebook 03 SHALL display the paired comparison table between Semantic and Ward Confusion, reporting deltas, Cohen's d, and p-values.
2. The notebook SHALL present the $2 \times 2$ factorial mechanism ablation table highlighting main effects and interactions.
3. Notebook 03 SHALL execute end-to-end without errors in SMOKE mode with exit code 0.

**Independent Test**: Executar teste de fumaça do Notebook 03 validando a presença das tabelas e estatísticas pareadas.

---

## Edge Cases

- IF fewer than 2 seeds are provided, THEN inferential statistical tests (t-test, Wilcoxon) SHALL NOT execute, but mean pairwise differences SHALL still be computed.
- IF tree topology contains singleton root or uniform depth, THEN topological metrics SHALL return depth values without raising division-by-zero.
- IF all differences between two methods are zero, THEN Cohen's d SHALL be safely bounded to 0.0 without NaN propagation.

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| -------------- | ----- | ----- | ------ |
| ABL-01 | P1: Grid Fatorial $2 \times 2$ de Mecanismos e Controles de Custo | Tasks | Pending |
| HIER-01 | P1: Comparação Estatística Pareada de Hierarquias no Axis 2 | Tasks | Pending |
| NOTE-01 | P1: Atualização do Notebook 03 com Tabelas Inferenciais | Tasks | Pending |

**Coverage:** 3 total, 3 mapped to tasks, 0 unmapped.

---

## Success Criteria

- [ ] Grid fatorial $2 \times 2$ de ablações (`mcts_strict`, `mcts_qnorm_only`, `mcts_dualstream_only`, `mcts_enhanced`) executável em `run_benchmark()` com registro de custos computacionais.
- [ ] Axis 2 calcula testes inferenciais pareados seed a seed (t-test, Wilcoxon, Cohen's d, IC 95%) entre Semantic e Ward.
- [ ] Diagnósticos topológicos das árvores registrados (`max_depth`, `mean_leaf_depth`, `total_nodes`, `num_leaves`).
- [ ] Notebook 03 Seção 3 e Seção de Ablação integradas com tabelas comparativas inferenciais.
- [ ] Suíte de testes automatizados passa com 100% de aprovação.

