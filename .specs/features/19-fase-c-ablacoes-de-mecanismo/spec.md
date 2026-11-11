# F19: Fase C — Ablações de Mecanismo com Variantes Imutáveis

**Approved**: 2026-11-11 - user: "sim"

## Problem Statement

Com a conclusão e aprovação dos Gates A e B (integridade metodológica de splits, segregação de teste, observabilidade de exposição, calibração out-of-test e viabilidade do stress test 1:500), o roadmap revisado (`roadmap_mcts_hnm_proxima_run_v2.md`) estabelece a **Fase C (P1)** como a etapa de ablações científicas de mecanismo.

O objetivo da Fase C é isolar rigorosamente as variáveis da política MCTS e da amostragem de dados para entender quando e por que o MCTS ajuda, é neutro ou prejudica a classificação. Na literatura e nas execuções preliminares, a política MCTS Enhanced difere da amostragem Uniforme e do GiantHunter original por múltiplos fatores simultâneos, tornando ambíguas as atribuições de causa e efeito:

1. **Efeito do agrupamento focal no batch vs. efeito da política MCTS (Item C1)**: Quando o MCTS Enhanced seleciona uma classe focal e constrói um batch com 32 amostras dessa classe e 32 amostras uniformes, a distribuição de rótulos dentro do minibatch torna-se altamente concentrada (baixa diversidade de rótulos por batch). Isso levanta a dúvida: a degradação de acurácia multiclasse observada se deve a uma má seleção da árvore MCTS (ex.: starvation de classes raras) ou à perda de diversidade e representatividade estatística dentro dos minibatches de gradiente? Para responder a isso, é obrigatório implementar o controle **Uniform Grouped** (32 amostras de uma classe uniforme aleatória + 32 uniformes) e medir a diversidade de rótulos ($N_{\text{unique}}$, entropia $H_{\text{batch}}$) por batch.
2. **Conflito entre Normalização Min-Max de Q e Dual-Stream Batch (Item C2)**: O GiantHunter estrito utiliza valores brutos de $Q$ e batch 100% focado/negativo, enquanto o MCTS-HNM Aprimorado introduz conjuntamente normalização Min-Max de $Q$ em $[0, 1]$ e lote híbrido 50/50 Dual-Stream. É fundamental executar um desenho fatorial $2 \times 2$ completo e ortogonal:
   - **Strict-equivalente**: $Q$ bruto (`q_norm=False`), Dual-Stream ausente (`dual_stream_ratio=0.0`).
   - **QNorm-only**: $Q$ normalizado Min-Max (`q_norm=True`), Dual-Stream ausente (`dual_stream_ratio=0.0`).
   - **DualStream-only**: $Q$ bruto (`q_norm=False`), Dual-Stream ativo (`dual_stream_ratio=0.5`).
   - **Enhanced-equivalente**: $Q$ normalizado Min-Max (`q_norm=True`), Dual-Stream ativo (`dual_stream_ratio=0.5`).
3. **Valor da Hierarquia vs. Exploração vs. Dificuldade Amostral (Item C3)**: É necessário separar o ganho de uma árvore hierárquica e do bônus de exploração UCB em relação a heurísticas mais simples:
   - **Greedy por EMA de loss por classe**: Selecionador sem árvore e sem bônus UCB de exploração, amostrando avidamente a classe com maior perda móvel.
   - **Flat UCB**: Selecionador com exploração UCB1, porém sem hierarquia (as 100 classes atuam diretamente como braços de um multi-armed bandit).
4. **Influência da Topologia da Árvore (Item C5)**: Comparação estruturada entre hierarquias (Semântica Nativa, Clustering de Ward, Average Linkage e Árvore Aleatória de controle com topologia equiparável), mensurando métricas estruturais (profundidade máxima, profundidade média, número de nós e ramificação média).
5. **Gate C**: Verificação automatizada de que todas as ablações isolam estritamente seus fatores controlados, operando sob hiperparâmetros idênticos (mesmas seeds, mesmos pesos iniciais, mesma taxa de aprendizado e mesmo orçamento de passos).

---

## Out of Scope

- Executar a campanha final completa com todas as 10 seeds e 5.000 passos (escopo da Fase D).
- Modificar o conjunto de teste oficial ou reintroduzir dados sintéticos em modo REAL.
- Implementar novas cotas de exploração artificiais não descritas no roadmap.
- Otimizar hiperparâmetros no conjunto de teste.

---

## Assumptions & Open Questions

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| Uniform Grouped Batch | Minibatch de 64 composto por 32 amostras de uma classe aleatória uniforme + 32 uniformes gerais | Isola o efeito de homogeneidade do lote da política de seleção MCTS | y |
| Métrica de diversidade de lote | Contagem de classes distintas $N_{\text{unique}}$ e entropia de Shannon dos rótulos no lote $H_{\text{batch}}$ | Quantifica quantitativamente a concentração de rótulos por gradiente | y |
| Grade Fatorial $2 \times 2$ | 4 estratégias imutáveis: `mcts_strict`, `mcts_qnorm_only`, `mcts_dualstream_only`, `mcts_enhanced` | Desacopla matematicamente a normalização de Q da composição do lote híbrido | y |
| Controles de seleção simples | `greedy_loss_ema` (amostragem greedy da classe com maior EMA de loss) e `flat_ucb` (UCB1 com 100 braços sem árvore) | Separa o impacto da hierarquia em árvore da exploração UCB | y |
| Árvore Aleatória de Controle | Construção de `FastHierarchyTree` aleatória balanceada de 100 folhas com profundidade matching à semântica | Serve como controle nulo para testar a hipótese de relevância semântica | y |
| Gate C Verifier | Script automatizado que audita a equivalência estrita de fatores entre execuções pareadas | Garante ausência de variáveis de confusão ou desvios ocultos de hiperparâmetros | y |

**Open questions:** none - all ablation designs follow the requirements in `roadmap_mcts_hnm_proxima_run_v2.md` lines 120-162.

---

## User Stories

### P1: Controle Uniform Grouped e Diversidade de Batch (C1) (`ABLA-01`) ⭐ MVP

**User Story**: Como pesquisador investigando o desempenho multiclasse, quero comparar o treinamento sob amostragem Uniform, Uniform Grouped e MCTS Enhanced, medindo a diversidade de classes por batch, para identificar se a degradação advém da homogeneidade dos minibatches.

**Why P1**: Requisito C1 mandatório do roadmap para responder à principal hipótese causal sobre minibatches focais no Cenário 2.

**Acceptance Criteria**:

1. WHILE training under Scenario 2 with strategy `uniform_grouped`, WHEN sampling a batch of size $B$ with dual-stream ratio $\beta = 0.5$, the system SHALL draw $B/2$ samples from a single uniformly chosen focal class and $B/2$ samples uniformly from the entire training dataset.
2. WHILE training under Scenario 2 for any sampling strategy, WHEN a batch of labels is sampled, the system SHALL compute batch diversity metrics including the number of unique classes $N_{\text{unique}}$ and the Shannon entropy of batch labels $H_{\text{batch}}$.
3. WHEN a training run completes, the system SHALL record in `run_metadata` and output summary the mean batch diversity metrics ($N_{\text{unique}}$, $H_{\text{batch}}$) and cumulative class coverage $C_{\text{cov}} \in [0.0, 1.0]$.
4. IF `dual_stream_ratio` is not in $[0.0, 1.0]$ WHEN configuring `uniform_grouped`, the system SHALL raise a `ValueError`.

**Independent Test**: Executar testes em `tests/test_phase_c_ablations.py` verificando a composição do lote em `uniform_grouped`, os cálculos de diversidade de batch e o registro nas métricas.

---

### P1: Fatorial $2 \times 2$ entre Normalização de Q e Dual-Stream (C2) (`ABLA-02`) ⭐ MVP

**User Story**: Como pesquisador de aprendizado por reforço, quero treinar as 4 variantes ortogonais da matriz $2 \times 2$ (Strict, QNorm-only, DualStream-only, Enhanced) sob condições rigorosamente idênticas, para isolar a contribuição individual de cada inovação proposta.

**Why P1**: Requisito C2 mandatório para comprovar empiricamente a legitimidade científica das modificações do MCTS-HNM sobre o GiantHunter.

**Acceptance Criteria**:

1. WHERE strategy is `mcts_strict`, the system SHALL set `q_norm=False` and `dual_stream_ratio=0.0`.
2. WHERE strategy is `mcts_qnorm_only`, the system SHALL set `q_norm=True` and `dual_stream_ratio=0.0`.
3. WHERE strategy is `mcts_dualstream_only`, the system SHALL set `q_norm=False` and `dual_stream_ratio=0.5`.
4. WHERE strategy is `mcts_enhanced`, the system SHALL set `q_norm=True` and `dual_stream_ratio=0.5`.
5. WHILE executing any of the four factorial variants under identical seeds, the system SHALL ensure that initial model weights, learning rate, momentum $\alpha$, exploration constant $c$, and step budget remain strictly identical.

**Independent Test**: Executar testes em `tests/test_phase_c_ablations.py` validando os parâmetros de cada uma das 4 variantes e a invariância dos fatores mantidos constantes.

---

### P1: Controles Simples de Seleção: Greedy por EMA de Loss e Flat UCB (C3) (`ABLA-03`) ⭐ MVP

**User Story**: Como pesquisador, quero avaliar seletores de controle simples (Greedy por EMA de loss por classe e Flat UCB sem hierarquia), para discernir o valor intrínseco da estrutura de árvore em relação à exploração pura e dificuldade amostral.

**Why P1**: Requisito C3 para responder se a hierarquia em árvore agrega valor mensurável frente a um bandit plano de 100 classes ou a uma seleção puramente gulosa.

**Acceptance Criteria**:

1. WHERE strategy is `greedy_loss_ema`, the system SHALL maintain an exponential moving average (EMA) of loss for each class and greedily select the class with highest EMA loss without exploration bonus ($c = 0$).
2. WHERE strategy is `flat_ucb`, the system SHALL maintain visit counts and Q-values for all $K$ classes without tree hierarchy, selecting the class with highest UCB1 score.
3. WHILE training with `greedy_loss_ema` or `flat_ucb`, the system SHALL update class statistics at each step from the evaluated minibatch loss.
4. IF a class has zero visits in `flat_ucb`, the system SHALL assign infinite priority to ensure all classes are sampled at least once before exploitation.

**Independent Test**: Executar testes em `tests/test_phase_c_selectors.py` verificando a seleção determinística da classe de maior perda em `greedy_loss_ema` e a política UCB sem hierarquia em `flat_ucb`.

---

### P1: Comparação Estrutural de Hierarquias (C5) (`ABLA-05`) ⭐ MVP

**User Story**: Como pesquisador, quero construir e caracterizar hierarquias alternativas (Semântica Nativa, Clustering de Ward, Average Linkage e Árvore Aleatória balanceada), registrando métricas de topologia, para comparar o impacto da estrutura hierárquica.

**Why P1**: Requisito C5 para mensurar a topologia de cada árvore e prover o baseline nulo (árvore aleatória) para testes de hipótese.

**Acceptance Criteria**:

1. WHEN constructing any `FastHierarchyTree`, the system SHALL provide a function `compute_tree_structural_metrics(tree)` returning maximum depth, mean leaf depth, total node count, leaf count, and mean branching factor.
2. WHEN building a Random Hierarchy with $K$ classes, the system SHALL construct a valid `FastHierarchyTree` with $K$ leaves, parent-child invariants intact, and depth profile matching the Semantic tree.
3. WHEN building trees via Average Linkage (UPGMA), the system SHALL verify that dissimilarity entries are finite non-negative values and produce a strictly valid hierarchy tree.

**Independent Test**: Executar testes em `tests/test_phase_c_hierarchies.py` validando métricas topológicas e a integridade estrutural da Árvore Aleatória e UPGMA.

---

### P1: Verificador do Gate C (Isolamento de Variáveis) (`GATE-03`) ⭐ MVP

**User Story**: Como Tech Lead, quero executar um verificador automatizado `verify_gate_c()` que valida a ausência de variáveis de confusão, a equivalência de parâmetros fixados e a completude diagnóstica em todas as variantes da Fase C.

**Why P1**: Requisito do Gate C para autorizar formalmente o congelamento do protocolo da Fase D.

**Acceptance Criteria**:

1. WHEN `verify_gate_c()` is called on a collection of ablation experiment runs, the system SHALL verify that for every paired comparison, all non-ablated parameters (learning rate, optimizer, architecture, seed, step count) are strictly identical.
2. IF any ablation run contains non-finite metrics (`NaN` or `Inf`) or missing batch diversity telemetry, `verify_gate_c()` SHALL raise a `ValueError` or report a failure verdict.
3. WHEN comparing `uniform`, `uniform_grouped`, and `mcts_enhanced`, the system SHALL output a structured diagnostic verdict classifying the primary driver of performance differences between batch homogeneity and policy starvation.

**Independent Test**: Executar testes em `tests/test_phase_c_gate.py` verificando a aprovação do Gate C sob execuções válidas e rejeição imediata sob parâmetros divergentes ou dados ausentes.

---

## Requirement Traceability

| Requirement ID | Description | Component | Verification Method | Status |
| -------------- | ----------- | --------- | ------------------- | ------ |
| ABLA-01 | Controle Uniform Grouped e métricas de diversidade de lote | `src/training/trainer.py`, `src/training/exposure.py` | `tests/test_phase_c_ablations.py` | pending |
| ABLA-02 | Grade fatorial 2x2 entre normalização de Q e Dual-Stream | `src/training/trainer.py`, `src/mcts/agent.py` | `tests/test_phase_c_ablations.py` | pending |
| ABLA-03 | Controles simples de seleção: Greedy por EMA de loss e Flat UCB | `src/training/selectors.py`, `src/training/trainer.py` | `tests/test_phase_c_selectors.py` | pending |
| ABLA-05 | Métricas estruturais de árvores e baseline de Árvore Aleatória | `src/trees/hierarchy_metrics.py`, `src/trees/random_tree.py` | `tests/test_phase_c_hierarchies.py` | pending |
| GATE-03 | Verificador do Gate C para isolamento de variáveis | `src/utils/gate_c_verifier.py` | `tests/test_phase_c_gate.py` | pending |

