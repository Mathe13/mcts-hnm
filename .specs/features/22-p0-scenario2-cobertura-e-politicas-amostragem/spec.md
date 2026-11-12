# F22: P0 — Investigação da Cobertura no Scenario 2 e Variantes de Amostragem

**Approved**: 2026-11-11 - user: "sim"

## Problem Statement

No Scenario 2 (100-way Multi-Class Curriculum), os resultados empíricos demonstraram um colapso severo de desempenho do MCTS multiclasse em relação ao baseline uniforme:
- Uniform: Top-1 Accuracy 63,85%, Macro-F1 63,72%, Worst-10 recall 33,46%
- MCTS Enhanced: Top-1 Accuracy 33,43%, Macro-F1 34,26%, Worst-10 recall 0,68%

Embora a infraestrutura de telemetria já colete visitas, entropia, cobertura e recall por classe, o runner e o Notebook 03 permaneciam restritos a uma comparação bivariada entre `uniform` e `mcts_enhanced`. Essa restrição impedia responder à pergunta causal central formulada no roadmap: a degradação decorre da **homogeneidade dos minibatches** (efeito da concentração de 32 amostras em uma única classe focal no batch) ou da **política MCTS / inanição de classes** (starvation e baixa entropia de exploração)?

Para resolver este P0, é necessário:
1. Executar as variantes de controle e ablação propostas na Fase C (`uniform_grouped`, `mcts_strict`, `mcts_qnorm_only`, `mcts_dualstream_only`) e uma política corretiva de cobertura (`mcts_coverage_balanced`), preservando o `mcts_enhanced` histórico intacto.
2. Transformar a telemetria em um diagnóstico quantitativo comparativo completo (Top-1, Macro-F1, Worst-10 recall, entropia $H_{\text{norm}}$, cobertura $C_{\text{cov}}$ e correlação exposição–recall).
3. Gerar um relatório estatístico pareado completo e tabela de dispersão por classe no Notebook 03 e nos artefatos exportados.

## Goals

- [ ] Habilitar e parametrizar a execução completa das 7 estratégias de amostragem no Scenario 2 em `run_benchmark()` e no Notebook 03: `uniform`, `uniform_grouped`, `mcts_strict`, `mcts_qnorm_only`, `mcts_dualstream_only`, `mcts_enhanced` e `mcts_coverage_balanced`.
- [ ] Implementar a variante corretiva de política de cobertura `mcts_coverage_balanced` (fração focal reduzida de 25% = 16 amostras focais + 48 uniformes em batch de 64), preservando estritamente a política histórica de 50% em `mcts_enhanced`.
- [ ] Estruturar o diagnóstico quantitativo comparativo consolidando Top-1 Accuracy, Macro-F1, Balanced Accuracy, Worst-10 Class Recall, Entropia Normalizada ($H_{\text{norm}}$), Cobertura Acumulada ($C_{\text{cov}}$), contagem de classes nunca expostas ($N_{\text{unvisited}}$) e correlação Exposição-Recall.
- [ ] Integrar ao Notebook 03 um relatório estatístico pareado completo com tabela de dispersão por classe e isolamento causal definitivo (homogeneidade de batch vs política MCTS).
- [ ] Adicionar suíte de testes de regressão automatizada cobrindo todas as variantes e asserções de diagnóstico sob SLA < 60s.

## Out of Scope

- Modificação dos pipelines e splits do Scenario 1 (congelados e validados na Feature 21).
- Modificação ou substituição destrutiva da estratégia histórica `mcts_enhanced` (deve ser preservada para análise comparativa contínua).
- Execução exaustiva de 2000 passos em GPU em tempo real dentro dos testes unitários rápidos de CI.

---

## Assumptions & Open Questions

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| Grid de variantes no Scenario 2 | 7 estratégias: Uniform, Uniform Grouped, Strict, QNorm-only, DualStream-only, Enhanced e Coverage Balanced | Cobre o isolamento causal da composição do batch, o grid fatorial 2x2 e a política corretiva | y |
| Fração da política de cobertura | `mcts_coverage_balanced`: 25% focal (16 focais + 48 uniformes) | Reduz a saturação por classe no batch enquanto mantém viés ativo de mineração | y |
| Diagnóstico quantitativo e pareado | Métricas agregadas + Entropia H_norm + Cobertura + Worst-10 + Correlação + Estatística Pareada | Atende conjuntamente a Opção 1 e a Opção 3 solicitadas pelo usuário | y |
| Sementes pareadas | 5 seeds: (42, 43, 44, 45, 46) no REAL mode; (42,) em SMOKE | Garante pareamento estrito e reprodutibilidade perfeita | y |

**Open questions:** none - escopo e abordagens de diagnóstico acordados com o usuário.

---

## User Stories

### P1: Orquestração e Comparação das Variantes de Amostragem no Scenario 2 (`S2-01`) ⭐ MVP

**User Story**: Como pesquisador de aprendizado por reforço, quero executar e comparar as variantes de amostragem (`uniform`, `uniform_grouped`, `mcts_strict`, `mcts_qnorm_only`, `mcts_dualstream_only`, `mcts_enhanced`) no Scenario 2 para identificar a causa exata da queda de desempenho multiclasse.

**Why P1**: Requisito fundamental para isolar se a degradação é efeito da concentração de classe no batch (`uniform_grouped`) ou do algoritmo MCTS.

**Acceptance Criteria**:

1. WHEN `run_benchmark()` is called with `scenario="scenario2"` and any subset of the supported strategies, THEN the system SHALL train and evaluate each strategy under identical initializations, seeds, and step budgets.
2. The system SHALL support `uniform_grouped` alongside MCTS variants in Scenario 2 benchmarks, logging identical evaluation metrics across all strategies.
3. The system SHALL export all strategy runs to `tier1_scenario2_curriculum.json` and `tier1_scenario2_curriculum.csv` accompanied by `.meta.json` provenance sidecars.
4. IF an unsupported strategy is provided to Scenario 2, THEN the system SHALL raise `ValueError` listing valid choices.

**Independent Test**: Executar `run_benchmark(scenario="scenario2", strategies=("uniform", "uniform_grouped", "mcts_enhanced"))` em SMOKE mode e verificar que todas as 3 estratégias são avaliadas e persistidas no CSV/JSON.

---

### P1: Variante de Política de Cobertura Balanceada (`S2-02`)

**User Story**: Como cientista de dados, quero dispor de uma variante `mcts_coverage_balanced` que utilize uma fração focal reduzida no batch (25% MCTS / 75% uniforme) para avaliar se a dispersão de rótulos reverte o colapso do Worst-10 recall.

**Why P1**: Fornece uma política corretiva sem modificar os parâmetros do baseline Enhanced histórico.

**Acceptance Criteria**:

1. WHEN `strategy="mcts_coverage_balanced"` is executed in Scenario 2, THEN the system SHALL compose minibatches with 25% samples from the MCTS-mined focal class and 75% uniform samples (ex.: 16 MCTS + 48 uniformes em batch de 64).
2. The system SHALL maintain Min-Max Q-normalization and semantic tree traversal identical to `mcts_enhanced`.
3. The system SHALL preserve `mcts_enhanced` unchanged with its 50/50 ratio (32 MCTS + 32 uniformes).

**Independent Test**: Executar treino de 10 passos com `mcts_coverage_balanced` e verificar que minibatches contêm a proporção 16/48 de classes focais/uniformes.

---

### P1: Diagnóstico Quantitativo e Análise Estatística Pareada (`S2-03`)

**User Story**: Como revisor de artigo científico, quero diagnósticos quantitativos por classe (entropia de exposição $H_{\text{norm}}$, cobertura acumulada $C_{\text{cov}}$, Worst-10 recall e correlação exposição–recall) acompanhados de diferenças estatísticas pareadas por seed para atestar conclusões causais rigorosas.

**Why P1**: Transforma dados brutos de telemetria em evidências científicas reproduzíveis e publicáveis.

**Acceptance Criteria**:

1. WHEN Scenario 2 benchmarks execute, THEN the system SHALL record normalized exposure entropy $H_{\text{norm}}$, cumulative class coverage $C_{\text{cov}}$, unvisited class count $N_{\text{unvisited}}$, and Worst-10 recall for each run.
2. The system SHALL compute pairwise statistical differences (mean difference, std, Cohen's d) between `uniform`, `uniform_grouped`, `mcts_enhanced`, and `mcts_coverage_balanced` across paired seeds.
3. The system SHALL compute the Spearman/Pearson rank correlation between per-class exposure frequencies and per-class test recall.
4. The system SHALL persist these diagnostics in `tier1_scenario2_diagnostics.json` and export a consolidated comparison summary in CSV.

**Independent Test**: Verificar que o resultado do Scenario 2 exporta a matriz de diagnósticos contendo $H_{\text{norm}}$, $C_{\text{cov}}$ e correlação de exposição sem valores nulos.

---

### P1: Atualização do Notebook 03 com Grid Completo e Visualizações (`S2-04`)

**User Story**: Como usuário executando o pipeline no JupyterHub, quero que o Notebook 03 execute o grid completo de estratégias no Scenario 2 e apresente tabelas comparativas detalhadas e análise de dispersão por classe.

**Why P1**: Garante que o notebook executável reflita exatamente as descobertas científicas e o diagnóstico causal acordado.

**Acceptance Criteria**:

1. WHEN executing Notebook 03 under Scenario 2, THEN the system SHALL execute `uniform`, `uniform_grouped`, `mcts_enhanced`, and `mcts_coverage_balanced`.
2. The system SHALL display a consolidated comparison table including Accuracy, Macro-F1, Worst-10 Recall, Exposure Entropy $H_{\text{norm}}$, and Unvisited Classes.
3. The system SHALL print the causal diagnostic verdict identifying whether batch homogeneity or MCTS policy is the primary performance factor.

**Independent Test**: Executar Notebook 03 em SMOKE mode e verificar que a célula do Scenario 2 executa todas as estratégias e exibe a tabela comparativa.

---

## Edge Cases

- IF a class is never visited during MCTS execution, THEN per-class metrics SHALL handle division by zero safely (returning 0.0 recall without raising exceptions).
- IF `num_classes` in Scenario 2 is less than 2, THEN the system SHALL raise `ValueError`.
- IF all classes have identical exposure, THEN entropy $H_{\text{norm}}$ SHALL equal 1.0 within floating-point tolerance.

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| -------------- | ----- | ----- | ------ |
| S2-01 | P1: Orquestração das Variantes de Amostragem | Tasks | Pending |
| S2-02 | P1: Variante de Política de Cobertura Balanceada | Tasks | Pending |
| S2-03 | P1: Diagnóstico Quantitativo e Estatística Pareada | Tasks | Pending |
| S2-04 | P1: Atualização do Notebook 03 e Visualizações | Tasks | Pending |

**Coverage:** 4 total, 4 mapped to tasks, 0 unmapped.

---

## Success Criteria

- [ ] `run_benchmark(scenario="scenario2")` suporta e executa `uniform_grouped`, `mcts_strict`, `mcts_qnorm_only`, `mcts_dualstream_only`, `mcts_enhanced` e `mcts_coverage_balanced`.
- [ ] `mcts_coverage_balanced` opera com fração focal 25% (16/48) preservando `mcts_enhanced` intacto (32/32).
- [ ] Telemetria de diagnóstico quantitativo calcula $H_{\text{norm}}$, $C_{\text{cov}}$, $N_{\text{unvisited}}$ e correlação exposição–recall por seed.
- [ ] Notebook 03 executa a suíte multiclasse e emite a tabela de diagnóstico causal e análise pareada.
- [ ] Suíte de testes automatizados passa com 100% de sucesso.
