# F24: P2 — Expansão do Estudo: Múltiplas Classes-Alvo e Razões Pareadas

**Approved**: 2026-11-11 - user: "sim"

## Problem Statement

As investigações metodológicas anteriores (F21 a F23) estabeleceram a integridade das comparações no Scenario 1 e Scenario 2 para a razão canônica 1:50 e a classe-alvo padrão (`target_class=0`, *beaver*). No entanto, de acordo com o roadmap (`roadmap_mcts_hnm_proxima_run_v2.md`, Item 5 e Fase D1), o estudo ainda possui limitações de escopo que restringem a generalização científica das conclusões:
1. **Ausência de contraste sistemático entre diferentes regimes de desbalanceamento**:
   A eficácia do MCTS-HNM versus Uniforme e OHEM foi avaliada primariamente em $\rho = 1:50$. Para responder se a mineração hierárquica por reforço se torna mais vantajosa em regimes extremos ou se colapsa com a rarefação do sinal positivo, é necessário expandir a avaliação de forma estritamente controlada para as razões $\rho \in \{1:10, 1:50, 1:100\}$ sob sementes pareadas e calcular a interação método $\times$ razão ($\text{Interação}_{\text{método} \times \text{razão}}$).
2. **Dependência de uma única classe-alvo binária (`target_class=0`)**:
   Avaliar apenas a classe `0` (*beaver*, superclasse *aquatic mammals*) introduz o risco de viés de seleção de classe. É indispensável permitir a avaliação de múltiplas classes-alvo (ex.: classes representativas de diferentes superclasses semânticas) com agregações inter-classes pareadas por semente, testando se a dinâmica do MCTS é robusta ou se varia conforme a separabilidade intrínseca de cada classe.
3. **Compatibilidade dos relatórios e artefatos de publicação (Notebook 03 e 05)**:
   Os notebooks experimentais e funções de agregação precisam aceitar nativamente grades multi-razão e multi-classe no Scenario 1, preservando proveniência e integridade sem regressions na geração de tabelas e figuras do manuscrito.

## Goals

- [ ] Estender `run_benchmark()` no Scenario 1 para suportar uma lista de múltiplas classes-alvo `target_classes: tuple[int, ...]`, executando cada classe sob o mesmo protocolo pareado por semente e calculando resumos agregados inter-classes.
- [ ] Padronizar a execução de múltiplas razões de desbalanceamento `imbalance_ratios=("1:10", "1:50", "1:100")` com sementes pareadas e cálculo da interação método $\times$ razão em `src/utils/statistics.py` e persistência em `tier1_axis1_samplers.json`.
- [ ] Atualizar Notebook 03 (Seção 2) para parametrizar e exibir a expansão multi-razão e multi-classe no Scenario 1 de forma transparente.
- [ ] Garantir compatibilidade plena com os carregadores e geradores de artefatos de publicação (Notebook 05 e testes de integridade).
- [ ] Criar suíte de testes de regressão automatizada `tests/test_p2_study_expansion.py` cobrindo multi-classe, multi-razão e interação método $\times$ razão.

## Out of Scope

- Execução indiscriminada de campanhas pesadas em modo REAL com dezenas de classes ou passos exaustivos sem demanda explícita (testes executam em modo rápido / SMOKE com SLA < 60s).
- Alteração da lógica canônica do `build_scenario1_split` congelada na Feature 21.
- Inclusão do regime extremo 1:500 sem aprovação do Gate B.

---

## Assumptions & Open Questions

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| Razões de Desbalanceamento | Grade de 3 razões canônicas: `1:10`, `1:50`, `1:100` | Cobre desde desbalanceamento moderado até severo com 5 sementes pareadas | y |
| Classes-Alvo no Estudo Estendido | `target_classes=(0, 1, 2)` como padrão configurável no Scenario 1 | Amostra classes de superclasses distintas (aquatic mammals, fish, flowers) para testar generalização semântica | y |
| Interação Método x Razão | $\text{Interação}(r_1, r_2) = \Delta_{\text{método}}(r_2) - \Delta_{\text{método}}(r_1)$ com IC 95% e teste t | Mede formalmente se o ganho relativo de um método varia em função da escassez de positivos | y |
| Protocolo de Avaliação | 4-quadrantes (last vs best_val x 0.5 vs calibrado) mantido idêntico em todas as razões | Preserva rigor metodológico estabelecido em F21 | y |

**Open questions:** none - escopo e metodologia de expansão multi-razão e multi-classe acordados com o usuário.

---

## User Stories

### P2: Avaliação Multi-Razão e Interação Método x Razão (`EXP-01`) ⭐ MVP

**User Story**: Como pesquisador, quero avaliar os amostradores em múltiplos níveis de desbalanceamento ($\rho \in \{1:10, 1:50, 1:100\}$) sob sementes pareadas e quantificar o efeito de interação método $\times$ razão para determinar em qual regime de escassez o MCTS oferece maior benefício ou degradação.

**Why P2**: Responde à questão fundamental sobre a sensibilidade do método ao grau de rarefação da classe positiva.

**Acceptance Criteria**:

1. `run_benchmark()` with `scenario="scenario1"` SHALL accept a tuple of `imbalance_ratios` (e.g. `("1:10", "1:50", "1:100")`) and evaluate each ratio across all specified strategies and paired seeds.
2. For each ratio, positive pool subsampling via `build_scenario1_split()` SHALL map ratio string to corresponding positive fraction (`1:10` -> `pos_fraction=0.10`, `1:50` -> `0.02`, `1:100` -> `0.01`).
3. `compute_method_by_ratio_interaction()` in `src/utils/statistics.py` SHALL compute paired differences and interaction effects across ratios ($\Delta_{\text{method}}(r_2) - \Delta_{\text{method}}(r_1)$) with 95% confidence intervals and hypothesis tests.
4. Output JSON (`tier1_axis1_samplers.json`) and CSV SHALL record the `imbalance_ratio` for every individual run and include `ratio_interactions` in aggregated results when multiple ratios are evaluated.

**Independent Test**: Executar benchmark multi-razão em modo teste e verificar se todas as razões foram avaliadas com pareamento de seeds e se a matriz de interação foi gerada.

---

### P2: Generalização para Múltiplas Classes-Alvo (`EXP-02`)

**User Story**: Como pesquisador, quero avaliar o Scenario 1 em múltiplas classes-alvo (ex.: classes 0, 1, 2) mantendo o mesmo protocolo e sementes pareadas para verificar se as conclusões sobre MCTS generalizam além de uma classe particular.

**Why P2**: Elimina viés de seleção de classe e fortalece a validade externa do estudo experimental.

**Acceptance Criteria**:

1. `run_benchmark()` SHALL accept `target_classes: tuple[int, ...] | None` (defaulting to `(target_class_scenario1,)` if omitted) for Scenario 1.
2. For each target class in `target_classes`, runs SHALL execute with identical seed schedules and record `target_class` in each `BenchmarkRunRecord` and CSV output.
3. The benchmark runner SHALL aggregate metrics across both classes and strategies, reporting mean and dispersion across target classes.
4. Output JSON SHALL persist `target_classes` in provenance and metadata sidecars.

**Independent Test**: Executar benchmark com `target_classes=(0, 1)` e verificar que os registros foram criados para ambas as classes com a respectiva identificação e agregação.

---

### P2: Alinhamento de Notebooks e Artefatos a Jusante (`EXP-03`)

**User Story**: Como leitor ou revisor, quero que o Notebook 03 e o pipeline de geração de artefatos (Notebook 05) processem os resultados multi-razão e multi-classe sem quebras de esquema ou dados ausentes.

**Why P2**: Garante reprodutibilidade e integridade da geração de figuras e tabelas finais do manuscrito.

**Acceptance Criteria**:

1. Notebook 03 Section 2 SHALL allow configuring `IMBALANCE_RATIOS_AXIS1` and display comparative summary grouped by strategy and ratio when multiple ratios are evaluated.
2. Exporters and data loaders in `src/paper/loaders.py` and `tests/test_paper_artifacts_real.py` SHALL load multi-ratio and multi-class records without schema validation errors.
3. Notebook 03 SHALL execute end-to-end without errors in SMOKE mode with exit code 0.

**Independent Test**: Executar os testes de carga de artefatos do artigo e o smoke test do Notebook 03.

---

## Edge Cases

- IF a single imbalance ratio is provided, THEN method $\times$ ratio interaction calculation SHALL be skipped gracefully without raising errors.
- IF a single target class is provided, THEN aggregation across classes SHALL be identity without extra overhead.
- IF fewer than 2 common seeds exist between ratios, THEN interaction significance testing SHALL safely report `t_pvalue=1.0` without NaN propagation.

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| -------------- | ----- | ----- | ------ |
| EXP-01 | P2: Avaliação Multi-Razão e Interação Método x Razão | Tasks | Pending |
| EXP-02 | P2: Generalização para Múltiplas Classes-Alvo | Tasks | Pending |
| EXP-03 | P2: Alinhamento de Notebooks e Artefatos a Jusante | Tasks | Pending |

**Coverage:** 3 total, 3 mapped to tasks, 0 unmapped.

---

## Success Criteria

- [ ] `run_benchmark()` no Scenario 1 suporta execução de múltiplas razões `("1:10", "1:50", "1:100")` e múltiplas classes `target_classes`.
- [ ] Função `compute_method_by_ratio_interaction()` implementada e testada calculando a interação método $\times$ razão com IC 95% e testes pareados.
- [ ] Registros individuais e resumos agregados por classe e razão gravados no CSV e JSON de proveniência.
- [ ] Notebook 03 Seção 2 parametrizado e executável em modo SMOKE com exit code 0.
- [ ] Suíte de testes automatizados `tests/test_p2_study_expansion.py` passa com 100% de sucesso.
