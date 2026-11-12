# F25: P0 — Expansão Analítica: Fronteira de Pareto e Dinâmica de Entropia/Cobertura

**Approved**: 2026-11-11 - user: "aprovado"

## Problem Statement

O roadmap de expansão analítica (`roadmap_analises_e_figuras_expansao.md`) define as duas figuras analíticas prioritárias de Fase 1 (P0) para enriquecer a discussão científica e responder às principais objeções metodológicas dos revisores de conferências de topo (ICLR / NeurIPS / ICML):
1. **FIG-01 (Fronteira de Pareto: Custo Computacional vs. Desempenho)**:
   A principal crítica ao MCTS em aprendizado com classes desbalanceadas reside na suposição de um overhead computacional proibitivo de busca em árvore em relação a amostragem aleatória e OHEM. Para desmistificar essa objeção, é fundamental plotar a fronteira de Pareto de custo computacional (tempo de parede em segundos ou amostras com gradiente) versus métrica de eficácia (Minority F1 para Cenário 1 e Macro-F1 / Worst-10 Recall para Cenário 2) com barras de dispersão ($\pm 1\ \text{std}$) entre seeds pareadas e a linha de Pareto delimitando as estratégias dominantes (`uniform`, `ohem`, `mcts_strict`, `mcts_coverage_balanced`, `mcts_enhanced`).
2. **FIG-02 (Evolução Temporal da Entropia de Visitação e Cobertura de Classes no Cenário 2)**:
   A causa raiz do colapso de exploração no Cenário 2 (100 classes) é a inanição de classes raras pela ganância de nós altamente recompensados. É indispensável gerar um gráfico temporal de painel duplo:
   - Subplot (a): Evolução da Entropia de Shannon normalizada $H(t)/H_{\max}$ ao longo dos 1.000 passos de treinamento, contrastando o colapso prematuro do `mcts_strict` com a preservação de exploração sustentada do `mcts_coverage_balanced` e `mcts_enhanced`.
   - Subplot (b): Cobertura cumulativa de classes (% de classes com $k \ge 1$ e $k \ge 5$ amostragens) ao longo dos passos, comprovando que a política balanceada elimina a inanição.
3. **Isolamento e Formato Duplo Obrigatório**:
   Ambas as figuras devem ser exportadas estritamente em formato duplo (vetor `.pdf` para publicação e `.png` 300 DPI para inspeção interativa), suportando tanto modo SMOKE (com tarja explícita de dados preliminares em `results/smoke/paper/figures/`) quanto modo REAL (sem sintetização a partir de `results/real/`, salvando em `paper/figures/`), além de incorporação dos blocos `\begin{figure}` correspondentes no manuscrito `paper/main.tex` e integração em `src/utils/paper_artifacts.py` e Notebook 05.

## Goals

- [ ] Implementar a função `generate_pareto_cost_performance_plot()` em `src/utils/paper_artifacts.py` gerando FIG-01 em formatos `.pdf` e `.png` (300 DPI) para Cenário 1 e Cenário 2 a partir dos registros de `computational_costs`.
- [ ] Implementar a função `generate_coverage_entropy_dynamics_plot()` em `src/utils/paper_artifacts.py` gerando FIG-02 em painel duplo (entropia normalizada temporal + cobertura cumulativa de classes) em formatos `.pdf` e `.png` (300 DPI).
- [ ] Integrar a geração de FIG-01 e FIG-02 no orquestrador `generate_all_paper_artifacts()` e no manifesto de artefatos.
- [ ] Atualizar o manuscrito `paper/main.tex` incluindo os ambientes `\begin{figure}` para FIG-01 e FIG-02 com legendas e referências completas.
- [ ] Atualizar o Notebook 05 (`notebooks/05_paper_artifacts.ipynb`) para exibir e validar a geração de FIG-01 e FIG-02 em modo SMOKE.
- [ ] Criar a suíte de testes automatizados `tests/test_p0_pareto_entropy_figures.py` validando geração, integridade dos formatos, metadados e isolamento estrito de modo.

## Out of Scope

- Implementação das figuras das Fases 2 (FIG-03, FIG-04, FIG-05) e Fase 3 (FIG-06, FIG-07) do roadmap, que serão abordadas nas Features subsequentes.
- Execução de campanhas pesadas em modo REAL (os testes utilizam mocks e fixtures em modo SMOKE com execução rápida < 10s).
- Alteração dos laços centrais de treinamento ou perda de retrocompatibilidade com artefatos legados.

---

## Assumptions & Open Questions

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| Estratégias na Fronteira de Pareto | `uniform`, `ohem`, `mcts_strict`, `mcts_coverage_balanced`, `mcts_enhanced` | Abrange todos os baselines canônicos e variantes de MCTS avaliados no estudo | y |
| Métrica no Eixo Y de Pareto | Suporte configurável: Minority F1 (Cenário 1) ou Macro-F1 (Cenário 2), com seleção automática baseada no cenário dos dados | Garante transparência para ambos os regimes avaliados | y |
| Fonte de Dados da FIG-02 | Extração de `step_history` (`normalized_entropy`) e `exposure_diagnostics` / `visitation_checkpoints` persistidos na run | Aproveita a instrumentação rica já implementada sem custo de re-treinamento | y |
| Tarja de Modo SMOKE | Marca d'água semi-transparente no topo ou diagonal superior em modo SMOKE | Garante conformidade com o contrato de integridade experimental AD-010 | y |

**Open questions:** none - escopo acordado e priorizado como Fase 1 (P0) do roadmap de expansão.

---

## User Stories

### P0: Fronteira de Pareto Custo vs. Desempenho (`FIG-01`) ⭐ MVP

**User Story**: Como autor do artigo, quero gerar a figura da fronteira de Pareto (`fig_pareto_cost_vs_accuracy`) comparando tempo/amostras versus eficácia para demonstrar visualmente que o ganho do MCTS supera o custo computacional relativo a amostradores aleatórios e OHEM.

**Why P0**: Responde diretamente à principal objeção de revisores de conferências sobre a viabilidade computacional de MCTS no pipeline de treinamento.

**Acceptance Criteria**:

1. `generate_pareto_cost_performance_plot()` SHALL accept run payloads, campaign directory, or manifest containing `computational_costs` (`elapsed_seconds`, `gradient_samples`) and performance metrics.
2. The generator SHALL calculate mean and standard deviation across seeds for each strategy and plot points with error bars ($\pm 1\ \text{std}$).
3. The generator SHALL identify non-dominated strategies and draw a dashed or solid Pareto frontier line connecting them.
4. When executed, the generator SHALL export both a vector `.pdf` and a 300 DPI `.png` with `fig_pareto_cost_vs_accuracy` prefix in the target output directory.
5. When running in SMOKE mode, the generator SHALL overlay the standard warning watermark banner.

**Independent Test**: Invocar a função com dados sintéticos controlados e verificar a criação dos arquivos `.pdf` e `.png`, checando se as coordenadas e a fronteira de Pareto foram calculadas corretamente.

---

### P0: Dinâmica Temporal de Entropia e Cobertura de Classes (`FIG-02`) ⭐ MVP

**User Story**: Como pesquisador, quero gerar o gráfico de painel duplo da dinâmica temporal de visitação (`fig_coverage_entropy_dynamics`) demonstrando o decaimento precoce de entropia no MCTS estrito e a sustentação de exploração e cobertura na política balanceada.

**Why P0**: Fornece a evidência empírica fundamental que valida o mecanismo de aprendizado por reforço hierárquico e justifica a variante Coverage-Balanced.

**Acceptance Criteria**:

1. `generate_coverage_entropy_dynamics_plot()` SHALL accept Scenario 2 runs containing `history` / `step_history` with `normalized_entropy` and exposure diagnostics.
2. Subplot (a) SHALL plot curves of normalized entropy $H(t)/H_{\max}$ over training steps for evaluated strategies (`uniform`, `mcts_strict`, `mcts_enhanced`, `mcts_coverage_balanced`).
3. Subplot (b) SHALL plot cumulative class coverage percentage over steps or compare final coverage thresholds ($k \ge 1$ and $k \ge 5$).
4. When executed, the generator SHALL export both a vector `.pdf` and a 300 DPI `.png` with `fig_coverage_entropy_dynamics` prefix in the designated output directory.
5. In SMOKE mode, the generator SHALL display the explicit preliminary data warning banner.

**Independent Test**: Invocar a função com histórico de passos conhecido e verificar que os dois subplots foram renderizados e salvos nos caminhos esperados em formatos `.pdf` e `.png`.

---

### P0: Integração no Manuscrito LaTeX, Notebook 05 e Pipeline Global (`FIG-03`)

**User Story**: Como leitor ou revisor, quero que as novas figuras estejam integradas em `paper/main.tex`, invocadas no Notebook 05 e geradas automaticamente por `generate_all_paper_artifacts()` sem quebras de integridade.

**Why P0**: Garante que o manuscrito acadêmico final reflita diretamente os novos artefatos e mantenha compilação reprodutível de ponta a ponta.

**Acceptance Criteria**:

1. `generate_all_paper_artifacts()` SHALL invoke both `generate_pareto_cost_performance_plot()` and `generate_coverage_entropy_dynamics_plot()` and include their paths in the returned artifact manifest.
2. `paper/main.tex` SHALL contain `\begin{figure}` environments for both `fig_pareto_cost_vs_accuracy` and `fig_coverage_entropy_dynamics` with informative captions and labels.
3. Notebook 05 (`notebooks/05_paper_artifacts.ipynb`) SHALL contain execution cells displaying the new figures.
4. `tests/test_p0_pareto_entropy_figures.py` SHALL pass 100% of its test cases.

**Independent Test**: Executar a suíte de testes dedicada e `generate_all_paper_artifacts()` em modo SMOKE, inspecionando o manifesto gerado e os arquivos exportados.

---

## Edge Cases

- IF a strategy has only 1 seed, THEN error bars SHALL collapse to zero without causing NaN errors.
- IF `elapsed_seconds` is zero or missing, THEN the generator SHALL fall back to `total_updates` or `gradient_samples` as the cost metric.
- IF a run does not contain `normalized_entropy` at every step, THEN linear interpolation SHALL fill missing values across the step axis.
- IF output directory does not exist, THEN the generator SHALL create it automatically.

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| -------------- | ----- | ----- | ------ |
| FIG-01 | P0: Fronteira de Pareto Custo vs. Desempenho | Tasks | Pending |
| FIG-02 | P0: Dinâmica Temporal de Entropia e Cobertura de Classes | Tasks | Pending |
| FIG-03 | P0: Integração no Manuscrito LaTeX, Notebook 05 e Pipeline Global | Tasks | Pending |

**Coverage:** 3 total, 3 mapped to tasks, 0 unmapped.

---

## Success Criteria

- [ ] FIG-01 (`fig_pareto_cost_vs_accuracy`) gerada em `.pdf` e `.png` (300 DPI) com pontos médios, barras de erro e linha de fronteira de Pareto.
- [ ] FIG-02 (`fig_coverage_entropy_dynamics`) gerada em `.pdf` e `.png` (300 DPI) com painel duplo de evolução de entropia e cobertura cumulativa de classes.
- [ ] Manuscrito `paper/main.tex` atualizado com seções e blocos de figura LaTeX para FIG-01 e FIG-02.
- [ ] Orquestrador `generate_all_paper_artifacts()` em `src/utils/paper_artifacts.py` e Notebook 05 integrados gerando as novas figuras.
- [ ] Suíte de testes `tests/test_p0_pareto_entropy_figures.py` aprovada com 100% de sucesso.
