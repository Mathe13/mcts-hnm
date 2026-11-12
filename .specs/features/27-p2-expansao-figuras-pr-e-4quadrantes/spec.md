# F27: P2 — Expansão Analítica: Curvas Precision-Recall e Análise em 4 Quadrantes

**Approved**: 2026-11-12 - user: "prosseguir"

## Problem Statement

A Fase 3 (P2) do roadmap de expansão analítica (`roadmap_analises_e_figuras_expansao.md`, Seção 4.6 e 4.7) consolida duas análises visuais fundamentais de transparência metodológica e caracterização contínua de desempenho:
1. **FIG-06 (Curvas Precision-Recall Agregadas com Bandas de Variância — `fig_precision_recall_curves`)**:
   As tabelas do manuscrito reportam o escalar agregado de AUPRC, mas não exibem a dinâmica da fronteira Precision-Recall contínua $P(R)$ em todos os níveis de sensibilidade. É necessário traçar as curvas $P(R)$ interpoladas em grade uniforme de 100 pontos de recall ($R \in [0.0, 1.0]$) com médias sólidas e bandas de variância ($\pm 1\ \text{std}$) entre sementes pareadas para cada estratégia (`Uniform`, `OHEM`, `MCTS Strict`, `Enhanced MCTS-HNM`), além de assinalar o ponto de operação sob o limiar ótimo calibrado na validação ($\tau_{\text{val}}$).
2. **FIG-07 (Superfície de Trade-off dos 4 Quadrantes de Avaliação — `fig_evaluation_4quadrants_comparison`)**:
   Para demonstrar de forma visualmente transparente que a superioridade do MCTS independe de escolhas arbitrárias de protocolo de avaliação (eliminando qualquer suspeita de cherry-picking), é indispensável gerar um gráfico comparativo em 4 quadrantes para cada método:
   - Q1: Checkpoint Final com limiar fixo $\tau = 0.5$
   - Q2: Checkpoint Final com limiar calibrado $\tau = \tau_{\text{val}}$
   - Q3: Melhor Checkpoint de Validação com limiar fixo $\tau = 0.5$
   - Q4: Melhor Checkpoint de Validação com limiar calibrado $\tau = \tau_{\text{val}}$
3. **Integração no Manuscrito LaTeX e Notebook 05**:
   Ambas as figuras devem suportar formatos duplos (.pdf vetorial e .png 300 DPI), watermark em modo SMOKE, incorporação de ambientes `\begin{figure}` em `paper/main.tex` e integração em `generate_all_paper_artifacts()` e `notebooks/05_paper_artifacts.ipynb`.

## Goals

- [ ] Implementar `generate_precision_recall_curves_plot()` em `src/utils/paper_artifacts.py` gerando FIG-06 com interpolação contínua em 100 pontos de recall, bandas $\pm 1\ \text{std}$ e marcadores do limiar calibrado $\tau_{\text{val}}$.
- [ ] Implementar `generate_quadrant_evaluation_plot()` em `src/utils/paper_artifacts.py` gerando FIG-07 com barras agrupadas nos 4 quadrantes de avaliação para cada método.
- [ ] Integrar os geradores de FIG-06 e FIG-07 no orquestrador `generate_all_paper_artifacts()` e registrar no manifesto de artefatos.
- [ ] Atualizar `notebooks/05_paper_artifacts.ipynb` com células interativas para exibição de FIG-06 e FIG-07.
- [ ] Adicionar blocos `\begin{figure}` com legendas e referências acadêmicas em `paper/main.tex`.
- [ ] Criar a suíte de testes automatizados `tests/test_p2_pr_and_quadrant_figures.py` validando 100% dos critérios.

## Out of Scope

- Modificação dos critérios canônicos de calibração de limiar congelados na Feature 21.
- Execução indiscriminada de campanhas pesadas em modo REAL.

---

## Assumptions & Open Questions

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| Grade de Interpolação de Recall | Grade uniforme de 100 pontos linearmente espaçados em $[0.0, 1.0]$ | Permite agregação e bandas de variância consistentes entre sementes | y |
| Métrica nos 4 Quadrantes | F1-Score da classe minoritária (Scenario 1) ou Macro-F1 (Scenario 2) | Reflete diretamente o padrão estabelecido na Feature 21 (AD-024) | y |
| Marcador de Limiar Calibrado | Ponto com marcador destacado sobre a curva PR indicando o ponto de operação de $\tau_{\text{val}}$ | Permite visualizar o trade-off precisão-sensibilidade escolhido pela validação | y |

**Open questions:** none - escopo derivado diretamente da Seção 4 do roadmap.

---

## User Stories

### P2: Curvas Precision-Recall com Bandas de Variância (`FIG-06`) ⭐ MVP

**User Story**: Como autor do artigo, quero gerar as curvas Precision-Recall contínuas com bandas de dispersão (`fig_precision_recall_curves`) para demonstrar em qual regime de recall o MCTS supera os baselines e destacar a operação sob o limiar calibrado.

**Why P2**: Complementa as tabelas escalares de AUPRC com a curva $P(R)$ completa, permitindo verificar a retenção de precisão em regimes de alto recall.

**Acceptance Criteria**:

1. `generate_precision_recall_curves_plot()` SHALL compute continuous precision curves interpolated across a 100-point uniform recall grid $[0.0, 1.0]$ for each seed.
2. The generator SHALL calculate mean and standard deviation across seeds, plotting solid curve lines and $\pm 1\ \text{std}$ shaded variance bands.
3. The generator SHALL plot a distinct operating point marker indicating the precision and recall achieved under validation-calibrated threshold $\tau_{\text{val}}$.
4. The generator SHALL export both a vector `.pdf` and a 300 DPI `.png` with `fig_precision_recall_curves` prefix.
5. In SMOKE mode, the generator SHALL overlay the standard preliminary data watermark banner.

**Independent Test**: Invocar o gerador com previsões sintéticas e verificar que as curvas e bandas foram geradas e salvas em formatos `.pdf` e `.png`.

---

### P2: Superfície de Trade-off dos 4 Quadrantes de Avaliação (`FIG-07`) ⭐ MVP

**User Story**: Como pesquisador, quero gerar o gráfico comparativo dos 4 quadrantes de avaliação (`fig_evaluation_4quadrants_comparison`) para demonstrar que o ganho do MCTS se mantém independente de limiar fixo ou calibrado e de checkpoint final ou selecionado.

**Why P2**: Garante transparência metodológica total, provando que os resultados não dependem de calibração favorável artificial.

**Acceptance Criteria**:

1. `generate_quadrant_evaluation_plot()` SHALL evaluate metrics across the four quadrants: (Q1) Last Checkpoint $\tau=0.5$, (Q2) Last Checkpoint $\tau=\tau_{\text{val}}$, (Q3) Best Val Checkpoint $\tau=0.5$, and (Q4) Best Val Checkpoint $\tau=\tau_{\text{val}}$.
2. The generator SHALL plot grouped bars per method with error bars ($\pm 1\ \text{std}$) across seeds.
3. The generator SHALL export both a vector `.pdf` and a 300 DPI `.png` with `fig_evaluation_4quadrants_comparison` prefix.
4. In SMOKE mode, the generator SHALL display the explicit preliminary data watermark banner.

**Independent Test**: Invocar o gerador com resultados multi-quadrante e verificar a integridade visual e estrutural das barras e arquivos exportados.

---

### P2: Integração no Manuscrito LaTeX, Notebook 05 e Pipeline Global (`INT-02`)

**User Story**: Como leitor ou revisor, quero que as figuras de Fase 3 estejam integradas no pipeline global `generate_all_paper_artifacts()`, visualizadas no Notebook 05 e referenciadas em `paper/main.tex`.

**Why P2**: Assegura a integridade de ponta a ponta de todos os 7 artefatos do roadmap analítico.

**Acceptance Criteria**:

1. `generate_all_paper_artifacts()` SHALL execute FIG-06 e FIG-07 e incluir seus caminhos no manifesto de artefatos.
2. `paper/main.tex` SHALL contain `\begin{figure}` environments for FIG-06 e FIG-07 com legendas e labels acadêmicos.
3. Notebook 05 (`notebooks/05_paper_artifacts.ipynb`) SHALL contain execution cells displaying FIG-06 e FIG-07.
4. `tests/test_p2_pr_and_quadrant_figures.py` SHALL pass 100% of its test cases.

**Independent Test**: Executar a suíte de testes automatizados e verificar o manifesto global gerado.

---

## Edge Cases

- IF precision values contain zeros or NaNs in extreme recall regimes, THEN the generator SHALL enforce monotonic precision envelopes avoiding plotting artifacts.
- IF a strategy has only 1 seed, THEN error bars and variance bands SHALL collapse gracefully to zero.
- IF output directory does not exist, THEN the generator SHALL create parent directories automatically.

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| -------------- | ----- | ----- | ------ |
| FIG-06 | P2: Curvas Precision-Recall com Bandas de Variância | Tasks | Pending |
| FIG-07 | P2: Superfície de Trade-off dos 4 Quadrantes de Avaliação | Tasks | Pending |
| INT-02 | P2: Integração no Manuscrito LaTeX, Notebook 05 e Pipeline Global | Tasks | Pending |

**Coverage:** 3 total, 3 mapped to tasks, 0 unmapped.

---

## Success Criteria

- [ ] FIG-06 (`fig_precision_recall_curves`) gerada em `.pdf` e `.png` (300 DPI) com curvas $P(R)$, bandas $\pm 1\ \text{std}$ e ponto operacional $\tau_{\text{val}}$.
- [ ] FIG-07 (`fig_evaluation_4quadrants_comparison`) gerada em `.pdf` e `.png` (300 DPI) com gráfico de barras agrupadas em 4 quadrantes por método.
- [ ] Manuscrito `paper/main.tex` atualizado com seções e blocos `\begin{figure}` para FIG-06 e FIG-07.
- [ ] Orquestrador `generate_all_paper_artifacts()` e `notebooks/05_paper_artifacts.ipynb` integrados gerando as novas figuras.
- [ ] Suíte `tests/test_p2_pr_and_quadrant_figures.py` aprovada com 100% de sucesso.
