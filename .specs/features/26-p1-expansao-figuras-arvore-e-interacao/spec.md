# F26: P1 — Expansão Analítica: Geometria de Árvores e Interação Método x Desbalanceamento

**Approved**: 2026-11-12 - user: "sim"

## Problem Statement

A Fase 2 (P1) do roadmap de expansão analítica (`roadmap_analises_e_figuras_expansao.md`, Seção 4) aborda três questões científicas centrais sobre o comportamento estrutural do MCTS e sua robustez em cenários extremos:
1. **FIG-03 (Alocação Amostral vs. Ganho Marginal de Recall por Classe — `fig_class_exposure_vs_delta_recall`)**:
   É fundamental responder se o MCTS age efetivamente como um currículo ativo e adaptativo, alocando esforço de amostragem proporcionalmente às classes onde o benefício marginal de aprendizado ($\Delta \text{Recall}(c) = \text{Recall}_{\text{MCTS}}(c) - \text{Recall}_{\text{Uniform}}(c)$) é maior, correlacionando a frequência de visitas com a recuperação de classes difíceis via coeficiente de Spearman ($\rho_s$).
2. **FIG-04 (Dendrograma Comparativo: Ward vs. Semântica — `fig_ward_vs_semantic_dendrogram`)**:
   Visualizar onde o agrupamento empírico da rede neural (árvore de Ward calculada sobre a matriz de confusão de validação) diverge da ontologia taxonômica humana pré-definida (20 superclasses do CIFAR-100), evidenciando clusters perceptuais cruzados (ex.: mamíferos aquáticos agrupados com peixes) que justificam a superioridade empírica da árvore de Ward.
3. **FIG-05 (Curvas de Interação Método $\times$ Severidade do Desbalanceamento — `fig_interaction_ratio_methods`)**:
   Demonstrar a hipótese de que a vantagem do MCTS-HNM se amplia conforme o desbalanceamento se torna mais severo ($\rho \in \{1:10, 1:50, 1:100\}$), traçando curvas de Minority F1 e AUPRC com intervalos de confiança de 95% e anotações estatísticas do teste de interação ($\text{Interação}_{\text{método} \times \text{razão}}$).
4. **Integração no Manuscrito LaTeX e Notebooks**:
   Todas as figuras devem suportar formatos duplos (.pdf vetorial e .png 300 DPI), isolamento de modo (marca d'água preliminar em SMOKE), compilação transparente em `paper/main.tex` e visualização interativa em `notebooks/05_paper_artifacts.ipynb`.

## Goals

- [ ] Implementar `generate_class_exposure_vs_delta_recall_plot()` em `src/utils/paper_artifacts.py` gerando FIG-03 com scatter de classes, linha de tendência e métricas de correlação de Spearman.
- [ ] Implementar `generate_ward_vs_semantic_dendrogram_plot()` em `src/utils/paper_artifacts.py` gerando FIG-04 com dendrograma colorido por superclasse taxonômica e anotações de divergência semântica.
- [ ] Implementar `generate_interaction_ratio_plot()` em `src/utils/paper_artifacts.py` gerando FIG-05 em painel duplo (F1 e AUPRC vs. $\rho$) com bandas IC 95% e anotação do efeito de interação.
- [ ] Conectar os três novos geradores em `generate_all_paper_artifacts()` e registrar no manifesto de artefatos.
- [ ] Atualizar `notebooks/05_paper_artifacts.ipynb` com células interativas para FIG-03, FIG-04 e FIG-05.
- [ ] Incorporar ambientes `\begin{figure}` correspondentes em `paper/main.tex` com discussões e labels acadêmicos.
- [ ] Criar a suíte de testes de regressão `tests/test_p1_expansion_figures.py` cobrindo 100% dos critérios.

## Out of Scope

- Implementação das figuras da Fase 3 (FIG-06 e FIG-07), que serão abordadas na Feature 27.
- Execução de campanhas completas em modo REAL com 10.000 passos (testes utilizam fixtures rápidas em modo SMOKE).
- Alteração na geração da matriz de confusão canônica congelada no módulo `src/trees/confusion_tree.py`.

---

## Assumptions & Open Questions

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| Taxonomia do CIFAR-100 para Dendrograma | 20 superclasses canônicas de 5 classes cada de `src/data/cifar100.py` | Padrão oficial do dataset CIFAR-100 | y |
| Cores do Dendrograma | Color-map categórico mapeando as 20 superclasses aos ramos e folhas | Permite visualização imediata de ramificações que mesclam superclasses distintas | y |
| Correlação de Recall vs. Visitas | Spearman $\rho_s$ e Pearson $r$ via `compute_exposure_recall_correlation()` | Fornece métrica não-paramétrica robusta a outliers | y |
| Eixo X de Interação | Razões de desbalanceamento $\rho \in [1:10, 1:50, 1:100]$ | Cobre o espectro do estudo com pareamento de seeds | y |

**Open questions:** none - escopo e detalhamento derivados diretamente da Seção 4 do roadmap.

---

## User Stories

### P1: Alocação Amostral vs. Ganho Marginal de Recall (`FIG-03`) ⭐ MVP

**User Story**: Como autor do artigo, quero gerar a figura de correlação entre frequência de visitas do MCTS e ganho marginal de recall por classe (`fig_class_exposure_vs_delta_recall`) para demonstrar empiricamente que o MCTS aloca esforço nas classes mais beneficiadas.

**Why P1**: Valida a hipótese fundamental de currículo ativo do MCTS frente a amostragem aleatória uniforme.

**Acceptance Criteria**:

1. `generate_class_exposure_vs_delta_recall_plot()` SHALL calculate class exposure frequencies and recall improvements ($\Delta \text{Recall} = \text{Recall}_{\text{MCTS}} - \text{Recall}_{\text{Uniform}}$) for all classes.
2. The generator SHALL calculate and annotate the Spearman rank correlation coefficient ($\rho_s$) and two-sided $p$-value on the plot.
3. Points SHALL be colored or categorized by class difficulty or superclass ontology.
4. The generator SHALL export both a vector `.pdf` and a 300 DPI `.png` with `fig_class_exposure_vs_delta_recall` prefix.
5. In SMOKE mode, the generator SHALL overlay the standard warning watermark banner.

**Independent Test**: Invocar o gerador com dados de exposição e recall sintéticos e verificar cálculo correto da correlação e exportação nos dois formatos.

---

### P1: Dendrograma Comparativo Ward vs. Semântica (`FIG-04`) ⭐ MVP

**User Story**: Como pesquisador, quero gerar o dendrograma comparativo entre a árvore empírica de Ward e a taxonomia semântica humana (`fig_ward_vs_semantic_dendrogram`) para mostrar as divergências perceptuais descobertas pela rede neural.

**Why P1**: Justifica visualmente por que a árvore orientada a confusão (Ward) supera a taxonomia humana no aprendizado com desbalanceamento.

**Acceptance Criteria**:

1. `generate_ward_vs_semantic_dendrogram_plot()` SHALL accept a confusion matrix, Ward linkage matrix, or results dictionary and reconstruct the hierarchical dendrogram.
2. The dendrogram leaves and links SHALL be colored according to the 20 ground-truth superclasses of CIFAR-100.
3. Visual annotations or bounding highlights SHALL point to clusters where perceptual confusion transcends human taxonomic boundaries.
4. The generator SHALL export both a vector `.pdf` and a 300 DPI `.png` with `fig_ward_vs_semantic_dendrogram` prefix.
5. In SMOKE mode, the generator SHALL display the explicit preliminary data warning banner.

**Independent Test**: Invocar o gerador com uma matriz de confusão 100x100 e validar que o dendrograma é gerado e salvo sem erros nos caminhos esperados.

---

### P1: Curvas de Interação Método x Severidade (`FIG-05`) ⭐ MVP

**User Story**: Como leitor ou revisor, quero visualizar o gráfico de interação método $\times$ severidade de desbalanceamento (`fig_interaction_ratio_methods`) para comprovar se a vantagem relativa do MCTS cresce em regimes mais extremos de desbalanceamento.

**Why P1**: Responde se o método é resiliente ao aumento da escassez de dados positivos ou se degrada.

**Acceptance Criteria**:

1. `generate_interaction_ratio_plot()` SHALL plot curves for evaluated strategies across imbalance ratios $\rho \in [1:10, 1:50, 1:100]$.
2. The plot SHALL present dual panels: (a) Minority F1-Score and (b) AUPRC with 95% confidence interval shaded bands across paired seeds.
3. The generator SHALL compute and annotate the interaction effect ($\Delta(1:100) - \Delta(1:10)$) or test statistic on the figure.
4. The generator SHALL export both a vector `.pdf` and a 300 DPI `.png` with `fig_interaction_ratio_methods` prefix.
5. In SMOKE mode, the generator SHALL apply the standard preliminary data watermark banner.

**Independent Test**: Invocar o gerador com resultados multi-razão e verificar que as curvas e bandas de confiança foram geradas e exportadas corretamente.

---

### P1: Integração no Manuscrito LaTeX, Notebook 05 e Pipeline Global (`INT-01`)

**User Story**: Como leitor ou pesquisador, quero que todas as figuras de Fase 2 estejam integradas em `generate_all_paper_artifacts()`, visualizáveis no Notebook 05 e incluídas no manuscrito `paper/main.tex`.

**Why P1**: Assegura a integridade do artigo científico e a reprodutibilidade dos artefatos de publicação.

**Acceptance Criteria**:

1. `generate_all_paper_artifacts()` SHALL invoke FIG-03, FIG-04 e FIG-05 e registrar seus caminhos no manifesto de artefatos.
2. `paper/main.tex` SHALL contain `\begin{figure}` environments for FIG-03, FIG-04 e FIG-05 with descriptive captions and labels.
3. Notebook 05 (`notebooks/05_paper_artifacts.ipynb`) SHALL contain execution cells displaying the new figures.
4. `tests/test_p1_expansion_figures.py` SHALL pass 100% of its test cases.

**Independent Test**: Executar a suíte de testes automatizados e o pipeline global em modo SMOKE.

---

## Edge Cases

- IF recall improvement is zero across all classes, THEN Spearman correlation SHALL safely return 0.0 without divide-by-zero errors.
- IF confusion matrix is symmetric or sparse, THEN Ward linkage computation SHALL apply a small epsilon to distance diagonal avoiding zero variances.
- IF only a single ratio is present in the results, THEN the interaction plot SHALL render single-point bars instead of lines without failing.
- IF output directory does not exist, THEN the generator SHALL create parent directories automatically.

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| -------------- | ----- | ----- | ------ |
| FIG-03 | P1: Alocação Amostral vs. Ganho Marginal de Recall | Tasks | Pending |
| FIG-04 | P1: Dendrograma Comparativo Ward vs. Semântica | Tasks | Pending |
| FIG-05 | P1: Curvas de Interação Método x Severidade | Tasks | Pending |
| INT-01 | P1: Integração no Manuscrito LaTeX, Notebook 05 e Pipeline Global | Tasks | Pending |

**Coverage:** 4 total, 4 mapped to tasks, 0 unmapped.

---

## Success Criteria

- [ ] FIG-03 (`fig_class_exposure_vs_delta_recall`) gerada em `.pdf` e `.png` (300 DPI) com correlação de Spearman anotada.
- [ ] FIG-04 (`fig_ward_vs_semantic_dendrogram`) gerada em `.pdf` e `.png` (300 DPI) com folhas e ramos coloridos pelas 20 superclasses do CIFAR-100.
- [ ] FIG-05 (`fig_interaction_ratio_methods`) gerada em `.pdf` e `.png` (300 DPI) com painel duplo de F1 e AUPRC vs. $\rho$ e bandas IC 95%.
- [ ] Manuscrito `paper/main.tex` atualizado com seções e blocos `\begin{figure}` para FIG-03, FIG-04 e FIG-05.
- [ ] Orquestrador `generate_all_paper_artifacts()` e `notebooks/05_paper_artifacts.ipynb` integrados gerando as novas figuras.
- [ ] Suíte `tests/test_p1_expansion_figures.py` aprovada com 100% de sucesso.
