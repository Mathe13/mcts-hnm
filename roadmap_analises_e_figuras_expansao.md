# Roadmap de Expansão Analítica e Novas Figuras — MCTS-HNM

**Projeto:** Monte Carlo Tree Search for Hierarchical Hard Negative Mining (MCTS-HNM)  
**Nível Acadêmico:** Seminário / Dissertação de Mestrado (PPGC/UFRGS) & Submissão para Conferência (ICLR / NeurIPS / ICML)  
**Versão:** 3.0  
**Data:** Novembro de 2026  
**Status:** Proposto / Pronto para Execução  

---

## 1. Visão Geral e Motivação

Com os fundamentos algorítmicos (P0, P1, P2) corrigidos, alinhados e validados com 100% de testes aprovados, este roadmap estabelece um conjunto de **novas figuras, diagnósticos e análises empíricas** para enriquecer a discussão científica do trabalho.

O objetivo é transformar os dados já coletados pela instrumentação moderna (`ExposureTracker`, custos computacionais, snapshots temporais de visitação, calibração em 4 quadrantes e ablações fatoriais) em **evidências visuais irrefutáveis** que respondem diretamente às principais críticas de revisores acadêmicos.

---

## 2. Princípios de Execução e Diretrizes Metodológicas

1. **Aproveitamento de Dados Existentes:** Priorizar figuras cujos dados e telemetrias já são gerados pelos runners (`exposure_diagnostics`, `computational_costs`, `visitation_checkpoints`), evitando retrabalho computacional desnecessário.
2. **Isolamento e Reprodutibilidade:** Todas as novas rotinas de plotagem devem residir em `src/utils/paper_artifacts.py` e suportar isolamento total entre modos:
   - **Modo SMOKE:** Geração com dados rápidos/sintéticos, marca d'água explícita e saída em `results/smoke/paper/figures/`.
   - **Modo REAL:** Geração estrita sem sintetização a partir de `results/real/`, salvando em `paper/figures/`.
3. **Formato Duplo Obrigatório:** Todas as figuras devem ser exportadas simultaneamente em vetor (`.pdf`) para publicação e em alta resolução rasterizada (`.png`, 300 DPI) para inspeção interativa nos notebooks.
4. **Alinhamento com Manuscrito LaTeX:** Cada figura terá um correspondente `\begin{figure}` configurado no manuscrito `paper/main.tex`.

---

## 3. Matriz de Novas Figuras e Análises

| ID | Nome do Artefato | Prioridade | Arquivo Alvo | Questão Científica / Hipótese |
| :--- | :--- | :---: | :--- | :--- |
| **FIG-01** | **Fronteira de Pareto: Custo vs. Desempenho** | **P0** | `fig_pareto_cost_vs_accuracy` | *O overhead de busca do MCTS se paga em eficiência amostral vs. Uniforme e OHEM?* |
| **FIG-02** | **Dinâmica de Entropia e Cobertura (Cenário 2)** | **P0** | `fig_coverage_entropy_dynamics` | *Como a variante Coverage-Balanced impede a inanição e o colapso de classes difíceis?* |
| **FIG-03** | **Alocação Amostral vs. Ganho Marginal de Recall** | **P1** | `fig_class_exposure_vs_delta_recall` | *O MCTS direciona esforço proporcionalmente às classes onde o benefício de aprendizado é maior?* |
| **FIG-04** | **Dendrograma Comparativo: Ward vs. Semântica** | **P1** | `fig_ward_vs_semantic_dendrogram` | *Onde o agrupamento empírico da CNN diverge da taxonomia ontológica humana?* |
| **FIG-05** | **Curvas de Interação Método $\times$ Severidade ($\rho$)** | **P1** | `fig_interaction_ratio_methods` | *A vantagem relativa do MCTS cresce quando o desbalanceamento se torna mais extremo?* |
| **FIG-06** | **Curvas Precision-Recall com Bandas de Variância** | **P2** | `fig_precision_recall_curves` | *Em qual regime de recall o MCTS supera os baselines (evitando falsos positivos)?* |
| **FIG-07** | **Trade-off dos 4 Quadrantes de Avaliação** | **P2** | `fig_evaluation_4quadrants_comparison` | *Qual é o impacto transparente da calibração de limiar e seleção de checkpoint?* |

---

## 4. Detalhamento Técnico das Novas Análises

### Fase 1: Eficiência Amostral e Dinâmica de RL (P0 — Alta Prioridade)

#### 4.1 FIG-01: Fronteira de Pareto de Custo Computacional vs. Desempenho
- **Motivação:** Responder de forma contundente à objeção sobre o custo computacional do MCTS.
- **Estrutura:**
  - **Eixo X:** Custo computacional total (amostras com gradiente calculadas ou tempo de parede em segundos).
  - **Eixo Y:** Desempenho (Minority F1 para Cenário 1; Macro-F1 / Worst-10 Recall para Cenário 2).
  - **Pontos com Barras de Erro ($\pm 1\ \text{std}$):**
    - `uniform`
    - `ohem`
    - `mcts_strict`
    - `mcts_coverage_balanced`
    - `mcts_enhanced`
  - **Linha de Pareto:** Conecta as estratégias não dominadas.
- **Implementação:**
  - Função em `src/utils/paper_artifacts.py`: `generate_pareto_cost_performance_plot(manifest, output_dir, ...)`.
  - Dados de entrada: Extraídos de `computational_costs` (`elapsed_seconds`, `gradient_samples`, `total_updates`) persistidos nos payloads de run.

#### 4.2 FIG-02: Evolução Temporal da Entropia de Visitação e Cobertura de Classes (Cenário 2)
- **Motivação:** Ilustrar quantitativamente a causa raiz do problema do Cenário 2 e a solução via política balanceada.
- **Estrutura (Painel Duplo):**
  - **Subplot (a) — Entropia de Visitação:** Entropia de Shannon normalizada $H(t)/H_{\max}$ ao longo dos 1.000 passos.
    - Curva `uniform`: Permanece em $\approx 1.0$ (amostragem totalmente aleatória).
    - Curva `mcts_strict`: Decai precocemente para $< 0.35$ (colapso prematuro de exploração).
    - Curva `mcts_enhanced`: Patamar intermediário ($\approx 0.50$).
    - Curva `mcts_coverage_balanced`: Estabiliza em patamar ótimo ($\approx 0.75 - 0.85$).
  - **Subplot (b) — Cobertura Cumulativa de Classes:** % de classes com pelo menos $k \ge 1$ e $k \ge 5$ amostragens ao longo dos passos.
- **Implementação:**
  - Função em `src/utils/paper_artifacts.py`: `generate_coverage_entropy_dynamics_plot(data, output_dir, ...)`.
  - Dados de entrada: Extraídos do histórico de checkpoints do `ExposureTracker`.

---

### Fase 2: Estrutura da Árvore e Geometria da Confusão (P1 — Média-Alta Prioridade)

#### 4.3 FIG-03: Alocação Amostral vs. Ganho Marginal de Recall por Classe
- **Motivação:** Demonstrar o funcionamento do MCTS como um currículo ativo adaptativo.
- **Estrutura:**
  - **Eixo X:** Frequência relativa de amostragem pelo MCTS ($N_{\text{visitas}}(c) / \sum N$).
  - **Eixo Y:** Ganho marginal de recall ($\Delta \text{Recall}(c) = \text{Recall}_{\text{MCTS}}(c) - \text{Recall}_{\text{Uniform}}(c)$).
  - **Pontos:** As 100 classes do CIFAR-100.
  - **Cores / Marcadores:** Categorizados por faixa de dificuldade da classe (entropia de predição inicial obtida na regressão logística de validação).
  - **Anotações Estatísticas:** Coeficiente de correlação de postos de Spearman ($\rho_s$) e $p$-valor.
- **Implementação:**
  - Função em `src/utils/paper_artifacts.py`: `generate_class_exposure_vs_delta_recall_plot(data, output_dir, ...)`.

#### 4.4 FIG-04: Dendrograma Comparativo Ward vs. Semântica
- **Motivação:** Visualizar as divergências entre a ontologia humana e o espaço de representação perceptual da rede neural.
- **Estrutura:**
  - Dendrograma hierárquico construído via SciPy a partir da matriz de distância de Ward condensada.
  - Folhas e ramos coloridos de acordo com as 20 superclasses ontológicas do CIFAR-100.
  - Destaques visuais (caixas/anotações) nos pontos onde classes pertencentes a superclasses distintas foram agregadas pela similaridade visual (ex: mamíferos marinhos próximos a peixes).
- **Implementação:**
  - Função em `src/utils/paper_artifacts.py`: `generate_ward_vs_semantic_dendrogram_plot(confusion_matrix, taxonomy, output_dir, ...)`.

---

### Fase 3: Robustez ao Desbalanceamento Extremo (P1 — Média-Alta Prioridade)

#### 4.5 FIG-05: Curvas de Interação Método $\times$ Severidade do Desbalanceamento
- **Motivação:** Comprovar a hipótese de que a superioridade do MCTS se acentua à medida que o desbalanceamento se torna mais severo.
- **Estrutura:**
  - **Eixo X:** Razão de desbalanceamento $\rho \in [1:10, 1:50, 1:100]$ (escala logarítmica).
  - **Eixo Y:** Minority F1-Score (subplot a) e AUPRC (subplot b).
  - **Linhas com Bandas de Confiança (IC 95%):**
    - `Uniform Random` (decaimento acentuado com o aumento do desbalanceamento).
    - `OHEM`
    - `MCTS Strict`
    - `Enhanced MCTS-HNM` (decaimento mais suave e resiliente).
  - **Anotação:** Estatística do teste $F$ de interação método $\times$ razão calculada em `compute_interaction_effects()`.
- **Implementação:**
  - Função em `src/utils/paper_artifacts.py`: `generate_interaction_ratio_plot(data, output_dir, ...)`.

#### 4.6 FIG-06: Curvas Precision-Recall Agregadas com Bandas de Variância
- **Motivação:** Complementar as tabelas escalares de AUPRC com as curvas $P(R)$ contínuas.
- **Estrutura:**
  - Curvas de Precisão vs. Recall com interpolação linear em grade de 100 pontos de recall $[0.0, 1.0]$.
  - Linhas sólidas para a média das 5 seeds e áreas sombreadas correspondendo a $\pm 1\ \text{std}$.
  - Destaque no ponto de operação do limiar calibrado ($\tau_{\text{val}}$).
- **Implementação:**
  - Função em `src/utils/paper_artifacts.py`: `generate_precision_recall_curves_plot(runs, output_dir, ...)`.

---

### Fase 4: Transparência Metodológica (P2 — Média Prioridade)

#### 4.7 FIG-07: Superfície de Trade-off dos 4 Quadrantes de Avaliação
- **Motivação:** Demonstrar de forma visualmente imediata que os ganhos do MCTS são consistentes em todos os critérios de avaliação (limiar fixo 0.5 vs. calibrado; checkpoint final vs. selecionado por validação).
- **Estrutura:**
  - Gráfico de barras agrupadas em 4 quadrantes para cada método:
    1. Checkpoint Final com $\tau = 0.5$
    2. Checkpoint Final com $\tau_{\text{val}}$
    3. Melhor Checkpoint de Validação com $\tau = 0.5$
    4. Melhor Checkpoint de Validação com $\tau_{\text{val}}$
  - Permite verificar visualmente o ganho de calibração isolado e o ganho de seleção de checkpoint isolado.
- **Implementação:**
  - Função em `src/utils/paper_artifacts.py`: `generate_quadrant_evaluation_plot(data, output_dir, ...)`.

---

## 5. Cronograma e Etapas de Execução Sugeridas

```mermaid
flowchart TD
    subgraph Fase 1: Eficiência & Entropia (P0)
        A1[Implementar gerador de Pareto FIG-01] --> A2[Implementar dinâmica de entropia FIG-02]
        A2 --> A3[Integrar em paper_artifacts.py e Notebook 05]
        A3 --> A4[Testes unitários e verificação SMOKE]
    end

    subgraph Fase 2: Estrutura da Árvore & Desbalanceamento (P1)
        B1[Implementar curva de interação FIG-05] --> B2[Implementar alocação vs recall FIG-03]
        B2 --> B3[Implementar dendrograma Ward FIG-04]
        B3 --> B4[Atualizar Notebook 03 e Notebook 05]
    end

    subgraph Fase 3: Curvas PR & 4 Quadrantes (P2)
        C1[Implementar curvas PR agregadas FIG-06] --> C2[Implementar gráfico dos 4 quadrantes FIG-07]
        C2 --> C3[Compilar manuscrito main.tex com novas seções]
    end

    Fase 1 --> Fase 2 --> Fase 3
```

---

## 6. Critérios de Sucesso e Aceite

1. **Compilação e Integridade:**
   - O comando `.venv/bin/pytest -q` continua executando com **100% de aprovação** em todos os 265+ testes.
   - O manuscrito `paper/main.tex` compila sem erros via `pdflatex` gerando o PDF atualizado.
2. **Isolamento Estrito:**
   - Nenhum artefato do modo SMOKE sobrescreve arquivos em `paper/figures/` ou `paper/tables/`.
   - Todas as figuras em SMOKE mantêm a tarja de aviso obrigatória.
3. **Reproduzibilidade nos Notebooks:**
   - Notebooks `03_evaluation.ipynb` e `05_paper_artifacts.ipynb` executam de ponta a ponta sem avisos de chave faltante ou erros de tipo.
