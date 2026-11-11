# F20: Fase D — Campanha Científica Final e Artefatos

**Approved**: 2026-11-11 - user: "sim"

## Problem Statement

Com a validação dos Gates A (integridade metodológica de dados e árvores), B (observabilidade, calibração e viabilidade 1:500) e C (ablação fatorial 2x2, controle Uniform Grouped e seletores baselines), o roadmap revisado (`roadmap_mcts_hnm_proxima_run_v2.md`) estabelece a **Fase D (P1/P2)** como o fechamento científico da campanha experimental.

O objetivo da Fase D é executar a campanha final sob um protocolo rigorosamente congelado e pré-registrado, gerando tabelas, figuras, análises estatísticas pareadas e artefatos reproduzíveis sem qualquer inferência espúria ou dado sintético:

1. **Congelamento do Protocolo Experimental (Item D1)**: Pré-registrar internamente hiperparâmetros, classes-alvo, splits aninhados, sementes pareadas (5 seeds: 42, 43, 44, 45, 46), orçamentos de passos, checkpoints (`last` e `best_validation`) e hipóteses causais. No Cenário 1 (binário), avaliar as razões 1:50, 1:100 e 1:500 com os mesmos 80 positivos únicos sob o teste oficial não reamostrado (~1:99). No Cenário 2 (multiclasse), avaliar o grid completo: Uniform, Uniform Grouped, Strict, QNorm-only, DualStream-only e Enhanced.
2. **Métricas Primárias e Eficiência Computacional (Item D2)**:
   - No Binário: reportar AUPRC como métrica primária de ranking sob desbalanceamento severo, acompanhada de F1 minoritário calibrado em validação ($\tau_{\text{val}}$), precision, recall e FPR@95.
   - No Multiclasse: reportar top-1 accuracy, macro-F1, balanced accuracy, Worst-10 recall e entropia normalizada de exposição.
   - Em Eficiência: computar updates, exemplos utilizados em gradientes, forwards adicionais de mineração/OHEM, wall-clock time e pico de memória com hardware documentado. Distinguir explicitamente `best_validation` e `last`.
3. **Inferência Estatística Pareada e Interações (Item D3)**:
   - Diferenças pareadas por seed e split dentro de cada razão;
   - Análise de interação método $\times$ razão para contrastes entre 1:50, 1:100 e 1:500;
   - Cálculo de intervalos de confiança e tamanho de efeito (Cohen's d). A unidade de inferência primária é a execução/seed.
4. **Rastreabilidade de Artefatos e Visualizações (Item D4)**:
   - Notebooks `03` e `05` parametrizados para consumo exclusivo de dados REAL válidos com `.meta.json`.
   - Geração automática de: curvas de validação com intervalos de incerteza, heatmap de exposição por classe, matriz de ablação $2 \times 2$ (QNorm $\times$ DualStream), curvas de trade-off desempenho vs. custo e tabelas LaTeX consolidadas sem lacunas simuladas.
5. **Gate D**: Verificador automatizado de integridade garantindo que todos os artefatos de publicação e tabelas provêm de resultados empíricos reais e que nenhuma conclusão extrapola os experimentos efetivamente executados.

---

## Out of Scope

- Executar benchmarks REAL exaustivos de 5.000 passos em GPU de forma automática durante o teste de unidade/integração do repositório (o código deve suportar execução automatizada rápida via `ExecutionMode.SMOKE` com SLA < 60s e permitir o disparo da campanha REAL pelo usuário no JupyterHub).
- Modificar o conjunto oficial de teste do CIFAR-100 ou sua distribuição natural.
- Introduzir novas heurísticas ou afrouxar restrições metodológicas dos Gates A, B e C.

---

## Assumptions & Open Questions

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| Sementes pareadas da campanha | 5 seeds fixas: 42, 43, 44, 45, 46 | Garante pareamento estrito entre todos os métodos e razões | y |
| Métrica primária binária | AUPRC (Area Under Precision-Recall Curve) | Métrica recomendada pelo roadmap para ranking em desbalanceamento severo | y |
| Avaliação em teste fixo | Teste oficial CIFAR-100 (100 positivos, 9.900 negativos) sem reamostragem artificial | Avalia a robustez do modelo treinado sem distorcer a população real de teste | y |
| Orçamento de passos | Tier-1: 500 steps (piloto/comparativo); Campanha principal: configurável (500 ou 5000) | Permite validação ultrarrápida em SMOKE e flexibilidade na execução REAL | y |
| Unidade estatística | Pareamento por seed e split | Evita tratar instâncias de teste como replicações independentes de modelo | y |
| Verificador Gate D | Script que inspeciona artefatos LaTeX, figuras geradas e manifestos de proveniência | Bloqueia afirmações não sustentadas por dados reais antes da submissão do artigo | y |

**Open questions:** none - all requirements established per `roadmap_mcts_hnm_proxima_run_v2.md` lines 164-196.

---

## User Stories

### P1: Congelamento do Protocolo e Orquestrador da Campanha Científica (D1) (`CAMP-01`) ⭐ MVP

**User Story**: Como pesquisador, quero um executor de campanha unificado que orquestre o grid congelado de experimentos para os Cenários 1 e 2 com sementes pareadas, registrando proveniência completa para cada execução.

**Why P1**: Requisito D1 mandatório para garantir que a campanha execute exatamente a matriz pré-registrada sem desvios de configuração.

**Acceptance Criteria**:

1. WHEN executing the scientific campaign runner under Scenario 1, the system SHALL execute the four sampling methods (Uniform, OHEM, Strict, Enhanced) across the configured imbalance ratios (1:50, 1:100, 1:500) using 5 paired seeds with identical positive instances.
2. WHEN executing the scientific campaign runner under Scenario 2, the system SHALL execute the full ablation suite (Uniform, Uniform Grouped, Strict, QNorm-only, DualStream-only, Enhanced) across the 5 paired seeds under identical budgets and architectures.
3. WHILE executing any campaign run, the system SHALL serialize atomic run JSON records with `.meta.json` sidecars containing git commit SHA, execution mode, seed, and timestamps.
4. IF a run execution encounters unhandled exceptions or NaN metrics, the system SHALL abort and log a diagnostic provenance error.

**Independent Test**: Executar testes em `tests/test_phase_d_campaign.py` validando o executor de campanha sob modo smoke com orquestração completa dos Cenários 1 e 2.

---

### P1: Telemetria de Métricas Primárias, Diagnóstico de Eficiência e Checkpoints Duplos (D2) (`METR-01`) ⭐ MVP

**User Story**: Como pesquisador, quero que cada execução da campanha registre AUPRC, F1 calibrado, métricas multiclasse e contabilidade detalhada de eficiência computacional (forwards, updates, tempo), distinguindo checkpoints `last` e `best_validation`.

**Why P1**: Requisito D2 essencial para comparar o desempenho frente ao custo computacional sem viés de parada precoce.

**Acceptance Criteria**:

1. WHILE evaluating Scenario 1 runs, the system SHALL compute and record AUPRC, precision, recall, and minority F1 at calibrated threshold $\tau_{\text{val}}$ on validation data.
2. WHILE evaluating Scenario 2 runs, the system SHALL compute and record macro-F1, top-1 accuracy, balanced accuracy, Worst-10 recall, and normalized exposure entropy.
3. WHEN a training run finishes, the system SHALL record computational cost telemetry including total step updates, sample gradient counts, additional forward passes, and elapsed wall-clock seconds.
4. WHEN reporting benchmark summaries, the system SHALL maintain distinct metric records for both `last` and `best_validation` checkpoints.

**Independent Test**: Executar testes em `tests/test_phase_d_metrics.py` validando a presença de AUPRC, métricas de eficiência e diferenciação de checkpoints.

---

### P1: Inferência Estatística Pareada e Efeitos de Interação (D3) (`STAT-01`) ⭐ MVP

**User Story**: Como pesquisador analisando dados experimentais, quero calcular diferenças pareadas por seed, intervalos de confiança e efeitos de interação método $\times$ razão, para relatar a significância estatística dos resultados.

**Why P1**: Requisito D3 para sustentação metodológica das alegações científicas do artigo.

**Acceptance Criteria**:

1. WHEN computing statistical comparisons between methods within a given ratio, the system SHALL compute paired differences across identical seeds, reporting mean difference, standard deviation, and Cohen's d effect size.
2. WHEN comparing methods across multiple imbalance ratios, the system SHALL compute interaction statistics (difference of differences) preserving seed pairing.
3. WHEN generating statistical summary tables, the system SHALL include confidence intervals and explicitly report sample size $n$ and individual seed values.

**Independent Test**: Executar testes em `tests/test_phase_d_statistics.py` verificando o pareamento estrito por seed, cálculo de Cohen's d e testes de interação.

---

### P1: Geração Automatizada de Artefatos Científicos e Tabelas LaTeX (D4) (`ARTF-01`) ⭐ MVP

**User Story**: Como autor do artigo, quero gerar automaticamente gráficos de validação, heatmaps de exposição, a matriz de ablação $2 \times 2$ e tabelas LaTeX consolidadas consumindo apenas dados REAL verificados.

**Why P1**: Requisito D4 para eliminar manipulação manual de dados e garantir rastreabilidade completa das figuras do artigo.

**Acceptance Criteria**:

1. WHEN generating publication figures from real benchmark results, the system SHALL generate validation curves with variability bands, exposure distribution heatmaps, and the $2 \times 2$ QNorm $\times$ DualStream ablation chart.
2. WHEN generating LaTeX tables, the system SHALL output publication-ready tables displaying both `last` and `best_validation` metrics, AUPRC, calibrated F1, and computational costs without placeholder defaults.
3. IF real benchmark results are missing or contain unverified synthetic artifacts, the system SHALL reject generation with `FileNotFoundError` or `ValueError`.

**Independent Test**: Executar testes em `tests/test_phase_d_artifacts.py` validando a geração de tabelas e figuras a partir de manifestos de campanha.

---

### P1: Verificador do Gate D (Auditabilidade e Rastreabilidade Final) (`GATE-04`) ⭐ MVP

**User Story**: Como Tech Lead, quero executar o verificador `verify_gate_d()` para comprovar que todos os artefatos derivam de execuções válidas, que o protocolo foi respeitado e que não há dados sintéticos ou conclusões injustificadas.

**Why P1**: Requisito final de conclusão do roadmap (Gate D).

**Acceptance Criteria**:

1. WHEN `verify_gate_d()` is executed on the campaign directory, the system SHALL verify that all required scenarios, ratios, methods, and paired seeds are present and complete.
2. IF any artifact or table references missing runs, unrecorded metrics, or synthetic smoke data in REAL mode, `verify_gate_d()` SHALL report a failure verdict.
3. WHEN all runs are validated, `verify_gate_d()` SHALL generate a signed `campaign_audit_manifest.json` certifying full scientific reproducibility.

**Independent Test**: Executar testes em `tests/test_phase_d_gate.py` validando aprovação do Gate D sob campanhas completas e rejeição sob dados corrompidos ou incompletos.

---

## Requirement Traceability

| Requirement ID | Description | Component | Verification Method | Status |
| -------------- | ----------- | --------- | ------------------- | ------ |
| CAMP-01 | Executor de campanha científica com protocolo congelado e proveniência | `src/training/campaign_runner.py` | `tests/test_phase_d_campaign.py` | pending |
| METR-01 | Métricas primárias (AUPRC), eficiência e checkpoints duplos | `src/training/trainer.py`, `src/utils/metrics.py` | `tests/test_phase_d_metrics.py` | pending |
| STAT-01 | Estatística pareada por seed, Cohen's d e testes de interação | `src/utils/statistics.py` | `tests/test_phase_d_statistics.py` | pending |
| ARTF-01 | Geração de artefatos, heatmaps, matriz 2x2 e tabelas LaTeX | `src/utils/paper_artifacts.py` | `tests/test_phase_d_artifacts.py` | pending |
| GATE-04 | Verificador de integridade e rastreabilidade final do Gate D | `src/utils/gate_d_verifier.py` | `tests/test_phase_d_gate.py` | pending |
