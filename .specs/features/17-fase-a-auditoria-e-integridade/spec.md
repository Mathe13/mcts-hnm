# F17: Fase A — Auditoria e Integridade do Protocolo Experimental

**Approved**: 2026-11-11 - user: "sim"

## Problem Statement

O documento `roadmap_mcts_hnm_proxima_run_v2.md` estabelece a necessidade de uma revisão metodológica profunda e auditoria de integridade do pipeline experimental antes da execução de novos benchmarks científicos (Fase A; P0 bloqueante). A auditoria do código atual revelou vulnerabilidades críticas no protocolo:

1. **Definição imprecisa de desbalanceamento (Item A2)**: Anteriormente, `pos_fraction=0.02` era interpretado como razão 1:50, quando na realidade apenas subsamplava 2% dos positivos (9 de 450) mantendo todos os 44.550 negativos (razão real ~1:4950). O protocolo exige separar explicitamente a razão real treino ($N_{\text{neg}}/N_{\text{pos}}$), a contagem de positivos únicos fixos (ex.: 80 positivos por classe/seed) e a composição do minibatch (ex.: 16 pos / 48 neg), preparando as razões empíricas 1:50, 1:100 e 1:500 com negativos aninhados ($N_{50} \subset N_{100} \subset N_{500}$) e sem reamostragem artificial do conjunto de teste oficial.
2. **Vazamento e contaminação de splits de dados (Item A3)**: Em `trainer.py`, os loops de treinamento (`train_scenario1`, `train_scenario2`) avaliavam métricas a cada `eval_every` passos diretamente sobre o conjunto de teste (`test_features`). Além disso, em `benchmark_runner.py`, a matriz de confusão para árvores Ward admitia fallback silencioso para `test_features` quando `val_features` não era fornecido. O protocolo exige segregação estrita: `train_fit` para gradientes/mineração, `validation` para curvas durante o treino, checkpoints e limiares de F1, e `test` exclusivamente na avaliação final.
3. **Equivalência e consistência entre runners (Item A4)**: Runners e rotinas de treino (`run_benchmark`, `run_sensitivity_ablation`, `train_scenario1`) divergiam na instanciação de hiperparâmetros (ex.: propagação de `c`, `alpha`, splits e dispositivos), impedindo comparabilidade determinística entre protocolos idênticos.
4. **Granularidade da persistência e pareamento estatístico (Item A5)**: Resultados eram agregados em arquivos consolidados sem serialização individual atômica por `run_id` (`scenario`, classe, razão, árvore, estratégia, seed, commit), arriscando perda de histórico por seed e pareamento estatístico inadequado.
5. **Fundamentação matemática do clustering hierárquico (Item A6)**: A transformação de dissimilaridades de confusão via `sqrt(2*distance)` para Ward pressupunha erroneamente que qualquer matriz normalizada torna-se euclidiana. É necessário verificar a euclidianidade formal ou aplicar Ward diretamente sobre vetores euclidianos válidos (perfis de confusão normalizados por linha), mantendo Average Linkage (UPGMA) para matrizes de dissimilaridade gerais.
6. **Falta de inventário e rastreabilidade histórica (Item A1)**: Ausência de `docs/experimental_audit.md` e manifestos por execução detalhando ambiente, hardware, hashes de embeddings, splits e integridade imutável dos experimentos históricos.

A Fase A é um pré-requisito mandatório e bloqueante (Gate A): nenhuma fase subsequente (B, C, D) pode ser executada enquanto a integridade do protocolo não for formalmente comprovada por testes determinísticos.

---

## Out of Scope

- Executar benchmarks em escala real de múltiplas horas na GPU (unitários e integração rodam em `ExecutionMode.SMOKE` e fixtures rápidas de `REAL` dentro do SLA < 60s).
- Alterar as equações matemáticas fundamentais do MCTS UCB1 ou Min-Max Q-normalization.
- Introduzir novas políticas de amostragem ou correções algorítmicas adicionais (reservadas para as Fases B e C).
- Regenerar arquivos de embeddings ResNet-18 existentes sem justificativa metodológica.
- Modificar o conjunto oficial de teste do CIFAR-100 ou sua prevalência natural (~1:99).

---

## Assumptions & Open Questions

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| Base de positivos únicos em `train_fit` | Fixar 80 positivos por classe-alvo/seed em `train_fit` para CIFAR-100 | Deixa 370 positivos disponíveis para `validation` (50) e calibração de árvore sem reposição e sem esgotamento de dados | y |
| Razões reais de negativos no treino | Suportar explicitamente razões reais `target_neg_pos_ratio` de 50, 100 e 500 | Correspondem a 4.000, 8.000 e 40.000 negativos para 80 positivos únicos, cobrindo o grid exigido | y |
| Amostragem aninhada de negativos | Subconjuntos aninhados $N_{50} \subset N_{100} \subset N_{500}$ estratificados entre as 99 classes negativas | Reduz a variância composicional de negativos entre diferentes níveis de severidade de desbalanceamento | y |
| Composição do minibatch | Manter $K_{\text{pos}}=16$ e $K_{\text{neg}}=48$ fixos (razão 1:3 no minibatch) para isolar o efeito do pool de dados | Separa o impacto do conjunto disponível no treino da prevalência instantânea do gradiente | y |
| Segregação estrita de dados | `validation` obrigatória em treinos; lançar `ValueError` se `val_features` for omitido em `run_benchmark` e runners em vez de usar `test` | Elimina vazamento de dados de teste para curvas de treinamento, tuning de limiar ou estimativa de matriz de confusão | y |
| Vetores válidos para Ward clustering | Aplicar Ward sobre linhas da matriz de confusão normalizada por linha ($p_i \in \mathbb{R}^{100}$) | Garante vetores euclidianos genuínos em $\mathbb{R}^K$, eliminando inversões de Lance-Williams em Ward | y |
| Granularidade de persistência | Persistir cada execução com identificador único `run_id` e metadados de proveniência | Garante pareamento exato por seed, split e orçamento, prevenindo sobrescritas acidentais | y |
| Documentação de auditoria | Criar `docs/experimental_audit.md` inventariando os artefatos históricos e seus hashes | Atende ao requisito A1 de auditoria e imutabilidade dos históricos | y |

**Open questions:** none - all resolved and recorded above.

---

## User Stories

### P1: Inventário e Congelamento do Histórico Experimental (A1) (`AUD-01`, `AUD-02`) ⭐ MVP

**User Story**: Como pesquisador auditando o repositório, quero um inventário documentado em `docs/experimental_audit.md` e um gerador de manifesto de execução para que todos os resultados históricos sejam imutáveis e qualquer métrica seja rastreável até suas condições exatas.

**Why P1**: Requisito A1 do roadmap e pré-condição de reprodutibilidade para não mesclar campanhas experimentais distintas.

**Acceptance Criteria**:

1. The system SHALL maintain a comprehensive audit document at `docs/experimental_audit.md` cataloging legacy execution runs, file paths, embedding hashes, and immutability invariants.
2. WHEN `generate_run_manifest(run_id, config, seed, paths)` is called THEN the system SHALL produce an auditable dictionary containing git commit hash, timestamp, hardware details, dependency versions, embedding checksums, split checksums, and tree configuration.
3. IF any historical artifact in `results/legacy/` is modified by a new experiment THEN the system SHALL raise a `PermissionError` or reject execution to preserve historical immutability.

**Independent Test**: Executar teste de integridade documental e unitário de geração de manifesto em `tests/test_phase_a_audit.py`.

---

### P1: Construtor Estratificado de Desbalanceamento com Negativos Aninhados (A2) (`SPLIT-01`, `SPLIT-02`, `SPLIT-03`, `SPLIT-04`) ⭐ MVP

**User Story**: Como pesquisador investigando desbalanceamento extremo, quero um construtor de splits que separe a razão real treino ($N_{\text{neg}}/N_{\text{pos}}$), a contagem de positivos únicos e a fração do batch, fornecendo razões exatas 1:50, 1:100 e 1:500 com positivos idênticos e negativos aninhados ($N_{50} \subset N_{100} \subset N_{500}$) sem alterar o teste oficial.

**Why P1**: Corrige a falha metodológica crítica onde `pos_fraction=0.02` era confundido com 1:50.

**Acceptance Criteria**:

1. WHEN `build_stratified_imbalanced_split(fine_labels, target_class, target_positive_count, target_neg_pos_ratio, seed, nested_negative_parent=None)` is called THEN the system SHALL return a split object where `len(positive_indices) == target_positive_count` and `len(negative_indices) == target_positive_count * target_neg_pos_ratio`.
2. WHEN splits are generated for ratios 50, 100, and 500 under the same seed and target class using nested sampling THEN the system SHALL ensure that `positive_indices` are identical across all three ratios, and that `negative_indices_50` is a strict subset of `negative_indices_100`, which is a strict subset of `negative_indices_500`.
3. IF the requested `target_positive_count` or `target_positive_count * target_neg_pos_ratio` exceeds the available samples in `fine_labels` THEN the system SHALL raise a `ValueError` with an explicit diagnostic message and never silently alter the requested ratio.
4. The system SHALL leave the official CIFAR-100 test split un-resampled with its natural One-vs-Rest prevalence (~1:99) for all evaluation runs.

**Independent Test**: Executar testes em `tests/test_phase_a_splits.py` verificando igualdade de positivos, aninhamento exato de negativos, razões numéricas exatas (50, 100, 500) e falha explícita sob contagens inviáveis.

---

### P1: Segregação Estrita de Dados e Prevenção de Vazamento (A3) (`DATA-01`, `DATA-02`, `DATA-03`) ⭐ MVP

**User Story**: Como avaliador da integridade científica, quero que `train_fit`, `validation` e `test` tenham fronteiras impermeáveis, de modo que dados de teste nunca sejam usados durante o treinamento, tuning de limiar ou calibração de árvore.

**Why P1**: Elimina vazamento de dados de teste identificado em `trainer.py` e fallbacks indevidos em `benchmark_runner.py`.

**Acceptance Criteria**:

1. WHILE training in `train_scenario1` or `train_scenario2`, WHEN periodic evaluations occur at `eval_every` steps THEN the system SHALL compute metrics strictly against `val_features` and `val_labels` and NEVER against test sets.
2. IF `val_features` or `val_labels` is missing or None when invoking `train_scenario1`, `train_scenario2`, or `run_benchmark` THEN the system SHALL raise a `ValueError` and SHALL NOT fall back to test features.
3. WHEN estimating a confusion matrix for data-driven tree construction THEN the system SHALL compute confusion strictly on validation or internal out-of-fold calibration data and SHALL raise a `ValueError` if test data is supplied for tree construction.

**Independent Test**: Executar testes de integração em `tests/test_phase_a_data_segregation.py` verificando que a omissão de validação gera erro imediato e que tensores de teste não são acessados no loop de treino.

---

### P1: Equivalência e Governança entre Runners (A4) (`RUN-01`, `RUN-02`) ⭐ MVP

**User Story**: Como desenvolvedor de algoritmos de RL, quero que `run_benchmark`, `run_sensitivity_ablation` e rotinas de treino compartilhem a mesma resolução de configurações e propagação de hiperparâmetros para garantir que protocolos equivalentes produzam resultados consistentes.

**Why P1**: Requisito A4 para eliminar discrepâncias silenciosas de hiperparâmetros (`c`, `alpha`, batches, seeds).

**Acceptance Criteria**:

1. WHEN configuring an experiment via `ResolvedTrainingConfig` THEN the system SHALL resolve and validate all hyperparameters (`c`, `alpha`, batch sizes, learning rate, num_steps, seed, loss function) into an immutable structure shared across all runners.
2. WHEN `run_benchmark` and `run_sensitivity_ablation` are executed with identical configurations, seeds, and splits THEN the system SHALL execute the identical underlying training pipeline and log equivalent step metrics within floating-point tolerance.

**Independent Test**: Executar teste de regressão em `tests/test_phase_a_runner_equivalence.py` comparando execuções determinísticas curtas pelos dois caminhos.

---

### P1: Persistência por Run ID e Pareamento Estatístico (A5) (`STORE-01`, `STORE-02`) ⭐ MVP

**User Story**: Como estatístico analisando os experimentos, quero que cada execução seja salva com um `run_id` atômico e que comparações estatísticas sejam pareadas por seed, split e orçamento.

**Why P1**: Evita sobrescrita de execuções anteriores e garante testes pareados válidos.

**Acceptance Criteria**:

1. WHEN saving benchmark results THEN the system SHALL serialize an individual JSON record per `run_id` indexed by scenario, target class, imbalance ratio, tree type, strategy, seed, and git commit hash.
2. WHEN computing paired statistical comparisons between strategies THEN the system SHALL pair results strictly by matching seed, split indices, target class, and computational budget, raising a `ValueError` if keys do not match.

**Independent Test**: Executar testes em `tests/test_phase_a_persistence.py` validando geração de `run_id`, salvamento isolado e rejeição de pareamento com chaves incompatíveis.

---

### P1: Validação Matemática da Construção de Hierarquias (A6) (`TREE-01`, `TREE-02`) ⭐ MVP

**User Story**: Como cientista da computação, quero que a árvore hierárquica Ward seja construída exclusivamente sobre vetores euclidianos válidos e que Average Linkage esteja disponível para matrizes gerais, para que as propriedades do agrupamento sejam formalmente demonstráveis.

**Why P1**: Requisito A6 eliminando premissas infundadas sobre `sqrt(2*distance)` em matrizes de dissimilaridade não euclidianas.

**Acceptance Criteria**:

1. WHEN constructing a Ward confusion hierarchy via `build_confusion_tree(..., linkage_method="ward")` THEN the system SHALL compute Ward linkage over mathematically valid Euclidean row-normalized confusion vectors in $\mathbb{R}^K$ or verify Euclidean metric embedding before clustering.
2. WHEN constructing a tree via `build_confusion_tree(..., linkage_method="average")` THEN the system SHALL apply Average Linkage (UPGMA) directly to the confusion distance matrix without requiring Euclidean metric embedding.

**Independent Test**: Executar testes em `tests/test_phase_a_tree_validation.py` verificando propriedades euclidianas, ausência de inversões de altura e monotonicidade dendrogramática.

---

### P1: Verificação Formal do Gate A (`GATE-01`) ⭐ MVP

**User Story**: Como Tech Lead, quero uma verificação de integração do Gate A que valide conjuntamente splits, ausência de vazamento de teste, equivalência de runners, persistência e árvores antes de autorizar a Fase B.

**Why P1**: Critério de parada formal do roadmap (Gate A).

**Acceptance Criteria**:

1. WHEN `verify_gate_a()` is executed THEN the system SHALL evaluate all Fase A invariants (A1 a A6) and exit with code 0 only if 100% of the integrity checks succeed.
2. IF any Fase A check fails during `verify_gate_a()` THEN the system SHALL exit with a non-zero status code and print an explicit diagnostic report detailing the blocked condition.

**Independent Test**: Executar `verify_gate_a` em `tests/test_phase_a_gate.py`.

---

## Requirement Traceability

| Requirement ID | Description | Component | Verification Method | Status |
| -------------- | ----------- | --------- | ------------------- | ------ |
| AUD-01 | Documentação de auditoria histórica e integridade imutável | `docs/experimental_audit.md` | Inspeção de documentação e teste unitário | pending |
| AUD-02 | Geração de manifesto de proveniência por execução | `src/utils/experiment_config.py` | `tests/test_phase_a_audit.py` | pending |
| SPLIT-01 | Construtor estratificado com razão real e positivos únicos | `src/data/splits.py` | `tests/test_phase_a_splits.py` | pending |
| SPLIT-02 | Negativos aninhados ($N_{50} \subset N_{100} \subset N_{500}$) e positivos compartilhados | `src/data/splits.py` | `tests/test_phase_a_splits.py` | pending |
| SPLIT-03 | Falha explícita sob contagens de amostras inviáveis | `src/data/splits.py` | `tests/test_phase_a_splits.py` | pending |
| SPLIT-04 | Preservação do teste oficial com prevalência natural | `src/data/splits.py` | `tests/test_phase_a_splits.py` | pending |
| DATA-01 | Avaliação periódica durante treino restrita a validação | `src/training/trainer.py` | `tests/test_phase_a_data_segregation.py` | pending |
| DATA-02 | Rejeição imediata de omissão de validação sem fallback para teste | `src/training/benchmark_runner.py` | `tests/test_phase_a_data_segregation.py` | pending |
| DATA-03 | Calibração de árvores a partir de treino/validação, nunca teste | `src/trees/confusion_tree.py` | `tests/test_phase_a_data_segregation.py` | pending |
| RUN-01 | Resolução unificada de configuração e hiperparâmetros | `src/training/runner_config.py` | `tests/test_phase_a_runner_equivalence.py` | pending |
| RUN-02 | Equivalência determinística entre runners em protocolos iguais | `src/training/benchmark_runner.py` | `tests/test_phase_a_runner_equivalence.py` | pending |
| STORE-01 | Persistência individual atômica por `run_id` e metadados | `src/training/benchmark_runner.py` | `tests/test_phase_a_persistence.py` | pending |
| STORE-02 | Pareamento estatístico estrito por seed, split e orçamento | `src/utils/statistics.py` | `tests/test_phase_a_persistence.py` | pending |
| TREE-01 | Ward clustering matematicamente fundamentado sobre vetores euclidianos | `src/trees/confusion_tree.py` | `tests/test_phase_a_tree_validation.py` | pending |
| TREE-02 | Suporte formal a Average Linkage para matrizes de dissimilaridade | `src/trees/confusion_tree.py` | `tests/test_phase_a_tree_validation.py` | pending |
| GATE-01 | Verificador e gate determinístico de parada da Fase A | `src/utils/gate_a_verifier.py` | `tests/test_phase_a_gate.py` | pending |
