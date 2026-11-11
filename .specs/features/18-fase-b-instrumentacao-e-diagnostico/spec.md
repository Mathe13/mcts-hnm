# F18: Fase B — Instrumentação, Observabilidade e Piloto Diagnóstico

**Approved**: 2026-11-11 - user: "sim"

## Problem Statement

A Fase A estabeleceu com sucesso a integridade metodológica do protocolo experimental (splits estratificados aninhados com 80 positivos únicos, segregação de teste sem leakage, configuração unificada, persistência atômica por `run_id` e árvores euclidianas válidas). 

Com as bases de integridade validadas pelo Gate A, o roadmap revisado (`roadmap_mcts_hnm_proxima_run_v2.md`) define a **Fase B (P0)** como o estágio obrigatório de instrumentação e piloto diagnóstico antes da realização de novos benchmarks em larga escala. A literatura e os experimentos históricos preliminares demonstraram que a política MCTS Enhanced sofre quedas de desempenho em cenários multiclasse e em desbalanceamentos severos, mas as causas desse comportamento permanecem inconclusivas no código atual devido a quatro limitações:

1. **Observabilidade de amostragem incompleta (Item B1)**: O sistema de log atual registra apenas métricas globais esparsas, sem rastrear a distribuição de classes efetivamente selecionadas para os minibatches e utilizadas nos gradientes. Falta quantificar a entropia normalizada de exposição, a cobertura acumulada de classes, a frequência de classes negligenciadas (starvation) e a correlação entre exposição e recall por classe.
2. **Ausência de diagnóstico contínuo da política MCTS (Item B2)**: Os estados internos de busca (recompensa bruta, recompensa suavizada, estimativa de valor $Q$, termos de exploração $U$, scores UCB e contadores de visita) não são inspecionados periodicamente, impedindo verificar se a propagação para ancestrais está correta e se a exploração permanece estritamente positiva ($U > 0$).
3. **Gerenciamento de checkpoints e limiares sem protocolo de validação (Item B3)**: O treinamento não persiste simultaneamente o modelo final (`last`) e o melhor ponto de validação (`best_validation`), dificultando comparar modelos sob o mesmo orçamento fixo $N$ de passos. Além disso, no Cenário 1 (binário), o limiar de decisão deve ser calibrado estritamente sobre a validação ($\tau_{\text{val}} = \arg\max_\tau \text{F1}_{\text{val}}(\tau)$) e aplicado uma única vez ao teste, reportando tanto F1 padrão quanto F1 calibrado e AUPRC.
4. **Falta de um piloto diagnóstico curto e gate de viabilidade para 1:500 (Item B4)**: É necessário um executor de piloto reduzido (2 seeds, $\le 500$ passos) para validar experimentalmente a estabilidade numérica (sem perdas `NaN`/`Inf`), ausência de reposição oculta e viabilidade de tempo da razão 1:500 antes de autorizar execuções REAL extensas.

A Fase B culmina no **Gate B**, assegurando histórico completo e dados empíricos suficientes para discernir entre starvation da política e homogeneidade dos minibatches.

---

## Out of Scope

- Executar benchmarks REAL completos com 10 seeds e 5.000 passos (reservados para as campanhas das Fases C e D).
- Implementar novas heurísticas de recompensa ou novas formulações de UCB (foco estrito no diagnóstico e instrumentação do estado atual).
- Modificar o conjunto oficial de teste do CIFAR-100 ou sua prevalência natural (~1:99).
- Regenerar embeddings da ResNet-18 ou alterar caches sem necessidade metodológica.

---

## Assumptions & Open Questions

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| Registro de observabilidade | Persistir `step_history` detalhado e `exposure_diagnostics` no JSON do `run_id` | Centraliza as métricas por execução de forma autocontida sem espalhar arquivos auxiliares | y |
| Entropia normalizada de exposição | $H_{\text{norm}} = -\sum p_c \ln(p_c) / \ln(K)$ para classes com amostragem não-nula | Fornece métrica adimensional em $[0, 1]$ que mede a dispersão de amostragem entre as $K$ classes | y |
| Checkpoints `last` e `best_validation` | Salvar ambos ao término do orçamento de passos sem interromper o treino prematuramente | Permite comparar o modelo no fim do orçamento fixo versus o ponto de pico de validação sem viés | y |
| Calibração de limiar binário | Calibrar $\tau_{\text{val}} \in [0.01, 0.99]$ no conjunto de validação para maximizar F1 | Garante avaliação rigorosa de detecção de classes raras sem overfitting no teste | y |
| Piloto curto de diagnóstico | Limitar a 2 seeds (ex.: seeds 42 e 43) e 200 a 500 passos em modo de execução controlado | Testa todo o pipeline de ponta a ponta em tempo reduzido (< 3 minutos) antes de benchmarks pesados | y |
| Critério do Gate de Viabilidade 1:500 | Verificar 80 positivos, 40.000 negativos sem reposição externa, métricas finitas e sem NaNs | Valida matematicamente e numericamente o stress test de 1:500 antes de qualquer run completa | y |

**Open questions:** none - all defaults established per roadmap.

---

## User Stories

### P1: Observabilidade Uniforme e Diagnóstico de Exposição de Classes (B1) (`OBS-01`, `OBS-02`) ⭐ MVP

**User Story**: Como pesquisador analisando o comportamento do MCTS, quero registrar passo a passo o histórico de treino e validação, bem como a entropia de exposição das classes e classes não amostradas, para diagnosticar a cobertura amostral.

**Why P1**: Requisito B1 essencial para discernir se a perda de performance multiclasse ocorre por falta de exploração ou viés de gradiente.

**Acceptance Criteria**:

1. WHILE training in Scenario 1 or Scenario 2, WHEN an evaluation interval occurs THEN the system SHALL record in `step_history` the metrics `train_loss`, `val_loss`, `val_F1`, `val_AUPRC`, `learning_rate`, `step`, and elapsed wall-clock seconds.
2. WHILE training in Scenario 2, WHEN an evaluation interval occurs THEN the system SHALL additionally record `macro_f1` and top-1 `accuracy` on the validation set.
3. WHEN a training run completes THEN the system SHALL compute an `exposure_diagnostics` dictionary containing total node visits $N_v$, leaf selections $N_l$, instances selected $N_s$, unique instances selected $N_u$, normalized exposure entropy $H_{\text{norm}} \in [0.0, 1.0]$, count of unexposed classes $C_{\text{unexp}}$, and Worst-10 class recall.
4. WHEN computing exposure entropy $H_{\text{norm}}$ over $K$ classes with zero total selections THEN the system SHALL return $0.0$.
5. WHEN computing exposure entropy $H_{\text{norm}}$ over $K$ classes with positive counts THEN the system SHALL compute $H_{\text{norm}} = -\sum_{c=1}^K p_c \ln(p_c) / \ln(K)$ where $p_c$ is the selection probability of class $c$.

**Independent Test**: Executar testes em `tests/test_phase_b_observability.py` verificando a presença de `step_history` completo e o cálculo determinístico de entropia e Worst-10 recall.

---

### P1: Diagnóstico da Política MCTS e Prevenção de Starvation (B2) (`MCTS-01`, `MCTS-02`) ⭐ MVP

**User Story**: Como engenheiro de aprendizado por reforço, quero inspecionar periodicamente recompensas, valores Q e termos de exploração nos nós da árvore, para garantir que não ocorra starvation e que a propagação de recompensas seja exata.

**Why P1**: Requisito B2 para comprovar que a busca MCTS preserva exploração estritamente positiva ($U > 0$) e propaga valores corretamente até a raiz.

**Acceptance Criteria**:

1. The MCTS agent SHALL provide a diagnostic snapshot method returning per-node raw reward $R_{\text{raw}}$, smoothed reward $R_{\text{smooth}}$, $Q$-value, exploration term $U$, composite UCB score, and visit count $N$.
2. WHILE selecting actions during tree traversal, the MCTS agent SHALL guarantee a strictly positive exploration term $U > 0$ for every unvisited or infrequently visited sibling node whenever parent visits exceed zero and exploration constant $c_{\text{puct}} > 0$.
3. WHEN executing MCTS search on a synthetic micro-hierarchy of 3 classes with uniform rewards THEN the system SHALL visit every leaf at least once within the first 10 steps and propagate visit increments to all direct ancestors up to root `root_id = 0`.

**Independent Test**: Executar testes em `tests/test_phase_b_mcts_diagnostics.py` verificando invariantes de propagação e ausência de starvation em micro-árvores sintéticas.

---

### P1: Checkpoints Duplos e Calibração de Limiares em Validação (B3) (`CKPT-01`, `CKPT-02`) ⭐ MVP

**User Story**: Como experimentalista, quero salvar tanto o checkpoint final (`last`) quanto o de melhor validação (`best_validation`), além de calibrar o limiar de decisão binário na validação antes de avaliar o teste, para garantir avaliações justas e reprodutíveis.

**Why P1**: Requisito B3 para permitir comparação de orçamento fixo sem viés de early stopping e otimizar F1 binário em desbalanceamento.

**Acceptance Criteria**:

1. WHEN a training run completes an update budget of $N$ steps THEN the trainer SHALL save both a `last` checkpoint and a `best_validation` checkpoint containing model state dict, optimizer state dict, tree visit state, and RNG state seeds.
2. WHEN evaluating binary validation probabilities in Scenario 1 THEN the system SHALL determine the optimal decision threshold $\tau_{\text{val}} = \arg\max_{\tau \in [0.01, 0.99]} \text{F1}_{\text{val}}(\tau)$.
3. WHEN evaluating the final test set in Scenario 1 THEN the system SHALL report default F1 ($\tau=0.5$), validation-calibrated F1 ($\tau=\tau_{\text{val}}$), and AUPRC.

**Independent Test**: Executar testes em `tests/test_phase_b_checkpoints.py` verificando a integridade dos artefatos de checkpoint e a exatidão do calibrador de limiar $\tau_{\text{val}}$.

---

### P1: Piloto Diagnóstico Curto e Verificador do Gate B (B4) (`PILOT-01`, `GATE-02`) ⭐ MVP

**User Story**: Como responsável pela infraestrutura experimental, quero um executor de piloto curto e um verificador de integridade do Gate B, para comprovar a viabilidade e estabilidade numérica do desbalanceamento 1:500 antes de iniciar execuções em larga escala.

**Why P1**: Requisito B4 e marco decisório de parada formal (Gate B e Viabilidade 1:500) do roadmap.

**Acceptance Criteria**:

1. WHEN executing the diagnostic pilot runner with budget $\le 500$ steps and 2 seeds THEN the system SHALL execute Scenario 1 (ratios 50, 100, 500) and Scenario 2 (Uniform, Uniform Grouped, Enhanced) and serialize atomic outputs to `results/pilot/` or `results/smoke/`.
2. WHEN `verify_gate_b()` is executed THEN the system SHALL inspect all pilot run records and verify that: (a) all runs have complete `step_history` and `exposure_diagnostics`, (b) no metric or loss value is NaN or infinite, (c) the 1:500 split uses exactly 80 positives and 40,000 negatives without replacement outside minibatches, and (d) both `last` and `best_validation` metrics are recorded.
3. IF any pilot run record contains NaN values, missing keys, or violated split constraints THEN `verify_gate_b()` SHALL return exit code 1 and log a diagnostic error.

**Independent Test**: Executar testes em `tests/test_phase_b_gate.py` validando o executor do piloto em modo smoke e o verificado do Gate B.

---

## Requirement Traceability

| Requirement ID | Description | Component | Verification Method | Status |
| -------------- | ----------- | --------- | ------------------- | ------ |
| OBS-01 | Registro passo a passo de métricas de treino e validação | `src/training/trainer.py` | `tests/test_phase_b_observability.py` | pending |
| OBS-02 | Diagnóstico de exposição amostral, entropia e Worst-10 recall | `src/training/exposure.py` | `tests/test_phase_b_observability.py` | pending |
| MCTS-01 | Snapshot de estado e UCB da política MCTS | `src/mcts/mcts_agent.py` | `tests/test_phase_b_mcts_diagnostics.py` | pending |
| MCTS-02 | Verificação de propagação de recompensa e não-starvation em micro-árvores | `src/mcts/mcts_agent.py` | `tests/test_phase_b_mcts_diagnostics.py` | pending |
| CKPT-01 | Persistência determinística de checkpoints `last` e `best_validation` | `src/training/checkpoints.py` | `tests/test_phase_b_checkpoints.py` | pending |
| CKPT-02 | Calibração de limiar de decisão $\tau_{\text{val}}$ na validação | `src/utils/metrics.py` | `tests/test_phase_b_checkpoints.py` | pending |
| PILOT-01 | Executor do piloto diagnóstico curto (2 seeds, $\le 500$ passos) | `src/training/pilot_runner.py` | `tests/test_phase_b_gate.py` | pending |
| GATE-02 | Verificador do Gate B e viabilidade do desbalanceamento 1:500 | `src/utils/gate_b_verifier.py` | `tests/test_phase_b_gate.py` | pending |
