# 22-p0-scenario2-cobertura-e-politicas-amostragem verification

**Verdict**: PASS
**Proofs**: `proofs.md` at 71f07ae
**Verifier**: independent sub-agent (author != verifier)
**Date**: 2026-11-11

## Scope and Acceptance Criteria Audit

- **S2-01 (P1 - Orquestração e Comparação das Variantes de Amostragem no Scenario 2)**:
  - `run_benchmark()` no Scenario 2 treina e avalia qualquer subconjunto das estratégias suportadas sob idênticas sementes e orçamentos computacionais. (AC 1)
  - `uniform_grouped` é suportado como controle de homogeneidade ao lado das variantes MCTS, registrando métricas idênticas de avaliação. (AC 2)
  - Resultados são exportados com proveniência atômica para `tier1_scenario2_curriculum.json` e `tier1_scenario2_curriculum.csv` acompanhados por `.meta.json`. (AC 3)
  - Estratégias não suportadas no Scenario 2 levantam `ValueError` descritivo listando as opções válidas. (AC 4)
- **S2-02 (P1 - Variante de Política de Cobertura Balanceada `mcts_coverage_balanced`)**:
  - `mcts_coverage_balanced` compõe minibatches com 25% de amostras da classe focal minerada pelo MCTS e 75% uniformes (16 focais + 48 uniformes em batch 64). (AC 1)
  - Mantém Min-Max Q-normalização (`q_norm=True`) e travessia semântica na árvore idêntica a `mcts_enhanced`. (AC 2)
  - A estratégia histórica `mcts_enhanced` permanece estritamente inalterada com proporção 50/50 (32 MCTS + 32 uniformes). (AC 3)
- **S2-03 (P1 - Diagnóstico Quantitativo e Análise Estatística Pareada)**:
  - O runner calcula e registra entropia normalizada de exposição $H_{\text{norm}}$, cobertura acumulada $C_{\text{cov}}$, contagem de classes não visitadas $N_{\text{unvisited}}$ e Worst-10 recall para cada execução. (AC 1)
  - O runner computa diferenças estatísticas pareadas ($\Delta_{\text{acc}}$, $\Delta_{\text{macro\_f1}}$, $\Delta_{\text{worst10}}$, Cohen's d e teste t pareado) contra a estratégia `uniform` através de sementes compartilhadas. (AC 2)
  - O sistema calcula a correlação de postos de Spearman e correlação de Pearson entre as contagens de amostragem por classe e os recalls de teste correspondentes. (AC 3)
  - Os diagnósticos são persistidos atomicamente em `tier1_scenario2_diagnostics.json` e colunas correspondentes são consolidadas no CSV de resultados. (AC 4)
- **S2-04 (P1 - Atualização do Notebook 03 com Grid Completo e Visualizações)**:
  - A Seção 4 do Notebook 03 executa as 4 estratégias nucleares no Scenario 2: `uniform`, `uniform_grouped`, `mcts_enhanced` e `mcts_coverage_balanced`. (AC 1)
  - Exibe tabela comparativa consolidada contendo Top-1 Accuracy, Macro-F1, Worst-10 Recall, Entropia $H_{\text{norm}}$, Cobertura $C_{\text{cov}}$, Classes não visitadas e Correlação Exposição-Recall. (AC 2)
  - Emite parecer causal automatizado distinguindo o impacto da homogeneidade de batch (saturação de rótulo) versus a política de busca do MCTS, avaliando o ganho da variante balanceada. (AC 3)

## Checks

| Check | Test | Holds |
| --- | --- | --- |
| C1 | `tests/test_scenario2_coverage_and_variants.py` `test_mcts_coverage_balanced_strategy_registration` | yes |
| C2 | `tests/test_scenario2_coverage_and_variants.py` `test_mcts_coverage_balanced_batch_composition` | yes |
| C3 | `tests/test_scenario2_coverage_and_variants.py` `test_scenario2_benchmark_runs_all_variants` | yes |
| C4 | `tests/test_scenario2_coverage_and_variants.py` `test_scenario2_records_quantitative_diagnostics` | yes |
| C5 | `tests/test_scenario2_coverage_and_variants.py` `test_scenario2_paired_differences_against_uniform` | yes |
| C6 | `tests/test_scenario2_coverage_and_variants.py` `test_scenario2_rejects_unsupported_strategy` | yes |
| Build | `.venv/bin/python -m compileall -q src tests` | yes |

## Code Quality & Invariants Audit

- **Resolução de Hiperparâmetros e Particionamento de Batch (`src/training/trainer.py`)**:
  - `resolve_mcts_hyperparameters()` mapeia `"mcts_coverage_balanced"` para `q_norm=True` e `beta=0.75` (fração de âncora uniforme), resultando em 25% de amostras focais da folha MCTS ($K_{\text{mcts}} = \lfloor (1 - 0.75) \times 64 \rfloor = 16$) e 75% uniformes ($K_{\text{uni}} = 48$).
  - O baseline histórico `mcts_enhanced` permanece estritamente preservado com `q_norm=True` e `beta=0.5` ($32/32$).
  - Ambas as variantes MCTS preservam integralmente a travessia UCB hierárquica e a atualização de visitas e recompensas por backpropagation na árvore `FastHierarchyTree`.
- **Orquestração e Telemetria no Runner (`src/training/benchmark_runner.py`)**:
  - `run_benchmark()` valida exaustivamente as estratégias fornecidas contra `SUPPORTED_STRATEGIES` e mapeia `MCTS_STRATEGIES` de forma coesa tanto para o Scenario 1 quanto para o Scenario 2.
  - Para cada execução do Scenario 2, o runner computa as predições de teste de forma vetorizada em batches sem vazamento de gradiente (`torch.no_grad()`), calculando o recall por classe para todas as 100 classes.
  - Extrai telemetria de exposição via `ExposureTracker` (`exposure_entropy`, `cumulative_class_coverage`, `unexposed_classes_count`) e correlaciona as frequências de exposição com os recalls por classe via `compute_exposure_recall_correlation()`.
  - Persiste atomicamente `tier1_scenario2_diagnostics.json` com proveniência `.meta.json` e popula colunas diagnósticas no arquivo CSV principal.
- **Análise Estatística Pareada (`src/training/benchmark_runner.py`)**:
  - As execuções são agrupadas por semente compartilhada contra `uniform`, calculando deltas pareados de acurácia, Macro-F1 e Worst-10 recall.
  - Quando $\ge 2$ sementes estão disponíveis, computa teste t pareado, p-valor e tamanho de efeito (Cohen's d).
- **Notebook 03 e Parecer Causal (`notebooks/03_tier1_mcts_benchmarks.ipynb`)**:
  - A Seção 4 parametriza e executa as 4 estratégias nucleares (`uniform`, `uniform_grouped`, `mcts_enhanced`, `mcts_coverage_balanced`).
  - O parecer causal analisa se `uniform_grouped` reproduz a perda de acurácia de `mcts_enhanced`, isolando conclusivamente o efeito da homogeneidade de batch contra anomalias na política da árvore.
- **Não-Regressão e Cobertura de Testes (`tests/test_scenario2_coverage_and_variants.py`)**:
  - Suíte com 6 testes automatizados cobrindo registro de estratégias, composição de batch bit-a-bit, execução completa no runner sob mock controlado, extração de diagnósticos quantitativos, análise pareada e rejeição de estratégias inválidas.
  - Tempo de execução total dos testes unitários inferior a 10 segundos, cumprindo o SLA estrito da suite de CI.

## Findings

None - Todos os 6 checks (C1..C6) e o Build foram comprovados com sucesso no commit 71f07ae. Todos os critérios de aceitação das estórias S2-01 a S2-04 estão plenamente atendidos.

Gaps: None
