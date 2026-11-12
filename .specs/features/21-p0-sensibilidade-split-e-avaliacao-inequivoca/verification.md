# 21-p0-sensibilidade-split-e-avaliacao-inequivoca verification

**Verdict**: PASS
**Proofs**: `proofs.md` at 7ecb09b
**Verifier**: independent sub-agent (author != verifier)
**Date**: 2026-11-11

## Scope and Acceptance Criteria Audit

- **SENS-01 (P1 - Subsampling Equivalente no Runner de Sensibilidade)**:
  - `run_sensitivity_ablation()` executa particionamento via `build_scenario1_split` por semente com `imbalance_ratio` especificado. (AC 1)
  - `imbalance_ratio` possui valor default `"1:50"` e `num_steps` possui default `1000`, permitindo override explícito. (AC 2)
  - Os índices positivos e negativos por semente são estritamente idênticos aos gerados por `build_scenario1_split` sob a mesma semente. (AC 3)
  - Metadados `imbalance_ratio`, `num_positives_train` e `num_negatives_train` são registrados no dicionário e no JSON exportado. (AC 4)
- **EVAL-01 / EVAL-02 (P1 - Política Inequívoca de Avaliação nos Resultados)**:
  - Métricas avaliadas tanto para `threshold_0.5` quanto para `threshold_calibrated_val` calibrado na validação. (AC 1)
  - Métricas calculadas e reportadas de forma independente para ambos os candidatos a checkpoint: `checkpoint_last` e `checkpoint_best_val`. (AC 2)
  - JSON e CSV exportados contêm chaves explícitas dos 4 quadrantes (`test_f1_last_thresh05`, `test_f1_last_calibrated`, `test_f1_best_val_thresh05`, `test_f1_best_val_calibrated`, com precisão, recall, AUPRC e limiares). (AC 3)
  - Ausência de dados de validação ou razão de desbalanceamento inválida levanta `ValueError`, impedindo execução descalibrada silenciosa. (AC 4)
- **EQUIV-01 (P1 - Teste Pareado de Equivalência Estrita de Índices)**:
  - Índices positivos e negativos de treino entre `run_benchmark()` e `run_sensitivity_ablation()` são estritamente idênticos (`np.array_equal`) sob as mesmas sementes. (AC 1, AC 2)
  - Validação automatizada realizada através de múltiplas sementes (42, 43, 44, 45, 46). (AC 3)

## Checks

| Check | Test | Holds |
| --- | --- | --- |
| C1 | `tests/test_sensitivity_runner_equivalence.py` `test_sensitivity_runner_uses_scenario1_split` | yes |
| C2 | `tests/test_sensitivity_runner_equivalence.py` `test_sensitivity_indices_match_scenario1_split` | yes |
| C3 | `tests/test_sensitivity_runner_equivalence.py` `test_sensitivity_output_records_pool_metadata` | yes |
| C4 | `tests/test_sensitivity_runner_equivalence.py` `test_sensitivity_exports_4_quadrant_metrics` | yes |
| C5 | `tests/test_sensitivity_runner_equivalence.py` `test_sensitivity_requires_validation_data` | yes |
| C6 | `tests/test_sensitivity_runner_equivalence.py` `test_benchmark_and_sensitivity_index_equivalence` | yes |
| Build | `.venv/bin/python -m compileall -q src tests` | yes |

## Code Quality & Invariants Audit

- **Particionamento e Subsampling Equivalente (`src/training/benchmark_runner.py`)**:
  - `run_sensitivity_ablation()` adota estritamente `build_scenario1_split` por semente com `pos_frac = IMBALANCE_RATIOS[imbalance_ratio]`, eliminando a discrepância histórica em que 450 positivos eram retidos no pool de treino.
  - No regime 1:50 sobre CIFAR-100 real (450 positivos no split de treino pós-validação), o número de positivos selecionados passa a ser de ~9 instâncias ($450 \times 0.02 = 9$), perfeitamente alinhado com o Axis 1 (`run_benchmark`).
  - Parâmetros padrão alinhados: `imbalance_ratio="1:50"` e `num_steps=1000`.
- **Política Inequívoca de Avaliação (`src/training/trainer.py` e `src/training/benchmark_runner.py`)**:
  - `evaluate_scenario1_metrics()` calcula métricas completas com limiar padrão 0.5 (`f1_thresh05`, `precision_thresh05`, `recall_thresh05`) e com o limiar calibrado (`calibrated_f1`, `calibrated_precision`, `calibrated_recall`, `calibrated_threshold`), além de AUPRC e FPR@95.
  - `train_scenario1()` preserva de forma independente os pesos e avaliações de `last_metrics` e `best_validation_metrics`.
  - `run_sensitivity_ablation()` consolida e exporta explicitamente os 4 quadrantes para cada execução:
    - Quadrante 1: `(last, thresh05)` -> `test_f1_last_thresh05`, `test_precision_last_thresh05`, `test_recall_last_thresh05`
    - Quadrante 2: `(last, calibrated)` -> `test_f1_last_calibrated`, `test_precision_last_calibrated`, `test_recall_last_calibrated`, `best_tau_val_last`
    - Quadrante 3: `(best_val, thresh05)` -> `test_f1_best_val_thresh05`, `test_precision_best_val_thresh05`, `test_recall_best_val_thresh05`
    - Quadrante 4: `(best_val, calibrated)` -> `test_f1_best_val_calibrated`, `test_precision_best_val_calibrated`, `test_recall_best_val_calibrated`, `best_tau_val`
    - Métrica de ranking primária invariante ao limiar: `test_auprc`
  - Exportação serializada atômica em `.json` e `.csv` com metadados de proveniência (`.meta.json`).
- **Equivalência Estrita e Cobertura de Testes (`tests/test_sensitivity_runner_equivalence.py`)**:
  - Teste `test_benchmark_and_sensitivity_index_equivalence` assere rigorosamente igualdade bit-a-bit (`np.array_equal`) de índices positivos e negativos entre os geradores de split de `run_benchmark` e `run_sensitivity_ablation` em 5 sementes consecutivas.
  - Testes unitários cobrem o comportamento sob ausência de validação (`ValueError`), razões inválidas e integridade de colunas do CSV e JSON.
  - Não-regressão verificada contra `test_tier1_benchmarks_real.py` e `test_phase_a_runner_equivalence.py`.

## Findings

None - Todos os testes unitários e de integração passam sem ressalvas, os 6 checks C1-C6 e o Build estão 100% satisfeitos, e a auditoria de código confirma a eliminação definitiva do desvio metodológico P0.

Gaps: None
