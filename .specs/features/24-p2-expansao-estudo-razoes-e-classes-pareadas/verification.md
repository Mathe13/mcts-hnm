# 24-p2-expansao-estudo-razoes-e-classes-pareadas verification

**Verdict**: PASS
**Proofs**: `proofs.md` at 2eae17b
**Verifier**: independent sub-agent (author != verifier)
**Date**: 2026-11-11

## Scope and Acceptance Criteria Audit

- **EXP-01 (P2: Avaliação Multi-Razão e Interação Método x Razão)**:
  - `run_benchmark()` no Scenario 1 aceita tupla de razões `imbalance_ratios=("1:10", "1:50", "1:100")` e avalia cada razão sob sementes pareadas e estratégias especificadas. (AC 1: **Holds**)
  - Subsampling do pool positivo via `build_scenario1_split()` mapeia corretamente cada razão canônica para sua fração positiva proporcional (`1:10` -> `0.10`, `1:50` -> `0.02`, `1:100` -> `0.01`). (AC 2: **Holds**)
  - `compute_method_by_ratio_interaction()` em `src/utils/statistics.py` calcula diferenças pareadas e efeitos de interação entre razões ($\Delta_{\text{method}}(r_2) - \Delta_{\text{method}}(r_1)$) com intervalo de confiança de 95%, Cohen's d e testes de hipótese de Student. (AC 3: **Holds**)
  - Saídas JSON (`tier1_axis1_samplers.json`) e CSV registram `imbalance_ratio` para cada execução individual e incluem o dicionário `ratio_interactions` nos resultados agregados quando múltiplas razões são avaliadas. (AC 4: **Holds**)

- **EXP-02 (P2: Generalização para Múltiplas Classes-Alvo)**:
  - `run_benchmark()` aceita `target_classes: tuple[int, ...] | None` (com fallback retrocompatível para `(target_class_scenario1,)` se omitido) no Scenario 1. (AC 1: **Holds**)
  - Para cada classe-alvo em `target_classes`, executa com cronogramas de sementes idênticos e grava `target_class` em cada `BenchmarkRunResult`, linhas CSV e JSON. (AC 2: **Holds**)
  - O benchmark runner agrega métricas inter-classes (`aggregated_by_class`), reportando média e dispersão (std) por classe e estratégia. (AC 3: **Holds**)
  - Saída JSON persiste `target_classes` e `aggregated_by_class` nos metadados e provenientes. (AC 4: **Holds**)

- **EXP-03 (P2: Alinhamento de Notebooks e Artefatos a Jusante)**:
  - Notebook 03 Seção 2 parametriza `IMBALANCE_RATIOS_AXIS1` e `TARGET_CLASSES_AXIS1`, exibindo resumo comparativo agrupado e tabela de efeitos de interação método $\times$ razão quando múltiplas razões/classes são avaliadas. (AC 1: **Holds**)
  - Exportadores e carregadores de dados em `src/paper/loaders.py` e `src/utils/paper_artifacts.py` suportam esquemas com múltiplos `target_classes` e `ratio_interactions` sem erros de validação de esquema. (AC 2: **Holds**)
  - Notebook 03 executa ponta-a-ponta em modo SMOKE com exit code 0 (`tests/test_tier1_benchmarks_real.py::test_notebook_03_smoke_execution`). (AC 3: **Holds**)

## Checks

| Check | Test | Holds |
| --- | --- | --- |
| C1 | `tests/test_p2_study_expansion.py::test_multi_ratio_execution_and_positive_fractions` | yes |
| C2 | `tests/test_p2_study_expansion.py::test_method_by_ratio_interaction_calculation` | yes |
| C3 | `tests/test_p2_study_expansion.py::test_multi_ratio_persistence_and_interactions_payload` | yes |
| C4 | `tests/test_p2_study_expansion.py::test_multi_target_class_execution` | yes |
| C5 | `tests/test_p2_study_expansion.py::test_target_class_persistence_and_aggregation` | yes |
| C6 | `tests/test_p2_study_expansion.py::test_notebook_03_multi_ratio_and_multi_class_integration` | yes |
| Build | `.venv/bin/python -m compileall -q src tests` | yes |

## Code Quality & Invariants Audit

- **Orquestração Multi-Razão e Multi-Classe (`src/training/benchmark_runner.py`)**:
  - `BenchmarkRunResult` e `csv_rows` incorporam o campo `target_class` mantendo ordenação padronizada de colunas (`scenario`, `target_class`, `strategy`, `tree_type`, `seed`, `imbalance_ratio`, ...).
  - Iteração externa estruturada em `classes_to_run` $\times$ `ratios_to_run` $\times$ `tree_types` $\times$ `strategies` $\times$ `seeds`, preservando o pareamento exato de sementes e splits canônicos com `build_scenario1_split()`.
  - Avaliação de rótulos de teste e validação binários recalibrada para cada classe-alvo (`(test_lbl_np == tc).astype(np.int64)` e `(val_lbl_np == tc).astype(np.int64)`).
  - Cálculo robusto de `ratio_interactions` entre todos os pares de razões $(r_1, r_2)$ sobre a interseção pareada de sementes comuns (`len(common_sd) >= 2`), consumindo a função estatística `compute_method_by_ratio_interaction()`.
  - Agregação inter-classes estruturada em `aggregated_by_class` computando média e desvio padrão (`ddof=1`) para `auprc` e `f1`.
  - Retrocompatibilidade total preservada para cenários de razão única e classe única.
- **Robustez de Carga em Artefatos de Publicação (`src/utils/paper_artifacts.py`)**:
  - `load_real_benchmark_results()` estendido para carregar e repassar `ratio_interactions`, `target_classes` e `aggregated_by_class` sem violar nenhuma asserção de proveniência ou esquema dos artefatos.
- **Parametrização e Transparência no Notebook 03 (`notebooks/03_tier1_mcts_benchmarks.ipynb`)**:
  - Seção 2 documenta os pontos de extensão `IMBALANCE_RATIOS_AXIS1` e `TARGET_CLASSES_AXIS1`.
  - Exibe quebras detalhadas por estratégia, razão e classe quando grades estendidas são acionadas.
  - Formata e imprime a tabela de interação método $\times$ razão com médias, ICs 95%, Cohen's d e p-valores pareados.
  - Execução validada em modo SMOKE com exit code 0.
- **Suíte de Testes Automatizada (`tests/test_p2_study_expansion.py`)**:
  - Cobre exaustivamente C1 a C6, testando proporções de subsampling positivo, fórmulas matemáticas de interação, esquema e integridade de arquivos persistidos CSV/JSON, e agregação inter-classes.

## Findings

None - all acceptance criteria satisfied and regression test suite passes with zero defects.
