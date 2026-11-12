# 23-p1-ablacoes-mecanismo-e-hierarquias-pareadas verification

**Verdict**: PASS
**Proofs**: `proofs.md` at 2fdf64f
**Verifier**: independent sub-agent (author != verifier)
**Date**: 2026-11-11

## Scope and Acceptance Criteria Audit

- **ABL-01 (P1: Grid Fatorial $2 \times 2$ de Mecanismos e Controles de Custo)**:
  - `run_benchmark()` treina e avalia as 4 variantes canônicas sob idênticas sementes e orçamentos: `mcts_strict` ($S_{00}$), `mcts_qnorm_only` ($S_{10}$), `mcts_dualstream_only` ($S_{01}$), `mcts_enhanced` ($S_{11}$). (AC 1: **Holds**)
  - Rastreamento explícito de custos computacionais (`total_updates`, `gradient_samples`, `mining_forwards`, `elapsed_seconds`) em `final_metrics`, JSON e CSV. (AC 2: **Holds**)
  - Decomposição fatorial com cálculo dos efeitos marginais $\text{ME}_Q$, $\text{ME}_{\text{DS}}$ e termo de interação $\Delta_{\text{interaction}} = (S_{11} - S_{10}) - (S_{01} - S_{00})$ implementada em `compute_2x2_factorial_effects()`. (AC 3: **Holds**)
  - Exportação e persistência estruturada dos resultados com metadados de proveniência `.meta.json`. (AC 4: **Holds**)

- **HIER-01 (P1: Comparação Estatística Pareada de Hierarquias no Axis 2)**:
  - Execução pareada seed a seed entre Árvore Semântica e Ward Confusion sob as mesmas sementes e splits. (AC 1: **Holds**)
  - Computação inferencial completa ($\Delta_{\text{primary}}$, $\Delta_{\text{F1}}$, IC 95%, Cohen's d, teste t pareado e Wilcoxon). (AC 2: **Holds**)
  - Diagnósticos topológicos da árvore extraídos via `compute_tree_topology_diagnostics()` (`max_depth`, `mean_leaf_depth`, `min_leaf_depth`, `total_nodes`, `num_leaves`, `num_internal_nodes`, `branching_factor_mean/max/min`). (AC 3: **Holds**)
  - Persistência estruturada em `tier1_axis2_tree_ablation.json` e incorporação no payload geral. (AC 4: **Holds**)

- **NOTE-01 (P1: Atualização do Notebook 03 com Tabelas Inferenciais de Ablação e Hierarquias)**:
  - Seção 3 do Notebook 03 exibe tabela comparativa pareada entre Semantic e Ward Confusion com deltas, Cohen's d e p-valores. (AC 1: **Holds**)
  - Apresentação da tabela fatorial $2 \times 2$ de mecanismos no Notebook 03 com efeitos principais e interação. (AC 2: **Holds**)
  - Execução ponta-a-ponta do Notebook 03 em modo SMOKE com exit code 0. (AC 3: **Holds**)

## Checks

| Check | Test | Holds |
| --- | --- | --- |
| C1 | `tests/test_p1_ablations_and_hierarchies.py` `test_tree_topology_diagnostics` | yes |
| C2 | `tests/test_p1_ablations_and_hierarchies.py` `test_factorial_2x2_effects_calculation` | yes |
| C3 | `tests/test_p1_ablations_and_hierarchies.py` `test_benchmark_records_computational_costs` | yes |
| C4 | `tests/test_p1_ablations_and_hierarchies.py` `test_axis2_paired_hierarchy_inferential_tests` | yes |
| C5 | `tests/test_p1_ablations_and_hierarchies.py` `test_axis2_persists_tree_topology_diagnostics` | yes |
| C6 | `tests/test_p1_ablations_and_hierarchies.py` `test_factorial_2x2_execution_and_persistence` | yes |
| Build | `.venv/bin/python -m compileall -q src tests` | yes |

## Code Quality & Invariants Audit

- **Diagnósticos Topológicos (`src/trees/base_tree.py`)**:
  - `compute_tree_topology_diagnostics()` calcula corretamente métricas de profundidade e ramificação, com proteção contra divisão por zero para árvores sem nós internos.
- **Decomposição Fatorial $2 \times 2$ (`src/utils/statistics.py`)**:
  - `compute_2x2_factorial_effects()` implementa as fórmulas com precisão matemática, gerando as chaves canônicas: `main_effect_qnorm`, `main_effect_dualstream`, `interaction_effect` e `variant_means`. Trata vetores de tamanho incompatível, não-finitos e variância zero.
- **Integração no Runner (`src/training/benchmark_runner.py`)**:
  - Rastreia e injeta custos computacionais (`total_updates`, `gradient_samples`, `mining_forwards`, `elapsed_seconds`) em `final_metrics` e colunas do CSV.
  - Orquestra pareamento seed a seed para hierarquias (`paired_hierarchy_tests[strat]`), diagnósticos topológicos em `tree_diagnostics` e fatorial $2 \times 2$ (`factorial_2x2_effects[tt]`).
  - Fornece síntese formal do parecer estatístico (`verdict`) nas comparações pareadas de hierarquia.
- **Integração no Notebook 03 (`notebooks/03_tier1_mcts_benchmarks.ipynb`)**:
  - Na Seção 2 (Cell 5), o acesso aos efeitos primários consome rigorosamente as chaves `main_effect_dualstream` e `interaction_effect`.
  - Na Seção 3 (Cell 7), extrai adequadamente `primary_stats` e `f1_stats` de `pht["mcts_enhanced"]`, exibe a tabela comparativa pareada e renderiza o parecer estatístico formal.
  - O notebook executa de ponta a ponta em modo SMOKE com zero erros.
- **Não-Regressão e Cobertura de Testes (`tests/test_p1_ablations_and_hierarchies.py`)**:
  - A suíte de testes de regressão valida todas as chaves consumidas pelo Notebook 03, bem como o cálculo numérico das fórmulas fatoriais e testes pareados.

## Findings

| Finding | Evidence | Fix | Closed |
| --- | --- | --- | --- |
| Notebook 03 Cell 5 raises KeyError on factorial effects | Execução de Cell 5 com `EXECUTION_MODE=smoke` falhou com `KeyError: 'main_effect_dual_stream'` | Alinhar as chaves para `main_effect_dualstream` e `interaction_effect` | 2fdf64f |
| Notebook 03 Cell 7 renders empty paired table and missing verdict | `res_axis2["paired_hierarchy_tests"]` é indexado por estratégia (`mcts_enhanced`), mas Cell 7 iterava por nomes de métrica e esperava `pht.get('verdict')` | Acessar `pht["mcts_enhanced"]`, extrair `primary_stats` e `f1_stats`, e emitir o parecer formal sobre $H_0$ | 2fdf64f |
| Test suite does not execute Notebook 03 cells to detect runtime errors | `test_notebook_03_axis2_and_factorial_integration` fazia apenas `assert "..." in full_text` | Validar as chaves consumidas pelo notebook no teste de integração | 2fdf64f |
