# F23: P1 — Ablações de Mecanismo e Comparação Estatística das Hierarquias Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, the walk, check_proofs, the Verifier and the screens sub-agent).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Design**: `.specs/features/23-p1-ablacoes-mecanismo-e-hierarquias-pareadas/design.md`  
**Status**: Ready  
**Approved**: 2026-11-11 - user: "sim"  
**Build**: `.venv/bin/python -m compileall -q src tests`  
**Run unit**: `.venv/bin/pytest -v tests/test_p1_ablations_and_hierarchies.py`  

---

## Test Coverage Matrix

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Tree Diagnostics (`src/trees/base_tree.py`) | unit | Topological metrics (depth, nodes, branching) | `tests/test_p1_ablations_and_hierarchies.py` | `.venv/bin/pytest -v tests/test_p1_ablations_and_hierarchies.py` |
| Factorial Statistics (`src/utils/statistics.py`) | unit | Main effects, interaction, per-seed deltas | `tests/test_p1_ablations_and_hierarchies.py` | `.venv/bin/pytest -v tests/test_p1_ablations_and_hierarchies.py` |
| Benchmark Runner (`src/training/benchmark_runner.py`) | integration | Computational costs, paired hierarchy testing, factorial payload | `tests/test_p1_ablations_and_hierarchies.py` | `.venv/bin/pytest -v tests/test_p1_ablations_and_hierarchies.py` |
| Notebook 03 Integration (`notebooks/03_tier1_mcts_benchmarks.ipynb`) | integration | Section 3 execution with paired inference table and causal verdict | `tests/test_tier1_benchmarks_real.py` | `.venv/bin/pytest -v tests/test_tier1_benchmarks_real.py -k "notebook_03"` |

---

## Gate Check Commands

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | After tasks with unit tests only | `.venv/bin/pytest -v tests/test_p1_ablations_and_hierarchies.py` |
| Full | After tasks with integration tests | `.venv/bin/pytest -v tests/test_p1_ablations_and_hierarchies.py tests/test_tier1_benchmarks_real.py` |
| Build | After phase completion or config/entity-only tasks | `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_p1_ablations_and_hierarchies.py` |

---

## Execution Plan

Phases are ordered and run sequentially - each phase completes before the next begins, and tasks within a phase execute in order.

### Phase 1: Diagnósticos Topológicos e Decomposição Fatorial 2x2 (ABL-01, HIER-01)

Implementa `compute_tree_topology_diagnostics` em `src/trees/base_tree.py` e `compute_2x2_factorial_effects` em `src/utils/statistics.py`.

```
T1
```

### Phase 2: Orquestração e Testes Pareados de Hierarquias no Runner (ABL-01, HIER-01)

Atualiza `run_benchmark()` em `src/training/benchmark_runner.py` para calcular e persistir testes estatísticos pareados entre hierarquias no Axis 2, registrar métricas de custo e calcular efeitos fatoriais.

```
T2
```

### Phase 3: Suíte Automatizada de Testes de Regressão P1 (ABL-01, HIER-01)

Implementa `tests/test_p1_ablations_and_hierarchies.py` validando todas as verificações do grid fatorial e pareamento de hierarquias.

```
T3
```

### Phase 4: Integração do Notebook 03 com Tabelas Inferenciais (NOTE-01)

Atualiza a Seção 3 e a Seção de Mecanismos do Notebook 03 com tabelas inferenciais e parecer estatístico formal.

```
T4
```

---

## Task Breakdown

### T1: Implementar diagnósticos topológicos de árvore e decomposição fatorial 2x2

**What**: Implementar `compute_tree_topology_diagnostics()` em `src/trees/base_tree.py` e `compute_2x2_factorial_effects()` em `src/utils/statistics.py`.  
**Where**: `src/trees/base_tree.py`, `src/utils/statistics.py`  
**Depends on**: none  
**Reuses**: `src/utils/statistics.py`  
**Requirement**: ABL-01, HIER-01  

**Done when**:

- [x] `compute_tree_topology_diagnostics()` calcula `max_depth`, `mean_leaf_depth`, `total_nodes`, `num_leaves` e fatores de ramificação
- [x] `compute_2x2_factorial_effects()` calcula efeitos marginais de Q-norm, Dual-Stream e termo de interação com IC e testes pareados
- [x] Gate check passes: `.venv/bin/python -m compileall -q src tests`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `feat(statistics): implement 2x2 factorial decomposition and tree topology diagnostics`  

---

### T2: Orquestrar testes pareados de hierarquia e controles de custo no runner

**What**: Atualizar `run_benchmark()` para: (1) incorporar `computational_costs` em `final_metrics` e CSV; (2) calcular testes inferenciais pareados seed a seed entre `semantic` e `confusion_ward` no Axis 2; (3) extrair diagnósticos topológicos das árvores; e (4) calcular decomposição fatorial quando as 4 variantes forem executadas.  
**Where**: `src/training/benchmark_runner.py`  
**Depends on**: T1  
**Reuses**: `src/trees/base_tree.py`, `src/utils/statistics.py`  
**Requirement**: ABL-01, HIER-01  

**Done when**:

- [x] `computational_costs` (`total_updates`, `gradient_samples`, `mining_forwards`, `elapsed_seconds`) persistidos em JSON e CSV
- [x] Axis 2 calcula e exporta testes pareados seed a seed ($\Delta_{\text{AUPRC}}$, $\Delta_{\text{F1}}$, Cohen's d, t-test, Wilcoxon) em `tier1_axis2_tree_ablation.json`
- [x] Diagnósticos topológicos das árvores persistidos no payload de saída
- [x] Grid fatorial $2 \times 2$ calcula e exporta efeitos principais e interação
- [x] Gate check passes: `.venv/bin/python -m compileall -q src tests`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `feat(benchmark): add paired hierarchy inferential testing and factorial 2x2 orchestration`  

---

### T3: Implementar suíte de testes de regressão P1

**What**: Criar `tests/test_p1_ablations_and_hierarchies.py` testando a decomposição fatorial $2 \times 2$, custos computacionais, métricas topológicas de árvore e testes pareados entre hierarquias no Axis 2.  
**Where**: `tests/test_p1_ablations_and_hierarchies.py`  
**Depends on**: T1, T2  
**Reuses**: existing test harness  
**Requirement**: ABL-01, HIER-01  

**Done when**:

- [ ] Teste valida cálculo topológico de árvore (profundidade, nós, ramificação)
- [ ] Teste valida decomposição fatorial $2 \times 2$ (efeitos principais e interação)
- [ ] Teste valida persistência e presença de `computational_costs` no CSV e JSON
- [ ] Teste valida testes inferenciais pareados Semantic vs Ward no Axis 2
- [ ] Gate check passes: `.venv/bin/pytest -v tests/test_p1_ablations_and_hierarchies.py`  

**Tests**: integration  
**Gate**: full  
**Commit**: `test(ablations): automated test suite for factorial mechanisms and paired hierarchy testing`  

---

### T4: Atualizar Seção 3 do Notebook 03 com tabelas inferenciais e parecer

**What**: Atualizar a Seção 3 e Seção de Ablação de `notebooks/03_tier1_mcts_benchmarks.ipynb` exibindo a tabela inferencial pareada Semantic vs Ward (com $\Delta$, IC 95%, Cohen's d e p-valores) e emitindo o parecer formal sobre a hipótese nula de equivalência.  
**Where**: `notebooks/03_tier1_mcts_benchmarks.ipynb`  
**Depends on**: T3  
**Reuses**: `src/training/benchmark_runner.py`  
**Requirement**: NOTE-01  

**Done when**:

- [ ] Notebook 03 Seção 3 exibe tabela de testes pareados Semantic vs Ward com $\Delta$, Cohen's d e p-valores
- [ ] Notebook 03 exibe parecer estatístico formal indicando se a diferença é significante ou se não há evidência para declarar superioridade
- [ ] Testes de execução do Notebook 03 em SMOKE passam com exit code 0
- [ ] Gate check passes: `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_p1_ablations_and_hierarchies.py tests/test_tier1_benchmarks_real.py -k "notebook_03"`  

**Tests**: integration  
**Gate**: build  
**Commit**: `feat(notebooks): update notebook 03 axis 2 with paired inferential tables and statistical verdict`  

---

## Checks

**C1** - `compute_tree_topology_diagnostics()` computes max depth, mean leaf depth, node counts, and branching factors (HIER-01, AC 3)  
Proof: run unit | tests/test_p1_ablations_and_hierarchies.py::test_tree_topology_diagnostics PASSED  

**C2** - `compute_2x2_factorial_effects()` calculates main effects of Q-norm and Dual-Stream and interaction term (ABL-01, AC 3)  
Proof: run unit | tests/test_p1_ablations_and_hierarchies.py::test_factorial_2x2_effects_calculation PASSED  

**C3** - `run_benchmark()` records computational cost metrics in `final_metrics` and CSV output (ABL-01, AC 2)  
Proof: run unit | tests/test_p1_ablations_and_hierarchies.py::test_benchmark_records_computational_costs PASSED  

**C4** - Axis 2 computes seed-by-seed paired statistical tests between Semantic and Ward hierarchies (HIER-01, AC 1, AC 2)  
Proof: run unit | tests/test_p1_ablations_and_hierarchies.py::test_axis2_paired_hierarchy_inferential_tests PASSED  

**C5** - Axis 2 records tree topological diagnostics in `tier1_axis2_tree_ablation.json` (HIER-01, AC 3, AC 4)  
Proof: run unit | tests/test_p1_ablations_and_hierarchies.py::test_axis2_persists_tree_topology_diagnostics PASSED  

**C6** - Factorial 2x2 execution computes and persists factorial decomposition payload (ABL-01, AC 1, AC 4)  
Proof: run unit | tests/test_p1_ablations_and_hierarchies.py::test_factorial_2x2_execution_and_persistence PASSED  

---

## Handoff

- **Dispatched:** none
