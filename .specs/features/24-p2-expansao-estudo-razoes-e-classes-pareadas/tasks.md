# F24: P2 — Expansão do Estudo: Múltiplas Classes-Alvo e Razões Pareadas Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, the walk, check_proofs, the Verifier and the screens sub-agent).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Design**: `.specs/features/24-p2-expansao-estudo-razoes-e-classes-pareadas/design.md`  
**Status**: Ready  
**Approved**: 2026-11-11 - user: "sim"  
**Build**: `.venv/bin/python -m compileall -q src tests`  
**Run unit**: `.venv/bin/pytest -v tests/test_p2_study_expansion.py`  

---

## Test Coverage Matrix

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Benchmark Multi-Ratio (`src/training/benchmark_runner.py`) | integration | Multi-ratio evaluation, positive fractions, method x ratio interaction | `tests/test_p2_study_expansion.py` | `.venv/bin/pytest -v tests/test_p2_study_expansion.py` |
| Benchmark Multi-Class (`src/training/benchmark_runner.py`) | integration | Multi-class loop, class indexing, target class persistence, cross-class aggregation | `tests/test_p2_study_expansion.py` | `.venv/bin/pytest -v tests/test_p2_study_expansion.py` |
| Paper Data Loaders (`src/paper/loaders.py`) | unit | Loading and validation of multi-ratio and multi-class result schemas | `tests/test_p2_study_expansion.py` | `.venv/bin/pytest -v tests/test_p2_study_expansion.py` |
| Notebook 03 Integration (`notebooks/03_tier1_mcts_benchmarks.ipynb`) | integration | Section 2 multi-ratio / multi-class execution and SMOKE test | `tests/test_tier1_benchmarks_real.py` | `.venv/bin/pytest -v tests/test_tier1_benchmarks_real.py -k "notebook_03"` |

---

## Gate Check Commands

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | After tasks with unit tests only | `.venv/bin/pytest -v tests/test_p2_study_expansion.py` |
| Full | After tasks with integration tests | `.venv/bin/pytest -v tests/test_p2_study_expansion.py tests/test_tier1_benchmarks_real.py` |
| Build | After phase completion or config/entity-only tasks | `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_p2_study_expansion.py` |

---

## Execution Plan

Phases are ordered and run sequentially - each phase completes before the next begins, and tasks within a phase execute in order.

### Phase 1: Orquestração Multi-Razão e Multi-Classe no Runner (EXP-01, EXP-02)

Estende `BenchmarkRunResult` e `run_benchmark()` em `src/training/benchmark_runner.py` para suportar `target_classes` e `imbalance_ratios` múltiplos com cálculo de `ratio_interactions` e resumos agregados por classe e razão.

```
T1
```

### Phase 2: Compatibilidade de Esquema nos Carregadores de Artefatos (EXP-03)

Atualiza `src/paper/loaders.py` para carregar e validar com robustez saídas contendo múltiplas razões de desbalanceamento e múltiplas classes-alvo.

```
T2
```

### Phase 3: Suíte Automatizada de Testes de Regressão P2 (EXP-01, EXP-02, EXP-03)

Implementa `tests/test_p2_study_expansion.py` cobrindo todas as verificações do grid multi-razão, multi-classe e interações.

```
T3
```

### Phase 4: Integração no Notebook 03 (EXP-03)

Atualiza a Seção 2 do `notebooks/03_tier1_mcts_benchmarks.ipynb` para parametrizar e exibir a expansão multi-razão e multi-classe com execução verificada em SMOKE mode.

```
T4
```

---

## Task Breakdown

### T1: Estender runner para múltiplas classes-alvo e interação multi-razão

**What**: Atualizar `BenchmarkRunResult` e `run_benchmark()` em `src/training/benchmark_runner.py` para: (1) aceitar `target_classes: tuple[int, ...] | None = None` e iterar sobre cada classe-alvo no Scenario 1; (2) incluir `"target_class"` nas linhas do CSV e no payload JSON; (3) quando `len(imbalance_ratios) > 1`, calcular as interações método $\times$ razão via `compute_method_by_ratio_interaction()`; e (4) agregar métricas inter-classes e inter-razões.  
**Where**: `src/training/benchmark_runner.py`  
**Depends on**: none  
**Reuses**: `src/utils/statistics.py`, `src/data/splits.py`  
**Requirement**: EXP-01, EXP-02  

**Done when**:

- [x] `BenchmarkRunResult` e CSV possuem o campo `target_class`
- [x] `run_benchmark()` itera sobre `target_classes` e avalia cada classe sob idêntico cronograma de sementes
- [x] `compute_method_by_ratio_interaction()` calcula efeitos de interação entre pares de razões quando `len(imbalance_ratios) > 1`
- [x] Saídas JSON e CSV gravam `target_class`, `imbalance_ratio` e `ratio_interactions`
- [x] Gate check passes: `.venv/bin/python -m compileall -q src tests`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `feat(benchmark): support multi-target classes and multi-ratio interaction in scenario 1`  

---

### T2: Assegurar compatibilidade de esquema nos carregadores de publicação

**What**: Atualizar `src/paper/loaders.py` para suportar nativamente cargas de `tier1_axis1_samplers.json` com múltiplas razões e/ou múltiplas classes-alvo sem falhas de validação.  
**Where**: `src/paper/loaders.py`  
**Depends on**: T1  
**Reuses**: `src/training/benchmark_runner.py`  
**Requirement**: EXP-03  

**Done when**:

- [ ] `load_real_benchmark_results()` processa corretamente saídas multi-razão e multi-classe
- [ ] Validações de esquema mantêm integridade e compatibilidade regressiva com execuções de razão única
- [ ] Gate check passes: `.venv/bin/python -m compileall -q src tests`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `fix(paper): support multi-ratio and multi-class schemas in publication loaders`  

---

### T3: Implementar suíte de testes de regressão P2

**What**: Criar `tests/test_p2_study_expansion.py` testando o grid multi-razão, frações positivas de subsampling, interação método $\times$ razão, iteração multi-classe e persistência.  
**Where**: `tests/test_p2_study_expansion.py`  
**Depends on**: T1, T2  
**Reuses**: existing test fixtures  
**Requirement**: EXP-01, EXP-02, EXP-03  

**Done when**:

- [ ] Teste valida execução multi-razão e subsampling positivo proporcional
- [ ] Teste valida cálculo de interação método $\times$ razão com IC 95% e testes pareados
- [ ] Teste valida execução multi-classe e identificação de `target_class` nos registros
- [ ] Teste valida compatibilidade dos carregadores de publicação com schemas multi-razão
- [ ] Gate check passes: `.venv/bin/pytest -v tests/test_p2_study_expansion.py`  

**Tests**: integration  
**Gate**: full  
**Commit**: `test(study_expansion): add regression test suite for multi-ratio and multi-target classes`  

---

### T4: Atualizar Seção 2 do Notebook 03 com parametrização e tabelas multi-razão

**What**: Atualizar a Seção 2 de `notebooks/03_tier1_mcts_benchmarks.ipynb` parametrizando `IMBALANCE_RATIOS_AXIS1` e `TARGET_CLASSES_AXIS1`, exibindo o resumo por estratégia/razão e a tabela de interação método $\times$ razão quando múltiplas razões forem avaliadas.  
**Where**: `notebooks/03_tier1_mcts_benchmarks.ipynb`  
**Depends on**: T3  
**Reuses**: `src/training/benchmark_runner.py`  
**Requirement**: EXP-03  

**Done when**:

- [ ] Notebook 03 Seção 2 documenta e expõe `IMBALANCE_RATIOS_AXIS1` e `TARGET_CLASSES_AXIS1`
- [ ] Notebook 03 exibe tabela de interação método $\times$ razão quando múltiplas razões forem executadas
- [ ] Teste de fumaça do Notebook 03 executa com exit code 0
- [ ] Gate check passes: `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_p2_study_expansion.py tests/test_tier1_benchmarks_real.py -k "notebook_03"`  

**Tests**: integration  
**Gate**: build  
**Commit**: `feat(notebooks): expose multi-ratio and multi-class parameterization in notebook 03 axis 1`  

---

## Checks

**C1** - `run_benchmark()` evaluates multiple imbalance ratios under paired seeds with proportional positive subsampling (EXP-01, AC 1, AC 2)  
Proof: run unit | tests/test_p2_study_expansion.py::test_multi_ratio_execution_and_positive_fractions PASSED  

**C2** - `compute_method_by_ratio_interaction()` computes paired interaction effects across imbalance ratios (EXP-01, AC 3)  
Proof: run unit | tests/test_p2_study_expansion.py::test_method_by_ratio_interaction_calculation PASSED  

**C3** - Individual CSV rows and JSON payload persist `imbalance_ratio` and `ratio_interactions` (EXP-01, AC 4)  
Proof: run unit | tests/test_p2_study_expansion.py::test_multi_ratio_persistence_and_interactions_payload PASSED  

**C4** - `run_benchmark()` accepts `target_classes: tuple[int, ...]` and evaluates multiple target classes under identical seed schedules (EXP-02, AC 1, AC 2)  
Proof: run unit | tests/test_p2_study_expansion.py::test_multi_target_class_execution PASSED  

**C5** - Output JSON and CSV record `target_class` for every individual run and aggregate metrics across classes (EXP-02, AC 3, AC 4)  
Proof: run unit | tests/test_p2_study_expansion.py::test_target_class_persistence_and_aggregation PASSED  

**C6** - Notebook 03 Section 2 displays multi-ratio and multi-class configuration and executes cleanly in SMOKE mode (EXP-03, AC 1, AC 3)  
Proof: run unit | tests/test_p2_study_expansion.py::test_notebook_03_multi_ratio_and_multi_class_integration PASSED  

---

## Handoff

- **Dispatched:** none
