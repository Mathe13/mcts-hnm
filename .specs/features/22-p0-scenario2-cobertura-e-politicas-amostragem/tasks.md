# F22: P0 — Investigação da Cobertura no Scenario 2 e Variantes de Amostragem Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, the walk, check_proofs, the Verifier and the screens sub-agent).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Design**: `.specs/features/22-p0-scenario2-cobertura-e-politicas-amostragem/design.md`  
**Status**: Ready  
**Approved**: 2026-11-11 - user: "sim"  
**Build**: `.venv/bin/python -m compileall -q src tests`  
**Run unit**: `.venv/bin/pytest -v tests/test_scenario2_coverage_and_variants.py`  

---

## Test Coverage Matrix

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Balanced Policy (`src/training/trainer.py`) | unit | 25% focal stream (16/48), MinMax Q-norm, semantic traversal | `tests/test_scenario2_coverage_and_variants.py` | `.venv/bin/pytest -v tests/test_scenario2_coverage_and_variants.py` |
| Benchmark Runner (`src/training/benchmark_runner.py`) | integration | Full variant execution, $H_{\text{norm}}$, $C_{\text{cov}}$, correlation | `tests/test_scenario2_coverage_and_variants.py` | `.venv/bin/pytest -v tests/test_scenario2_coverage_and_variants.py` |
| Paired Causal Diagnostics (`benchmark_runner.py`) | unit | Pairwise deltas vs uniform, Cohen's d, Spearman rank correlation | `tests/test_scenario2_coverage_and_variants.py` | `.venv/bin/pytest -v tests/test_scenario2_coverage_and_variants.py` |
| Notebook 03 Integration (`notebooks/03_tier1_mcts_benchmarks.ipynb`) | integration | Section 4 execution with all strategies and diagnostic report | `tests/test_tier1_benchmarks_real.py` | `.venv/bin/pytest -v tests/test_tier1_benchmarks_real.py -k "notebook_03"` |

---

## Gate Check Commands

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | After tasks with unit tests only | `.venv/bin/pytest -v tests/test_scenario2_coverage_and_variants.py` |
| Full | After tasks with integration tests | `.venv/bin/pytest -v tests/test_scenario2_coverage_and_variants.py tests/test_tier1_benchmarks_real.py` |
| Build | After phase completion or config/entity-only tasks | `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_scenario2_coverage_and_variants.py` |

---

## Execution Plan

Phases are ordered and run sequentially - each phase completes before the next begins, and tasks within a phase execute in order.

### Phase 1: Política Corretiva de Cobertura Balanceada (S2-02)

Implementa `mcts_coverage_balanced` em `src/training/trainer.py` com fração focal 25% (16 focais + 48 uniformes em batch 64).

```
T1
```

### Phase 2: Orquestração das Variantes e Diagnóstico Quantitativo (S2-01, S2-03)

Habilita a execução das 7 variantes de amostragem no Scenario 2 e calcula métricas completas de diagnóstico e correlação em `src/training/benchmark_runner.py`.

```
T2
```

### Phase 3: Suíte Automatizada de Testes de Cobertura (S2-01, S2-02, S2-03)

Implementa `tests/test_scenario2_coverage_and_variants.py` validando todas as variantes e asserções de diagnóstico por classe.

```
T3
```

### Phase 4: Integração do Notebook 03 e Relatório Causal (S2-04)

Atualiza a Seção 4 do Notebook 03 com a execução das variantes, tabela de diagnósticos e parecer causal.

```
T4
```

---

## Task Breakdown

### T1: Implementar política `mcts_coverage_balanced` no trainer

**What**: Adicionar `"mcts_coverage_balanced"` a `SUPPORTED_STRATEGIES` em `src/training/trainer.py`, configurando `dual_stream_ratio = 0.25` (16 instâncias da classe MCTS + 48 instâncias uniformes em batch 64) no `train_scenario2()`.  
**Where**: `src/training/trainer.py`  
**Depends on**: none  
**Reuses**: `src/training/trainer.py` (`train_scenario2`)  
**Requirement**: S2-02  

**Done when**:

- [x] `"mcts_coverage_balanced"` registrado em `SUPPORTED_STRATEGIES`
- [x] `train_scenario2()` particiona batch com 25% focal e 75% uniforme quando a estratégia é selecionada
- [x] Preserva `mcts_enhanced` inalterado com 50% focal
- [x] Gate check passes: `.venv/bin/python -m compileall -q src tests`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `feat(trainer): implement mcts coverage balanced sampling strategy`  

---

### T2: Orquestrar variantes completas e telemetria quantitativa no runner

**What**: Atualizar `run_benchmark(scenario="scenario2")` em `src/training/benchmark_runner.py` para suportar `uniform_grouped`, `mcts_strict`, `mcts_qnorm_only`, `mcts_dualstream_only`, `mcts_enhanced` e `mcts_coverage_balanced`, calculando $H_{\text{norm}}$, $C_{\text{cov}}$, $N_{\text{unvisited}}$ e correlação Spearman/Pearson de exposição vs recall.  
**Where**: `src/training/benchmark_runner.py`  
**Depends on**: T1  
**Reuses**: `src/training/trainer.py`  
**Requirement**: S2-01, S2-03  

**Done when**:

- [x] `run_benchmark()` executa as estratégias do Scenario 2 registrando métricas padronizadas
- [x] Telemetria por execução calcula $H_{\text{norm}}$, $C_{\text{cov}}$, $N_{\text{unvisited}}$ e correlação exposição–recall
- [x] JSON e CSV exportam todas as colunas de diagnóstico por estratégia e semente
- [x] Gate check passes: `.venv/bin/python -m compileall -q src tests`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `feat(benchmark): orchestrate scenario2 variants with quantitative coverage telemetry`  

---

### T3: Implementar suíte de testes de regressão do Scenario 2

**What**: Criar `tests/test_scenario2_coverage_and_variants.py` testando a execução das 7 variantes sob Scenario 2, validando as proporções de batch de `mcts_coverage_balanced`, asserindo o cálculo de $H_{\text{norm}}$ e verificando o formato de exportação.  
**Where**: `tests/test_scenario2_coverage_and_variants.py`  
**Depends on**: T1, T2  
**Reuses**: existing test harness  
**Requirement**: S2-01, S2-02, S2-03  

**Done when**:

- [x] Teste assere execução correta de `uniform_grouped` e `mcts_coverage_balanced`
- [x] Teste valida proporção 16/48 em batch 64 para `mcts_coverage_balanced`
- [x] Teste valida integridade dos diagnósticos quantitativos ($H_{\text{norm}}$, $C_{\text{cov}}$, correlação)
- [x] Gate check passes: `.venv/bin/pytest -v tests/test_scenario2_coverage_and_variants.py`  

**Tests**: integration  
**Gate**: full  
**Commit**: `test(scenario2): automated test suite for coverage variants and diagnostics`  

---

### T4: Atualizar Seção 4 do Notebook 03 com diagnósticos e análise causal

**What**: Atualizar a Seção 4 de `notebooks/03_tier1_mcts_benchmarks.ipynb` para executar `uniform`, `uniform_grouped`, `mcts_enhanced` e `mcts_coverage_balanced`, exibindo a tabela comparativa de diagnósticos e emitindo a inferência causal.  
**Where**: `notebooks/03_tier1_mcts_benchmarks.ipynb`  
**Depends on**: T3  
**Reuses**: `src/training/benchmark_runner.py`  
**Requirement**: S2-04  

**Done when**:

- [ ] Notebook 03 executa as 4 estratégias fundamentais no Scenario 2
- [ ] Exibe tabela comparativa consolidando Top-1, Macro-F1, Worst-10 recall, $H_{\text{norm}}$ e $N_{\text{unvisited}}$
- [ ] Testes de execução do Notebook 03 em SMOKE passam com exit code 0
- [ ] Gate check passes: `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_scenario2_coverage_and_variants.py tests/test_tier1_benchmarks_real.py -k "notebook_03"`  

**Tests**: integration  
**Gate**: build  
**Commit**: `feat(notebooks): expand notebook 03 scenario 2 with coverage diagnostics and causal report`  

---

## Checks

**C1** - `mcts_coverage_balanced` is registered in `SUPPORTED_STRATEGIES` and partitions batches with 25% focal ratio (S2-02, AC 1, AC 2)  
Proof: run unit | tests/test_scenario2_coverage_and_variants.py::test_mcts_coverage_balanced_strategy_registration PASSED  

**C2** - `mcts_coverage_balanced` composes minibatches with 16 focal and 48 uniform samples in batch 64 (S2-02, AC 1)  
Proof: run unit | tests/test_scenario2_coverage_and_variants.py::test_mcts_coverage_balanced_batch_composition PASSED  

**C3** - `run_benchmark()` executes `uniform_grouped` and `mcts_coverage_balanced` in Scenario 2 (S2-01, AC 1, AC 2)  
Proof: run unit | tests/test_scenario2_coverage_and_variants.py::test_scenario2_benchmark_runs_all_variants PASSED  

**C4** - Scenario 2 records $H_{\text{norm}}$, $C_{\text{cov}}$, $N_{\text{unvisited}}$, and exposure-recall correlation (S2-03, AC 1, AC 3)  
Proof: run unit | tests/test_scenario2_coverage_and_variants.py::test_scenario2_records_quantitative_diagnostics PASSED  

**C5** - Pairwise statistical differences against uniform baseline are computed and persisted (S2-03, AC 2, AC 4)  
Proof: run unit | tests/test_scenario2_coverage_and_variants.py::test_scenario2_paired_differences_against_uniform PASSED  

**C6** - Unsupported strategy in Scenario 2 raises `ValueError` with clear message (S2-01, AC 4)  
Proof: run unit | tests/test_scenario2_coverage_and_variants.py::test_scenario2_rejects_unsupported_strategy PASSED  

---

## Handoff

- **Dispatched:** none
