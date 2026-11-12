# F21: P0 — Sensibilidade com Subsampling Equivalente e Política de Avaliação Inequívoca Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, the walk, check_proofs, the Verifier and the screens sub-agent).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Design**: `.specs/features/21-p0-sensibilidade-split-e-avaliacao-inequivoca/design.md`  
**Status**: Ready  
**Approved**: 2026-11-11 - user: "prossiga"  
**Build**: `.venv/bin/python -m compileall -q src tests`  
**Run unit**: `.venv/bin/pytest -v tests/test_sensitivity_runner_equivalence.py tests/test_tier1_benchmarks_real.py -k "sensitivity"`  

---

## Test Coverage Matrix

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Sensitivity Runner (`src/training/benchmark_runner.py`) | unit | Subsampling via `build_scenario1_split`, `imbalance_ratio`, `num_steps=1000` | `tests/test_sensitivity_runner_equivalence.py` | `.venv/bin/pytest -v tests/test_sensitivity_runner_equivalence.py` |
| Evaluation Policy (`src/training/benchmark_runner.py`) | unit | 4-quadrant metrics (`last`/`best_val` x `0.5`/`calibrated`), JSON/CSV keys | `tests/test_sensitivity_runner_equivalence.py` | `.venv/bin/pytest -v tests/test_sensitivity_runner_equivalence.py` |
| Equivalence Assertions (`tests/test_sensitivity_runner_equivalence.py`) | integration | Exact numpy index equality across paired seeds with `run_benchmark` | `tests/test_sensitivity_runner_equivalence.py` | `.venv/bin/pytest -v tests/test_sensitivity_runner_equivalence.py` |
| Tier-1 Benchmark Regressions (`tests/test_tier1_benchmarks_real.py`) | integration | Full compatibility with sensitivity ablation and export assertions | `tests/test_tier1_benchmarks_real.py` | `.venv/bin/pytest -v tests/test_tier1_benchmarks_real.py -k "sensitivity"` |

---

## Gate Check Commands

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | After tasks with unit tests only | `.venv/bin/pytest -v tests/test_sensitivity_runner_equivalence.py` |
| Full | After tasks with integration tests | `.venv/bin/pytest -v tests/test_sensitivity_runner_equivalence.py tests/test_tier1_benchmarks_real.py` |
| Build | After phase completion or config/entity-only tasks | `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_sensitivity_runner_equivalence.py` |

---

## Execution Plan

Phases are ordered and run sequentially - each phase completes before the next begins, and tasks within a phase execute in order.

### Phase 1: Subsampling Equivalente e Harmonização da Sensibilidade (SENS-01, SENS-02)

Adota `build_scenario1_split` com `imbalance_ratio="1:50"` default e `num_steps=1000` default em `run_sensitivity_ablation()`, salvando metadados de razão e tamanho de pool.

```
T1
```

### Phase 2: Política Inequívoca de Avaliação nos 4 Quadrantes (EVAL-01, EVAL-02)

Computa e serializa estruturadamente nos resultados os 4 quadrantes métricos (`last`/`best_val` x `0.5`/`calibrated`), garantindo rastreabilidade sem ambiguidade.

```
T2
```

### Phase 3: Teste Pareado de Equivalência Estrita de Índices (EQUIV-01)

Implementa teste automatizado comparando semente por semente os índices positivos e negativos selecionados por `run_sensitivity_ablation()` e `run_benchmark()`.

```
T3
```

### Phase 4: Compatibilização da Suíte de Testes e Regressões

Atualiza testes legados de sensibilidade para a nova assinatura e validação da suíte completa de testes.

```
T4
```

---

## Task Breakdown

### T1: Implementar particionamento via `build_scenario1_split` em `run_sensitivity_ablation()`

**What**: Modificar `run_sensitivity_ablation()` em `src/training/benchmark_runner.py` para usar `build_scenario1_split` por semente, adicionar parâmetros `imbalance_ratio: str = "1:50"` e `num_steps: int = 1000` (com override), e persistir metadados `imbalance_ratio`, `num_positives_train`, `num_negatives_train`.  
**Where**: `src/training/benchmark_runner.py`  
**Depends on**: none  
**Reuses**: `src/data/splits.py` (`build_scenario1_split`, `IMBALANCE_RATIOS`)  
**Requirement**: SENS-01, SENS-02  

**Done when**:

- [x] `run_sensitivity_ablation` instancia `build_scenario1_split` para cada semente com `imbalance_ratio`
- [x] Defaults atualizados para `imbalance_ratio="1:50"` e `num_steps=1000`
- [x] Export de resultados registra `imbalance_ratio`, `num_positives_train`, e `num_negatives_train`
- [x] Gate check passes: `.venv/bin/python -m compileall -q src tests`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `feat(benchmark): align sensitivity ablation split and subsampling with axis 1`  

---

### T2: Formalizar avaliação inequívoca dos 4 quadrantes métricos

**What**: Estender `run_sensitivity_ablation()` em `src/training/benchmark_runner.py` para computar e reportar explicitamente os 4 quadrantes de métricas (`last`/`best_val` x `0.5`/`calibrated`), gerando chaves explícitas no JSON e CSV (`test_f1_last_thresh05`, `test_f1_last_calibrated`, `test_f1_best_val_thresh05`, `test_f1_best_val_calibrated`, etc.).  
**Where**: `src/training/benchmark_runner.py`  
**Depends on**: T1  
**Reuses**: `src/training/checkpoints.py`  
**Requirement**: EVAL-01, EVAL-02  

**Done when**:

- [x] Métricas nos checkpoints `last` e `best_val` avaliadas com threshold 0.5 e threshold calibrado na validação
- [x] Dicionário e JSON de exportação contêm os 4 pares de métricas sem lacunas
- [x] CSV rows incluem todas as colunas dos 4 quadrantes
- [x] Gate check passes: `.venv/bin/python -m compileall -q src tests`  

**Tests**: unit  
**Gate**: quick  
**Commit**: `feat(training): evaluate and export 4-quadrant metrics for sensitivity ablation`  

---

### T3: Implementar teste pareado de equivalência estrita de índices

**What**: Criar `tests/test_sensitivity_runner_equivalence.py` validando que `run_sensitivity_ablation()` e `run_benchmark()` selecionam conjuntos idênticos de índices positivos e negativos por semente, e conferindo que o pool em 1:50 possui ~9 positivos no CIFAR-100.  
**Where**: `tests/test_sensitivity_runner_equivalence.py`  
**Depends on**: T1, T2  
**Reuses**: `tests/test_phase_a_runner_equivalence.py`  
**Requirement**: EQUIV-01  

**Done when**:

- [x] Teste assere `np.array_equal(pos_indices_bench, pos_indices_sens)` para múltiplas seeds
- [x] Teste assere `np.array_equal(neg_indices_bench, neg_indices_sens)` para múltiplas seeds
- [x] Teste valida integridade dos 4 quadrantes métricos na saída da sensibilidade
- [x] Gate check passes: `.venv/bin/pytest -v tests/test_sensitivity_runner_equivalence.py`  

**Tests**: integration  
**Gate**: full  
**Commit**: `test(equivalence): strict paired index and evaluation equality between runners`  

---

### T4: Atualizar testes existentes e regressões de sensibilidade

**What**: Atualizar `tests/test_tier1_benchmarks_real.py` para compatibilidade com o novo split da sensibilidade e validação de não-regressão de toda a suíte.  
**Where**: `tests/test_tier1_benchmarks_real.py`  
**Depends on**: T3  
**Reuses**: existing test harness  
**Requirement**: SENS-01, EVAL-01, EQUIV-01  

**Done when**:

- [ ] Todos os testes de sensibilidade em `test_tier1_benchmarks_real.py` passam com a nova assinatura e split
- [ ] `test_phase_a_runner_equivalence.py` passa sem divergências
- [ ] Suíte completa de testes de benchmarks e sensibilidade passa com exit code 0
- [ ] Gate check passes: `.venv/bin/python -m compileall -q src tests && .venv/bin/pytest -v tests/test_sensitivity_runner_equivalence.py tests/test_tier1_benchmarks_real.py`  

**Tests**: integration  
**Gate**: build  
**Commit**: `fix(tests): reconcile tier-1 and phase-a test contracts with sensitivity split`  

---

## Checks

**C1** - `run_sensitivity_ablation()` applies `build_scenario1_split` using `imbalance_ratio="1:50"` and `seed` (SENS-01, AC 1, AC 2)  
Proof: run unit | tests/test_sensitivity_runner_equivalence.py::test_sensitivity_runner_uses_scenario1_split PASSED  

**C2** - `run_sensitivity_ablation()` produces identical positive and negative indices to `build_scenario1_split` per seed (SENS-01, AC 3)  
Proof: run unit | tests/test_sensitivity_runner_equivalence.py::test_sensitivity_indices_match_scenario1_split PASSED  

**C3** - `run_sensitivity_ablation()` output records `imbalance_ratio`, `num_positives_train`, and `num_negatives_train` (SENS-02, AC 4)  
Proof: run unit | tests/test_sensitivity_runner_equivalence.py::test_sensitivity_output_records_pool_metadata PASSED  

**C4** - 4-quadrant metrics (`last`/`best_val` x `0.5`/`calibrated`) computed and exported without nulls (EVAL-01, AC 1, AC 2, AC 3)  
Proof: run unit | tests/test_sensitivity_runner_equivalence.py::test_sensitivity_exports_4_quadrant_metrics PASSED  

**C5** - Missing validation data raises `ValueError` preventing silent uncalibrated execution (EVAL-01, AC 4)  
Proof: run unit | tests/test_sensitivity_runner_equivalence.py::test_sensitivity_requires_validation_data PASSED  

**C6** - Positive and negative training indices strictly match between `run_benchmark` and `run_sensitivity_ablation` (EQUIV-01, AC 1, AC 2, AC 3)  
Proof: run unit | tests/test_sensitivity_runner_equivalence.py::test_benchmark_and_sensitivity_index_equivalence PASSED  

---

## Handoff

- **Dispatched:** none
