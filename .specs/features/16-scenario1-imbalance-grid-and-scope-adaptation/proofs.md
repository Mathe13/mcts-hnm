# Proofs: F16 Scenario 1 Imbalance Grid Execution and Scope Adaptation

**Date**: 2026-11-10

## 1. Full Test Suite Regression Output
```text
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-9.2.0, pluggy-1.7.0
rootdir: /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm
configfile: pyproject.toml
plugins: platformdirs-4.12.4
collected 163 items

tests/test_automated_suite_real.py ............                          [  7%]
tests/test_documentation_and_reproducibility.py ........                 [ 12%]
tests/test_dynamic_expansion.py ...........                              [ 19%]
tests/test_experiment_config.py ...............                          [ 28%]
tests/test_extract_embeddings.py .....                                   [ 31%]
tests/test_mcts_agent.py ............                                    [ 38%]
tests/test_notebooks_00_01_02.py ............                            [ 46%]
tests/test_paper_artifacts.py ..........                                 [ 52%]
tests/test_paper_artifacts_real.py .................                     [ 62%]
tests/test_seed_and_taxonomy.py ......                                   [ 66%]
tests/test_splits_and_notebook.py ......                                 [ 69%]
tests/test_tier1_benchmarks_real.py .............                        [ 77%]
tests/test_tier2_e2e_real.py ...........                                 [ 84%]
tests/test_training_loop.py .............                                [ 92%]
tests/test_trees.py ............                                         [100%]

====================== 163 passed, 31 warnings in 41.62s =======================
```

## 2. Dynamic Scope Adaptation Verification Output
```text
tests/test_paper_artifacts_real.py::test_load_real_benchmark_results_extracts_imbalance_ratio PASSED
tests/test_paper_artifacts_real.py::test_generate_scenario1_table_adapts_to_single_ratio PASSED
tests/test_tier1_benchmarks_real.py::test_scenario1_imbalance_ratio_parameterization PASSED
```

## 3. Directory Isolation Audit
```bash
find results/real -type f
# Output: empty (0 files written to results/real during SMOKE test suite)
```
