# 06-paper-and-artifacts proofs

**HEAD**: bf906d3
**Tree**: clean
**Shell**: /bin/sh
**Result**: 10 pass, 0 fail

| Check | Proof | Result |
| --- | --- | --- |
| C1 | `run unit / tests/test_paper_artifacts.py::test_load_paper_metadata PASSED` | pass |
| C2 | `run unit / tests/test_paper_artifacts.py::test_publication_figures_generation PASSED` | pass |
| C3 | `run unit / tests/test_paper_artifacts.py::test_latex_tables_generation PASSED` | pass |
| C4 | `run unit / tests/test_paper_artifacts.py::test_benchmark_results_loader_and_synthesizer PASSED` | pass |
| C5 | `run unit / tests/test_paper_artifacts.py::test_generate_all_paper_artifacts_directory_creation PASSED` | pass |
| C6 | `run unit / tests/test_paper_artifacts.py::test_manuscript_syntax_and_citations PASSED` | pass |
| C7 | `run unit / tests/test_paper_artifacts.py::test_notebook_05_execution PASSED` | pass |
| C8 | `run unit / tests/test_paper_artifacts.py::test_notebook_00_execution PASSED` | pass |
| C9 | `run unit / tests/test_paper_artifacts.py::test_compile_paper_pdf PASSED` | pass |
| Build | `.venv/bin/python -m compileall -q src tests` | pass |

## Output

`Run unit` - exit 0

```
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/trainer.py:300: UserWarning: Leaf node 21 (class 4) has only 30 instances (< requested K_mcts=48); sampling with replacement.
    batch = mcts_sampler.sample_batch(mcts_leaf_id)

tests/test_training_loop.py::test_notebook_03_execution
tests/test_training_loop.py::test_notebook_03_execution
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/trainer.py:473: UserWarning: Leaf node 21 (class 4) has only 30 instances (< requested K_mcts=32); sampling with replacement.
    batch = mcts_sampler.sample_batch(mcts_leaf_id)

tests/test_training_loop.py::test_run_benchmark_multi_seed_and_export
tests/test_training_loop.py::test_run_benchmark_multi_seed_and_export
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/trainer.py:300: UserWarning: Leaf node 3 (class 1) has only 20 instances (< requested K_mcts=24); sampling with replacement.
    batch = mcts_sampler.sample_batch(mcts_leaf_id)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================== 74 passed, 7 warnings in 66.32s (0:01:06) ===================
```
