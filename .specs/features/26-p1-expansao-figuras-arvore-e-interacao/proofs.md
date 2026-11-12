# 26-p1-expansao-figuras-arvore-e-interacao proofs

**HEAD**: 3be20c3
**Tree**: clean
**Shell**: /bin/sh
**Result**: 7 pass, 0 fail

| Check | Proof | Result |
| --- | --- | --- |
| C1 | `run unit / tests/test_p1_expansion_figures.py::test_class_exposure_vs_delta_recall_plot_generation PASSED` | pass |
| C2 | `run unit / tests/test_p1_expansion_figures.py::test_ward_vs_semantic_dendrogram_plot_generation PASSED` | pass |
| C3 | `run unit / tests/test_p1_expansion_figures.py::test_interaction_ratio_plot_generation PASSED` | pass |
| C4 | `run unit / tests/test_p1_expansion_figures.py::test_generate_all_artifacts_includes_p1_figures PASSED` | pass |
| C5 | `run unit / tests/test_p1_expansion_figures.py::test_notebook_05_integration_p1 PASSED` | pass |
| C6 | `run unit / tests/test_p1_expansion_figures.py::test_latex_figure_blocks_p1 PASSED` | pass |
| Build | `.venv/bin/python -m compileall -q src tests` | pass |

## Output

`Run unit` - exit 0

```

tests/test_p1_expansion_figures.py::test_class_exposure_vs_delta_recall_plot_generation PASSED [ 16%]
tests/test_p1_expansion_figures.py::test_ward_vs_semantic_dendrogram_plot_generation PASSED [ 33%]
tests/test_p1_expansion_figures.py::test_interaction_ratio_plot_generation PASSED [ 50%]
tests/test_p1_expansion_figures.py::test_generate_all_artifacts_includes_p1_figures PASSED [ 66%]
tests/test_p1_expansion_figures.py::test_latex_figure_blocks_p1 PASSED   [ 83%]
tests/test_p1_expansion_figures.py::test_notebook_05_integration_p1 PASSED [100%]

=============================== warnings summary ===============================
tests/test_p1_expansion_figures.py::test_class_exposure_vs_delta_recall_plot_generation
  /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/src/training/exposure.py:177: NearConstantInputWarning: An input array is nearly constant; the computed correlation coefficient may be inaccurate.
    p_res = stats.pearsonr(exp, rec)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================== 6 passed, 1 warning in 14.13s =========================
```
