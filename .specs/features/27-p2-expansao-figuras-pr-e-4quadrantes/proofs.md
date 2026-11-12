# 27-p2-expansao-figuras-pr-e-4quadrantes proofs

**HEAD**: 4c03a8e
**Tree**: clean
**Shell**: /bin/sh
**Result**: 6 pass, 0 fail

| Check | Proof | Result |
| --- | --- | --- |
| C1 | `run unit / tests/test_p2_pr_and_quadrant_figures.py::test_precision_recall_curves_plot_generation PASSED` | pass |
| C2 | `run unit / tests/test_p2_pr_and_quadrant_figures.py::test_quadrant_evaluation_plot_generation PASSED` | pass |
| C3 | `run unit / tests/test_p2_pr_and_quadrant_figures.py::test_generate_all_artifacts_includes_p2_figures PASSED` | pass |
| C4 | `run unit / tests/test_p2_pr_and_quadrant_figures.py::test_notebook_05_integration_p2 PASSED` | pass |
| C5 | `run unit / tests/test_p2_pr_and_quadrant_figures.py::test_latex_figure_blocks_p2 PASSED` | pass |
| Build | `.venv/bin/python -m compileall -q src tests` | pass |

## Output

`Run unit` - exit 0

```
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-9.2.0, pluggy-1.7.0 -- /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm
configfile: pyproject.toml
plugins: platformdirs-4.12.4
collecting ... collected 5 items

tests/test_p2_pr_and_quadrant_figures.py::test_precision_recall_curves_plot_generation PASSED [ 20%]
tests/test_p2_pr_and_quadrant_figures.py::test_quadrant_evaluation_plot_generation PASSED [ 40%]
tests/test_p2_pr_and_quadrant_figures.py::test_generate_all_artifacts_includes_p2_figures PASSED [ 60%]
tests/test_p2_pr_and_quadrant_figures.py::test_latex_figure_blocks_p2 PASSED [ 80%]
tests/test_p2_pr_and_quadrant_figures.py::test_notebook_05_integration_p2 PASSED [100%]

============================== 5 passed in 8.21s ===============================
```
