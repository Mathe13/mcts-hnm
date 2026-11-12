# 25-p0-expansao-figuras-pareto-e-entropia proofs

**HEAD**: 5dc2c7d
**Tree**: clean
**Shell**: /bin/sh
**Result**: 6 pass, 0 fail

| Check | Proof | Result |
| --- | --- | --- |
| C1 | `run unit / tests/test_p0_pareto_entropy_figures.py::test_pareto_cost_performance_plot_generation PASSED` | pass |
| C2 | `run unit / tests/test_p0_pareto_entropy_figures.py::test_coverage_entropy_dynamics_plot_generation PASSED` | pass |
| C3 | `run unit / tests/test_p0_pareto_entropy_figures.py::test_generate_all_artifacts_includes_new_figures PASSED` | pass |
| C4 | `run unit / tests/test_p0_pareto_entropy_figures.py::test_latex_figure_blocks PASSED` | pass |
| C5 | `run unit / tests/test_p0_pareto_entropy_figures.py::test_notebook_05_integration PASSED` | pass |
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

tests/test_p0_pareto_entropy_figures.py::test_pareto_cost_performance_plot_generation PASSED [ 20%]
tests/test_p0_pareto_entropy_figures.py::test_coverage_entropy_dynamics_plot_generation PASSED [ 40%]
tests/test_p0_pareto_entropy_figures.py::test_generate_all_artifacts_includes_new_figures PASSED [ 60%]
tests/test_p0_pareto_entropy_figures.py::test_latex_figure_blocks PASSED [ 80%]
tests/test_p0_pareto_entropy_figures.py::test_notebook_05_integration PASSED [100%]

============================== 5 passed in 11.42s ==============================
```
