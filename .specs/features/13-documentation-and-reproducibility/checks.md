# Scientific Documentation and Reproducibility - Checks

Profile: light
Plan: `.specs/features/13-documentation-and-reproducibility/plan.md`

## Intent

8 checks in 2 slices · 2 one-way doors · 0 open

## Checks

### S1 - Root README Documentation Standard · 2 files · 20 KB · ~7k

**C1** - `README.md` defines `ExecutionMode.SMOKE` and `ExecutionMode.REAL` semantics with isolated directories (S1, AC 1)
Proof: `pytest tests/test_documentation_and_reproducibility.py::test_readme_execution_mode_definitions`

**C2** - `README.md` documents dataset requirements, directory layout, and path isolation rules (S1, AC 2)
Proof: `pytest tests/test_documentation_and_reproducibility.py::test_readme_dataset_and_directory_layout`

**C3** - `README.md` documents step-by-step commands for smoke testing, real experiments, and artifact generation (S1, AC 3)
Proof: `pytest tests/test_documentation_and_reproducibility.py::test_readme_reproduction_commands`

**C4** - `README.md` documents known experimental limitations and links to `docs/experimental_reproducibility.md` (S1, AC 4)
Proof: `pytest tests/test_documentation_and_reproducibility.py::test_readme_limitations_and_guide_link`

### S2 - Experimental Reproducibility Guide · 2 files · 25 KB · ~8k

**C5** - `docs/experimental_reproducibility.md` catalogs completed, pending, and historical experiments (S2, AC 1)
Proof: `pytest tests/test_documentation_and_reproducibility.py::test_reproducibility_guide_experiment_catalog`

**C6** - `docs/experimental_reproducibility.md` documents tree geometry methodological review (Ward vs Average linkage) (S2, AC 2)
Proof: `pytest tests/test_documentation_and_reproducibility.py::test_reproducibility_guide_tree_methodology`

**C7** - `docs/experimental_reproducibility.md` documents negative mining budget equivalence and dynamic leaf expansion (S2, AC 3)
Proof: `pytest tests/test_documentation_and_reproducibility.py::test_reproducibility_guide_budget_and_expansion`

**C8** - All internal markdown file links across `README.md` and `docs/experimental_reproducibility.md` resolve to valid files (S2, AC 4)
Proof: `pytest tests/test_documentation_and_reproducibility.py::test_documentation_link_integrity`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| `README.md` sections (4) | modes C1 · layout C2 · reproduction C3 · limitations C4 | - |
| `experimental_reproducibility.md` sections (4) | catalog C5 · linkage C6 · budget C7 · links C8 | - |

- Claims naming a status code, route or response shape: C1, C2, C3, C4, C5, C6, C7, C8 - each has a proof that crosses the boundary
- No other check claims more than the single case its proof exercises

## Swept

- validation: C1, C2, C5, C8
- failure modes: C1, C8
- idempotency: C1, C2, C3, C4, C5, C6, C7, C8
- authorization: n/a - documentation and test assertions
- concurrency: C8
- data lifecycle: C2, C5
- dependency failure: C8
- state transitions: C1
- observability: C3, C4, C5

## Handoff

- S1-S2 = 45 KB total across README update, reproducibility guide, and test suite; estimate ~15k tokens, well under the 150k budget - one builder
