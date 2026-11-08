# AGENTS.md: AI Pair Programming and Engineering Guidelines

This repository follows the **TLC Spec-Driven Development (SDD) v4** standard. All AI agents, contributors, and automated tooling working within this codebase MUST adhere to the principles, workflows, and quality gates outlined below.

---

## 1. Core Principles

1. **Precision of Scope & Zero Unverified Assumptions:**
   - Always clarify ambiguities before modifying core architectural interfaces.
   - Separate concerns strictly: algorithms (MCTS/RL logic) must remain decoupled from specific dataset loaders or neural network backbones.
2. **Scientific Rigor First:**
   - Every experimental claim must be substantiated by empirical data with random seeds ($N \ge 5$) and standard deviations.
   - Code must be reproducible: set explicit random seeds for PyTorch, NumPy, and Python standard libraries.
3. **Spec-Driven Execution Flow:**
   - For any non-trivial feature or refactoring, follow the 4-phase adaptive cycle: `Specify` $\rightarrow$ `Design` $\rightarrow$ `Tasks` $\rightarrow$ `Execute` (see `.agents/skills/tlc-spec-driven/SKILL.md`).
   - Write verifiable acceptance criteria using **EARS** notation (*Easy Approach to Requirements Syntax*).

---

## 2. Technical Stack & Environment

- **Language:** Python 3.10+
- **Deep Learning Framework:** PyTorch 2.x, Torchvision
- **Scientific & ML Libraries:** NumPy, SciPy (for hierarchical clustering), Scikit-Learn
- **Package & Dependency Management:** `uv` as the primary tool (with standard `pyproject.toml` support)
- **Typing:** Strict type hints (`typing.Optional`, `typing.Tuple`, `typing.Dict`, `torch.Tensor`, etc.)
- **Testing:** `pytest` for unit and integration testing

---

## 3. Architecture & Implementation Guidelines

1. **Two-Tier Development Strategy:**
   - **Tier 1 (Embedding Fast-Prototyping):** When evaluating search mechanics, bandit exploration parameters, or tree traversal rules, use frozen feature embeddings. Experiments should execute within minutes, not hours.
   - **Tier 2 (End-to-End Vision):** Run full CNN fine-tuning only for validating complete gradient interactions and final benchmark reporting.
2. **MCTS Implementation Standards:**
   - Keep tree node traversal in fast CPU arrays or NumPy structures to avoid Python object overhead.
   - Never allow unbounded loss values to dominate UCB exploration; always employ **Min-Max normalization** or clipping over $Q$-values.
   - Implement the **Dual-Stream Batching** mechanism (50% uniform anchor, 50% MCTS mined) to safeguard against catastrophic forgetting and gradient divergence.
3. **Modularity:**
   - `src/trees/`: Pure tree representations, building algorithms, and serialization.
   - `src/mcts/`: Tree search, rollout simulation, and backpropagation independent of vision details.
   - `src/models/`: Neural network classifiers and feature extractors.
   - `src/training/`: Training loops and baseline comparisons (Uniform, OHEM).
   - `src/utils/`: Evaluation metrics (F1, AUPRC, tree visit distribution) and LaTeX figure export.

---

## 4. Git & Commit Guidelines

- **Atomic Commits:** Make small, logical commits adhering to Conventional Commits:
  - `feat: <description>` (New feature or algorithm addition)
  - `fix: <description>` (Bug fix in search, training, or metric calculation)
  - `test: <description>` (Adding or updating unit/integration tests)
  - `docs: <description>` (Documentation or paper drafting updates)
  - `refactor: <description>` (Code restructuring without behavioral change)
- **Never Commit Large Artifacts:**
  - Raw image datasets, model checkpoints (`.pt`, `.pth`), and multi-gigabyte embeddings belong in `.gitignore`.
  - Only commit source code, unit tests, configurations, lightweight summary metrics, and LaTeX files.

---

## 5. TLC Spec-Driven Workflow Commands

The `tlc-spec-driven` skill is installed locally at `.agents/skills/tlc-spec-driven/`. Use its deterministic Python scripts to enforce quality gates:

```bash
# Validate specification formatting before implementation
python3 .agents/skills/tlc-spec-driven/scripts/validate_spec.py <feature_name>

# Validate task decomposition and criteria checks
python3 .agents/skills/tlc-spec-driven/scripts/validate_tasks.py <feature_name>

# Verify proofs and test assertions upon task completion
python3 .agents/skills/tlc-spec-driven/scripts/check_proofs.py <feature_name>
```

When interacting with human reviewers, present clean diffs, state exact evaluation metrics, and report empirical findings with statistical confidence.
