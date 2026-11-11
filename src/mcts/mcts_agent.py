"""Re-export of core MCTS agent and diagnostics for module naming consistency."""

from __future__ import annotations

from src.mcts.agent import (
    MCTSHNMAgent,
    compute_mcts_reward,
    compute_ucb1_scores,
    get_mcts_policy_diagnostics,
    normalize_sibling_q_values,
)

__all__ = [
    "MCTSHNMAgent",
    "compute_mcts_reward",
    "compute_ucb1_scores",
    "get_mcts_policy_diagnostics",
    "normalize_sibling_q_values",
]
