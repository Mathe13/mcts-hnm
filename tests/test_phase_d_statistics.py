"""Tests for Phase D: Inferential Statistics and Method-by-Ratio Interaction."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.statistics import (
    compute_method_by_ratio_interaction,
    compute_paired_method_differences,
    compute_paired_statistical_tests,
)


def test_paired_method_differences_and_cohens_d() -> None:
    # 5 paired scores for method A and method B
    scores_a = [0.85, 0.88, 0.84, 0.87, 0.86]
    scores_b = [0.80, 0.81, 0.82, 0.79, 0.80]

    stats = compute_paired_method_differences(scores_a, scores_b)

    assert stats["n"] == 5
    assert stats["mean_a"] > stats["mean_b"]
    assert stats["mean_diff"] > 0
    assert stats["std_diff"] > 0
    assert stats["se_diff"] > 0
    assert stats["cohens_d"] > 0
    # Cohen's d is mean_diff / std_diff
    assert np.isclose(stats["cohens_d"], stats["mean_diff"] / stats["std_diff"])

    # Check CI bounds
    lower, upper = stats["ci_95"]
    assert lower < stats["mean_diff"] < upper
    assert stats["ci_95_lower"] == lower
    assert stats["ci_95_upper"] == upper

    # Backward compatibility with compute_paired_statistical_tests
    stats_compat = compute_paired_statistical_tests(scores_a, scores_b)
    assert stats_compat["t_stat"] == stats["t_stat"]
    assert stats_compat["t_pvalue"] == stats["t_pvalue"]


def test_method_by_ratio_interaction_statistics() -> None:
    # Ratio 1 (1:50): MCTS is roughly +0.05 over Uniform
    r1_mcts = [0.85, 0.86, 0.87, 0.84, 0.88]
    r1_uni = [0.80, 0.81, 0.83, 0.79, 0.82]

    # Ratio 2 (1:500): MCTS advantage grows to roughly +0.15 over Uniform
    r2_mcts = [0.65, 0.67, 0.66, 0.64, 0.69]
    r2_uni = [0.50, 0.51, 0.52, 0.49, 0.53]

    interaction = compute_method_by_ratio_interaction(
        scores_r1_a=r1_mcts,
        scores_r1_b=r1_uni,
        scores_r2_a=r2_mcts,
        scores_r2_b=r2_uni,
    )

    assert interaction["n"] == 5
    assert interaction["mean_diff_r1"] > 0
    assert interaction["mean_diff_r2"] > interaction["mean_diff_r1"]
    assert interaction["mean_interaction"] > 0
    assert interaction["t_stat"] > 0
    assert interaction["ci_95_lower"] > 0.0  # Significant positive interaction
    assert interaction["ci_95_upper"] > interaction["ci_95_lower"]


def test_statistical_summary_confidence_intervals() -> None:
    # Zero differences case
    scores_identical = [0.5, 0.5, 0.5, 0.5, 0.5]
    zero_stats = compute_paired_method_differences(scores_identical, scores_identical)
    assert zero_stats["mean_diff"] == 0.0
    assert zero_stats["cohens_d"] == 0.0
    assert zero_stats["ci_95"] == (0.0, 0.0)
    assert zero_stats["t_stat"] == 0.0
    assert zero_stats["t_pvalue"] == 1.0

    # Error conditions
    with pytest.raises(ValueError, match="Expected 1D"):
        compute_paired_method_differences(np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="sizes must match"):
        compute_paired_method_differences([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="At least 2"):
        compute_paired_method_differences([1.0], [1.0])
    with pytest.raises(ValueError, match="finite"):
        compute_paired_method_differences([1.0, np.nan], [1.0, 2.0])
