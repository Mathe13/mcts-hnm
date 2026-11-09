"""Statistical hypothesis tests for multi-seed experimental evaluations."""

from __future__ import annotations

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def compute_paired_statistical_tests(
    scores_a: np.ndarray | list[float],
    scores_b: np.ndarray | list[float],
) -> dict[str, float]:
    r"""Compute paired Student's t-test and Wilcoxon signed-rank test across paired score samples.

    Args:
        scores_a: 1D numeric array or list of metrics from method A across $N \ge 2$ seeds.
        scores_b: 1D numeric array or list of metrics from method B across $N \ge 2$ seeds.

    Returns:
        Dictionary containing:
        - `mean_a`: mean metric for method A
        - `std_a`: sample standard deviation for method A ($ddof=1$)
        - `mean_b`: mean metric for method B
        - `std_b`: sample standard deviation for method B ($ddof=1$)
        - `mean_diff`: mean difference ($\text{mean}(A - B)$)
        - `t_stat`: paired Student's t-test statistic
        - `t_pvalue`: paired Student's t-test two-sided p-value
        - `wilcoxon_stat`: Wilcoxon signed-rank test statistic
        - `wilcoxon_pvalue`: Wilcoxon signed-rank test two-sided p-value

    Safe Fallback:
        If all differences $a_i - b_i == 0$, returns `t_stat=0.0`, `t_pvalue=1.0`,
        `wilcoxon_stat=0.0`, and `wilcoxon_pvalue=1.0` without raising SciPy exceptions.
    """
    arr_a = np.asarray(scores_a, dtype=np.float64)
    arr_b = np.asarray(scores_b, dtype=np.float64)

    if arr_a.ndim != 1 or arr_b.ndim != 1:
        raise ValueError(
            f"Expected 1D arrays, got shapes {arr_a.shape} and {arr_b.shape}."
        )
    if arr_a.size != arr_b.size:
        raise ValueError(
            f"Array sizes must match: got {arr_a.size} vs {arr_b.size}."
        )
    if arr_a.size < 2:
        raise ValueError(
            f"At least 2 paired samples are required for paired tests, got {arr_a.size}."
        )
    if not (np.all(np.isfinite(arr_a)) and np.all(np.isfinite(arr_b))):
        raise ValueError("Inputs must contain only finite numbers (no NaN or Inf).")

    mean_a = float(np.mean(arr_a))
    std_a = float(np.std(arr_a, ddof=1)) if arr_a.size > 1 else 0.0
    mean_b = float(np.mean(arr_b))
    std_b = float(np.std(arr_b, ddof=1)) if arr_b.size > 1 else 0.0
    diff = arr_a - arr_b
    mean_diff = float(np.mean(diff))

    # Check for zero difference case
    if np.allclose(diff, 0.0, atol=1e-12):
        return {
            "mean_a": mean_a,
            "std_a": std_a,
            "mean_b": mean_b,
            "std_b": std_b,
            "mean_diff": 0.0,
            "t_stat": 0.0,
            "t_pvalue": 1.0,
            "wilcoxon_stat": 0.0,
            "wilcoxon_pvalue": 1.0,
        }

    # Student's paired t-test
    t_res = ttest_rel(arr_a, arr_b)
    t_stat = float(t_res.statistic)
    t_pvalue = float(t_res.pvalue)

    # Wilcoxon signed-rank test
    # If all differences have same sign or zero differences exist
    try:
        w_res = wilcoxon(arr_a, arr_b, zero_method="wilcox")
        w_stat = float(w_res.statistic)
        w_pvalue = float(w_res.pvalue)
    except ValueError:
        # Fallback if wilcoxon fails on all zero differences
        w_stat = 0.0
        w_pvalue = 1.0

    return {
        "mean_a": mean_a,
        "std_a": std_a,
        "mean_b": mean_b,
        "std_b": std_b,
        "mean_diff": mean_diff,
        "t_stat": t_stat,
        "t_pvalue": t_pvalue,
        "wilcoxon_stat": w_stat,
        "wilcoxon_pvalue": w_pvalue,
    }
