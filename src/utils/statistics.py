"""Statistical hypothesis tests for multi-seed experimental evaluations."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import t, ttest_1samp, ttest_rel, wilcoxon


def compute_paired_statistical_tests(
    scores_a: np.ndarray | list[float],
    scores_b: np.ndarray | list[float],
    keys_a: list[Any] | None = None,
    keys_b: list[Any] | None = None,
) -> dict[str, Any]:
    r"""Compute paired Student's t-test and Wilcoxon signed-rank test across paired score samples."""
    return compute_paired_method_differences(
        scores_a=scores_a,
        scores_b=scores_b,
        keys_a=keys_a,
        keys_b=keys_b,
    )


def compute_paired_method_differences(
    scores_a: np.ndarray | list[float],
    scores_b: np.ndarray | list[float],
    keys_a: list[Any] | None = None,
    keys_b: list[Any] | None = None,
) -> dict[str, Any]:
    r"""Compute paired differences, confidence intervals, Cohen's d, and significance tests.

    Args:
        scores_a: 1D numeric array or list of metrics from method A across $N \ge 2$ seeds.
        scores_b: 1D numeric array or list of metrics from method B across $N \ge 2$ seeds.
        keys_a: Optional pairing keys (e.g. tuples of seed, class, split, budget) for method A.
        keys_b: Optional pairing keys for method B.

    Returns:
        Dictionary containing statistical test metrics and effect sizes.

    Raises:
        ValueError: If array sizes mismatch, have fewer than 2 samples, or pairing keys mismatch.
    """
    if keys_a is not None or keys_b is not None:
        if keys_a is None or keys_b is None:
            raise ValueError("Both keys_a and keys_b must be provided when pairing validation is requested.")
        if len(keys_a) != len(keys_b):
            raise ValueError(f"keys_a length ({len(keys_a)}) does not match keys_b length ({len(keys_b)}).")
        if keys_a != keys_b:
            raise ValueError(
                f"Paired statistical tests require identical matching pairing keys, but keys differ: {keys_a} != {keys_b}."
            )

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

    n = int(arr_a.size)
    mean_a = float(np.mean(arr_a))
    std_a = float(np.std(arr_a, ddof=1)) if n > 1 else 0.0
    mean_b = float(np.mean(arr_b))
    std_b = float(np.std(arr_b, ddof=1)) if n > 1 else 0.0
    diff = arr_a - arr_b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    se_diff = float(std_diff / np.sqrt(n)) if n > 0 else 0.0

    # Cohen's d for paired samples: mean_diff / std_diff
    cohens_d = float(mean_diff / std_diff) if std_diff > 1e-12 else 0.0

    # 95% Confidence Interval
    t_crit = float(t.ppf(0.975, df=n - 1)) if n > 1 else 0.0
    ci_lower = float(mean_diff - t_crit * se_diff)
    ci_upper = float(mean_diff + t_crit * se_diff)

    # Check for zero difference case
    if np.allclose(diff, 0.0, atol=1e-12):
        return {
            "n": n,
            "mean_a": mean_a,
            "std_a": std_a,
            "mean_b": mean_b,
            "std_b": std_b,
            "mean_diff": 0.0,
            "std_diff": 0.0,
            "se_diff": 0.0,
            "cohens_d": 0.0,
            "ci_95": (0.0, 0.0),
            "ci_95_lower": 0.0,
            "ci_95_upper": 0.0,
            "t_stat": 0.0,
            "t_pvalue": 1.0,
            "wilcoxon_stat": 0.0,
            "wilcoxon_pvalue": 1.0,
            "paired_differences": list(diff.tolist()),
        }

    # Student's paired t-test
    t_res = ttest_rel(arr_a, arr_b)
    t_stat = float(t_res.statistic)
    t_pvalue = float(t_res.pvalue)

    # Wilcoxon signed-rank test
    try:
        w_res = wilcoxon(arr_a, arr_b, zero_method="wilcox")
        w_stat = float(w_res.statistic)
        w_pvalue = float(w_res.pvalue)
    except ValueError:
        w_stat = 0.0
        w_pvalue = 1.0

    return {
        "n": n,
        "mean_a": mean_a,
        "std_a": std_a,
        "mean_b": mean_b,
        "std_b": std_b,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "se_diff": se_diff,
        "cohens_d": cohens_d,
        "ci_95": (ci_lower, ci_upper),
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "t_stat": t_stat,
        "t_pvalue": t_pvalue,
        "wilcoxon_stat": w_stat,
        "wilcoxon_pvalue": w_pvalue,
        "paired_differences": list(diff.tolist()),
    }


def compute_method_by_ratio_interaction(
    scores_r1_a: np.ndarray | list[float],
    scores_r1_b: np.ndarray | list[float],
    scores_r2_a: np.ndarray | list[float],
    scores_r2_b: np.ndarray | list[float],
) -> dict[str, Any]:
    r"""Compute interaction effect statistics between method (A vs B) and imbalance ratio ($r_1$ vs $r_2$).

    Tests whether the paired advantage $\Delta = A - B$ differs across ratios:
        $$\Delta_s = (A(s, r_2) - B(s, r_2)) - (A(s, r_1) - B(s, r_1))$$

    Args:
        scores_r1_a: Metrics for method A at ratio 1 across seeds.
        scores_r1_b: Metrics for method B at ratio 1 across identical seeds.
        scores_r2_a: Metrics for method A at ratio 2 across identical seeds.
        scores_r2_b: Metrics for method B at ratio 2 across identical seeds.

    Returns:
        Dictionary containing interaction effect sizes and hypothesis tests.
    """
    arr_r1_a = np.asarray(scores_r1_a, dtype=np.float64)
    arr_r1_b = np.asarray(scores_r1_b, dtype=np.float64)
    arr_r2_a = np.asarray(scores_r2_a, dtype=np.float64)
    arr_r2_b = np.asarray(scores_r2_b, dtype=np.float64)

    for arr, name in [
        (arr_r1_a, "scores_r1_a"),
        (arr_r1_b, "scores_r1_b"),
        (arr_r2_a, "scores_r2_a"),
        (arr_r2_b, "scores_r2_b"),
    ]:
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1D, got shape {arr.shape}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values (NaN or Inf).")

    n = int(arr_r1_a.size)
    if not (arr_r1_b.size == n and arr_r2_a.size == n and arr_r2_b.size == n):
        raise ValueError(
            f"All score arrays must have matching size, got {n}, {arr_r1_b.size}, {arr_r2_a.size}, {arr_r2_b.size}."
        )
    if n < 2:
        raise ValueError(f"At least 2 paired seeds are required, got {n}.")

    d1 = arr_r1_a - arr_r1_b
    d2 = arr_r2_a - arr_r2_b
    interaction = d2 - d1

    mean_diff_r1 = float(np.mean(d1))
    mean_diff_r2 = float(np.mean(d2))
    mean_interaction = float(np.mean(interaction))
    std_interaction = float(np.std(interaction, ddof=1)) if n > 1 else 0.0
    se_interaction = float(std_interaction / np.sqrt(n)) if n > 0 else 0.0

    cohens_d = float(mean_interaction / std_interaction) if std_interaction > 1e-12 else 0.0

    t_crit = float(t.ppf(0.975, df=n - 1)) if n > 1 else 0.0
    ci_lower = float(mean_interaction - t_crit * se_interaction)
    ci_upper = float(mean_interaction + t_crit * se_interaction)

    if np.allclose(interaction, 0.0, atol=1e-12):
        t_stat = 0.0
        t_pvalue = 1.0
    else:
        t_res = ttest_1samp(interaction, popmean=0.0)
        t_stat = float(t_res.statistic)
        t_pvalue = float(t_res.pvalue)

    return {
        "n": n,
        "mean_diff_r1": mean_diff_r1,
        "mean_diff_r2": mean_diff_r2,
        "mean_interaction": mean_interaction,
        "std_interaction": std_interaction,
        "se_interaction": se_interaction,
        "cohens_d": cohens_d,
        "ci_95": (ci_lower, ci_upper),
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "t_stat": t_stat,
        "t_pvalue": t_pvalue,
        "interaction_differences": list(interaction.tolist()),
    }


def _summarize_effect_series(series: np.ndarray) -> dict[str, Any]:
    """Helper to summarize a 1D effect difference series with statistics and tests."""
    arr = np.asarray(series, dtype=np.float64)
    n = int(arr.size)
    mean_val = float(np.mean(arr)) if n > 0 else 0.0
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se_val = float(std_val / np.sqrt(n)) if n > 0 else 0.0
    cohens_d = float(mean_val / std_val) if std_val > 1e-12 else 0.0

    t_crit = float(t.ppf(0.975, df=n - 1)) if n > 1 else 0.0
    ci_lower = float(mean_val - t_crit * se_val)
    ci_upper = float(mean_val + t_crit * se_val)

    if n >= 2 and not np.allclose(arr, 0.0, atol=1e-12):
        t_res = ttest_1samp(arr, popmean=0.0)
        t_stat = float(t_res.statistic)
        t_pval = float(t_res.pvalue)
    else:
        t_stat = 0.0
        t_pval = 1.0

    return {
        "n": n,
        "mean": mean_val,
        "std": std_val,
        "se": se_val,
        "cohens_d": cohens_d,
        "ci_95": (ci_lower, ci_upper),
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "t_stat": t_stat,
        "t_pvalue": t_pval,
        "per_seed": [float(x) for x in arr],
    }


def compute_2x2_factorial_effects(
    scores_strict: np.ndarray | list[float] | dict[str, Any],
    scores_qnorm: np.ndarray | list[float] | None = None,
    scores_dualstream: np.ndarray | list[float] | None = None,
    scores_enhanced: np.ndarray | list[float] | None = None,
) -> dict[str, Any]:
    r"""Compute 2x2 factorial mechanism decomposition across paired seeds.

    Evaluates:
    - Main Effect of Q-normalization (ME_Q):
      $$\text{ME}_Q = \frac{1}{2} [(S_{10} + S_{11}) - (S_{00} + S_{01})]$$
    - Main Effect of Dual-Stream (ME_DS):
      $$\text{ME}_{\text{DS}} = \frac{1}{2} [(S_{01} + S_{11}) - (S_{00} + S_{10})]$$
    - Interaction Term (IE):
      $$\text{IE} = (S_{11} - S_{10}) - (S_{01} - S_{00})$$

    Args:
        scores_strict: 1D array or dictionary containing variant keys.
        scores_qnorm: 1D array of scores for mcts_qnorm_only across seeds.
        scores_dualstream: 1D array of scores for mcts_dualstream_only across seeds.
        scores_enhanced: 1D array of scores for mcts_enhanced across seeds.

    Returns:
        Dictionary containing summarized effect sizes, p-values, and variant means.
    """
    if isinstance(scores_strict, dict):
        d = scores_strict
        s00_raw = d.get("mcts_strict", d.get("strict"))
        s10_raw = d.get("mcts_qnorm_only", d.get("qnorm_only", d.get("qnorm")))
        s01_raw = d.get("mcts_dualstream_only", d.get("dualstream_only", d.get("dualstream")))
        s11_raw = d.get("mcts_enhanced", d.get("enhanced"))
        if s00_raw is None or s10_raw is None or s01_raw is None or s11_raw is None:
            raise ValueError(
                "When passing a dictionary, it must contain keys for 'mcts_strict', "
                "'mcts_qnorm_only', 'mcts_dualstream_only', and 'mcts_enhanced'."
            )
        s00 = np.asarray(s00_raw, dtype=np.float64)
        s10 = np.asarray(s10_raw, dtype=np.float64)
        s01 = np.asarray(s01_raw, dtype=np.float64)
        s11 = np.asarray(s11_raw, dtype=np.float64)
    else:
        if scores_qnorm is None or scores_dualstream is None or scores_enhanced is None:
            raise ValueError("All 4 variant score arrays must be provided.")
        s00 = np.asarray(scores_strict, dtype=np.float64)
        s10 = np.asarray(scores_qnorm, dtype=np.float64)
        s01 = np.asarray(scores_dualstream, dtype=np.float64)
        s11 = np.asarray(scores_enhanced, dtype=np.float64)

    for arr, name in [
        (s00, "mcts_strict"),
        (s10, "mcts_qnorm_only"),
        (s01, "mcts_dualstream_only"),
        (s11, "mcts_enhanced"),
    ]:
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1D, got shape {arr.shape}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values (NaN or Inf).")

    n = int(s00.size)
    if not (s10.size == n and s01.size == n and s11.size == n):
        raise ValueError(
            f"All score arrays must have matching size, got {n}, {s10.size}, {s01.size}, {s11.size}."
        )
    if n < 1:
        raise ValueError("At least 1 paired observation is required.")

    # Main effect of Q-normalization: 0.5 * ((s10 + s11) - (s00 + s01))
    me_q = 0.5 * ((s10 + s11) - (s00 + s01))

    # Main effect of Dual-Stream: 0.5 * ((s01 + s11) - (s00 + s10))
    me_ds = 0.5 * ((s01 + s11) - (s00 + s10))

    # Interaction term: (s11 - s10) - (s01 - s00)
    interaction = (s11 - s10) - (s01 - s00)

    return {
        "n": n,
        "main_effect_qnorm": _summarize_effect_series(me_q),
        "main_effect_dualstream": _summarize_effect_series(me_ds),
        "interaction_effect": _summarize_effect_series(interaction),
        "variant_means": {
            "mcts_strict": float(np.mean(s00)),
            "mcts_qnorm_only": float(np.mean(s10)),
            "mcts_dualstream_only": float(np.mean(s01)),
            "mcts_enhanced": float(np.mean(s11)),
        },
    }

