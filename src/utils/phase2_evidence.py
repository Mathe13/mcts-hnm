"""Measured evidence for Phase 2 publication diagnostics.

No interpolation or synthetic fallback is allowed in REAL mode. Metrics are
computed from actual model predictions and observed training minibatches.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.utils.phase1_evidence import EvidenceUnavailable, _payloads


def capture_class_coverage(exposure_counts: Any) -> dict[str, float]:
    """Measure coverage for classes exposed >=1 and >=5 times, respectively."""
    counts = np.asarray(exposure_counts)
    if counts.ndim != 1 or counts.size < 2 or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError('Coverage counts must be a 1D integer vector of classes')
    if np.any(counts < 0):
        raise ValueError('Exposure counts must be nonnegative')
    return {
        'class_coverage_k1': float(np.mean(counts >= 1)),
        'class_coverage_k5': float(np.mean(counts >= 5)),
    }


def measure_binary_pr(model: Any, features: Any, labels: Any, *, device: Any = None,
                      batch_size: int = 512) -> dict[str, Any]:
    """Persist the *exact measured PR curve* at the final test checkpoint.

    This is evaluated after model selection is completed; do not use these data
    for tuning or selecting checkpoints. AUPRC below denotes Average Precision,
    not a trapezoidal integral of plotted points.
    """
    import torch
    import torch.nn as nn
    if not isinstance(model, nn.Module) or not isinstance(features, torch.Tensor):
        raise TypeError('Expected a torch model and tensor features')
    y = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    y = np.asarray(y, dtype=np.int64)
    if y.ndim != 1 or len(y) != len(features) or not np.isin(y, [0, 1]).all() or not np.any(y == 1) or not np.any(y == 0):
        raise ValueError('Measured test PR requires aligned labels with positive and negative examples')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    device = device if device is not None else next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        chunks = []
        with torch.no_grad():
            for k in range(0, len(features), batch_size):
                output = model(features[k:k + batch_size].to(device)).reshape(-1).to(dtype=torch.float64)
                chunks.append(torch.sigmoid(output).detach().cpu().numpy())
        scores = np.concatenate(chunks).astype(np.float64)
    finally:
        model.train(was_training)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Invalid predicted probabilities')
    p, r, _ = precision_recall_curve(y, scores)
    ap = float(average_precision_score(y, scores))
    return {
        'split': 'test', 'checkpoint': 'final', 'metric': 'average_precision',
        'precision': p.astype(float).tolist(), 'recall': r.astype(float).tolist(),
        'average_precision': ap, 'n_test': int(y.size),
        'n_positive': int(np.sum(y == 1)), 'n_negative': int(np.sum(y == 0)),
    }


def _validated_pr(p: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    pr = p.get('binary_test_pr')
    if not isinstance(pr, dict):
        raise EvidenceUnavailable('Atomic run lacks measured test precision-recall points')
    if pr.get('split') != 'test' or pr.get('checkpoint') != 'final' or pr.get('metric') != 'average_precision':
        raise ValueError('PR evidence has unknown split, checkpoint or metric')
    pre, rec = np.asarray(pr.get('precision'), dtype=np.float64), np.asarray(pr.get('recall'), dtype=np.float64)
    ap = float(pr.get('average_precision', np.nan))
    if (pre.ndim != 1 or rec.ndim != 1 or pre.size < 2 or pre.size != rec.size
            or not np.isfinite(pre).all() or not np.isfinite(rec).all()
            or np.any((pre < 0) | (pre > 1)) or np.any((rec < 0) | (rec > 1))
            or not np.isfinite(ap) or not 0 <= ap <= 1
            or np.any(np.diff(rec) > 1e-10)):
        raise ValueError(f'Invalid measured PR curve in {p.get("run_id")}')
    if pr.get('n_positive', 0) < 1 or pr.get('n_negative', 0) < 1:
        raise ValueError('PR curve must include positive and negative samples')
    observed = p.get('final_metrics', {}).get('auprc')
    if observed is not None and abs(float(observed) - ap) > 2e-5:
        raise ValueError('Measured PR average precision inconsistent with final_metrics.auprc')
    return pre, rec, ap


def _atomic_s1(run_items: dict[str, dict[str, Any]], *, target_class: int = 0,
               imbalance_ratio: str = '1:50') -> list[dict[str, Any]]:
    """Extract only the canonical Axis-1 class/ratio, excluding other experiments."""
    runs = []
    for p in _payloads(run_items, scenario='scenario1'):
        if p.get('target_class') != target_class or p.get('imbalance_ratio') != imbalance_ratio:
            continue
        if p['strategy'].startswith('mcts') and p.get('tree_type') != 'semantic':
            continue
        if p.get('benchmark_name') not in (None, 'tier1_axis1_samplers'):
            continue
        runs.append(p)
    return runs


def measured_pr_by_seed(run_items: dict[str, dict[str, Any]], *,
                        target_class: int = 0, imbalance_ratio: str = '1:50') -> dict[str, dict[int, tuple[np.ndarray, np.ndarray, float]]]:
    """Group exact per-seed PR measurements; reject duplicates, missing seeds and bad points."""
    names = ('uniform', 'ohem', 'mcts_strict', 'mcts_enhanced')
    by: dict[str, dict[int, tuple[np.ndarray, np.ndarray, float]]] = {k: {} for k in names}
    for run in _atomic_s1(run_items, target_class=target_class, imbalance_ratio=imbalance_ratio):
        strat = str(run['strategy'])
        if strat not in by:
            continue
        seed = int(run['seed'])
        if seed in by[strat]:
            raise ValueError(f'Duplicate canonical Axis-1 PR run: {strat}/{seed}')
        by[strat][seed] = _validated_pr(run)
    shared = sorted(set.intersection(*(set(v) for v in by.values())))
    if len(shared) < 2:
        raise EvidenceUnavailable('At least two paired measured PR curves per canonical strategy required')
    for name in names:
        by[name] = {seed: by[name][seed] for seed in shared}
    return by


def _interp_pr(pre: np.ndarray, rec: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate the measured PR staircase on a common recall grid.

    Duplicated recall positions are collapsed by retaining the maximum observed
    precision at each recall. The plotted interpolation is descriptive only;
    reported AP remains the original scikit-learn Average Precision.
    """
    asc_r, asc_p = rec[::-1], pre[::-1]
    unique, first, counts = np.unique(asc_r, return_index=True, return_counts=True)
    v = np.maximum.reduceat(asc_p, first)
    return np.interp(grid, unique, v, left=v[0], right=v[-1])


def plot_measured_pr(run_items: dict[str, dict[str, Any]], output_dir: str | Path,
                     watermark: str | None = None) -> dict[str, Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs = measured_pr_by_seed(run_items)
    grid = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for name, seeds in runs.items():
        mat = np.array([_interp_pr(p, r, grid) for p, r, _ in seeds.values()], dtype=float)
        mean, sd = mat.mean(axis=0), mat.std(axis=0, ddof=1)
        ap = np.mean([a for _, _, a in seeds.values()])
        line, = ax.plot(grid, mean, label=f'{name} (AP {ap:.3f}, N={len(seeds)})')
        ax.fill_between(grid, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1),
                        alpha=.16, color=line.get_color())
    ax.set(xlabel='Recall (test)', ylabel='Precision (test)', xlim=(0, 1), ylim=(0, 1.02),
           title='Measured test precision-recall (mean ± SD across paired seeds)')
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    if watermark:
        fig.text(.5, .5, watermark, ha='center', va='center', rotation=20, alpha=.16)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    pdf, png = out / 'fig_precision_recall_curves.pdf', out / 'fig_precision_recall_curves.png'
    fig.tight_layout(); fig.savefig(pdf, bbox_inches='tight'); fig.savefig(png, dpi=220, bbox_inches='tight'); plt.close(fig)
    return {'pdf': pdf, 'png': png}


def coverage_curves(run_items: dict[str, dict[str, Any]]) -> dict[str, dict[int, dict[int, tuple[float, float, float | None]]]]:
    """Return measured per-step coverage with optional MCTS visitation entropy."""
    by: dict[str, dict[int, dict[int, tuple[float, float, float | None]]]] = defaultdict(dict)
    for run in _payloads(run_items, scenario='scenario2'):
        strat, seed = str(run['strategy']), int(run['seed'])
        if seed in by[strat]:
            raise ValueError(f'Duplicate scenario2 run for {strat}/{seed}')
        steps = {}
        for row in run.get('history', []):
            if 'class_coverage_k1' not in row or 'class_coverage_k5' not in row:
                continue
            step = int(row['step'])
            k1, k5 = float(row['class_coverage_k1']), float(row['class_coverage_k5'])
            ent = row.get('normalized_entropy')
            e = float(ent) if ent is not None else None
            if (not np.isfinite([k1, k5]).all() or k5 > k1 or k5 < 0 or k1 > 1 or k1 < 0
                    or (e is not None and (not np.isfinite(e) or e < 0 or e > 1))):
                raise ValueError(f'Invalid coverage measurement in {run["run_id"]}/{step}')
            steps[step] = k1, k5, e
        if steps:
            ordered = [steps[i] for i in sorted(steps)]
            if any(ordered[i+1][0] + 1e-10 < ordered[i][0] or ordered[i+1][1] + 1e-10 < ordered[i][1] for i in range(len(ordered)-1)):
                raise ValueError(f'Nonmonotonic cumulative class coverage in {run["run_id"]}')
            by[strat][seed] = steps
    return dict(by)


def plot_coverage_entropy(run_items: dict[str, dict[str, Any]], output_dir: str | Path,
                          watermark: str | None = None) -> dict[str, Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curves = {s: data for s, data in coverage_curves(run_items).items() if len(data) >= 2}
    if not curves:
        raise EvidenceUnavailable('Per-step class coverage histories for >=2 seeds not available; rerun Scenario 2')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for strat, seeds in sorted(curves.items()):
        common = sorted(set.intersection(*(set(x) for x in seeds.values())))
        if len(common) < 2:
            continue
        coverage = np.asarray([[seeds[s][k][0] for k in common] for s in sorted(seeds)], dtype=float)
        coverage5 = np.asarray([[seeds[s][k][1] for k in common] for s in sorted(seeds)], dtype=float)
        one, = ax2.plot(common, 100 * coverage.mean(axis=0), label=f'{strat}: >=1 (N={len(seeds)})')
        ax2.fill_between(common, 100 * np.clip(coverage.mean(axis=0) - coverage.std(axis=0, ddof=1), 0, 1),
                         100 * np.clip(coverage.mean(axis=0) + coverage.std(axis=0, ddof=1), 0, 1), alpha=.1, color=one.get_color())
        ax2.plot(common, 100 * coverage5.mean(axis=0), linestyle='--', color=one.get_color(), alpha=.9,
                 label=f'{strat}: >=5')
        ent_seeds = [s for s in sorted(seeds) if all(seeds[s][k][2] is not None for k in common)]
        if len(ent_seeds) >= 2:
            ent = np.asarray([[seeds[s][k][2] for k in common] for s in ent_seeds], dtype=float)
            line, = ax1.plot(common, ent.mean(axis=0), label=f'{strat} (N={len(ent_seeds)})')
            ax1.fill_between(common, np.clip(ent.mean(axis=0)-ent.std(axis=0, ddof=1), 0, 1),
                             np.clip(ent.mean(axis=0)+ent.std(axis=0, ddof=1), 0, 1), alpha=.12, color=line.get_color())
    if not ax2.lines:
        plt.close(fig)
        raise EvidenceUnavailable('At least two common checkpoints per strategy are required')
    ax1.set(xlabel='Training step', ylabel='Normalized MCTS visitation entropy', ylim=(0, 1.04),
            title='(a) Measured tree visitation entropy')
    ax2.set(xlabel='Training step', ylabel='Cumulative class coverage (%)', ylim=(0, 104),
            title='(b) Class coverage (>=1 and >=5 samples)')
    if ax1.lines: ax1.legend(fontsize=7)
    ax2.legend(fontsize=7, loc='lower right')
    for ax in (ax1, ax2): ax.grid(alpha=.2)
    if watermark:
        fig.text(.5,.5,watermark,ha='center',va='center',rotation=20,alpha=.16)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    pdf,png=out/'fig_coverage_entropy_dynamics.pdf',out/'fig_coverage_entropy_dynamics.png'
    fig.tight_layout();fig.savefig(pdf,bbox_inches='tight');fig.savefig(png,dpi=220,bbox_inches='tight');plt.close(fig)
    return {'pdf': pdf, 'png': png}
