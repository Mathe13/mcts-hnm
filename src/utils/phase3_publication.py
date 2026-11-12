"""Publication-only measured aggregations for the MCTS-HNM CIFAR-100 experiment.

No training, cached result rewriting, synthetic substitution, or test-based
checkpoint selection is performed here. Requires atomic REAL JSON provenance.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import json
import math

import numpy as np
from scipy import stats

from src.utils.experiment_config import ExecutionMode, validate_provenance_metadata


class MissingPublicationEvidence(ValueError):
    """Required measured per-seed data are unavailable."""


S1 = ('uniform', 'ohem', 'mcts_strict', 'mcts_enhanced')
S2 = ('uniform', 'uniform_grouped', 'mcts_coverage_balanced', 'mcts_enhanced')
RATIOS = ('1:10', '1:50', '1:100')


def load_campaign_runs(results_dir: str | Path, scenario: str) -> list[dict[str, Any]]:
    """Load only atomic runs explicitly listed by the corresponding suite output.

    Refuses mismatched IDs, provenance or benchmark identities. Never scans for
    unrelated historical runs, including Axis-2 same-strategy records.
    """
    root = Path(results_dir)
    expected = {'scenario1': 'tier1_axis1_samplers',
                'scenario2': 'tier1_scenario2_curriculum'}[scenario]
    report = root / (expected + '.json')
    if not report.is_file():
        raise MissingPublicationEvidence(f'Missing REAL suite result: {report}')
    validate_provenance_metadata(report.with_name(report.name + '.meta.json'),
                                 expected_mode=ExecutionMode.REAL)
    suite = json.loads(report.read_text(encoding='utf-8'))
    ids = [x.get('run_id') for x in suite.get('runs', [])]
    if not ids or any(not x for x in ids) or len(ids) != len(set(ids)):
        raise MissingPublicationEvidence(f'Missing/duplicate atomic run IDs in {report}')
    out: list[dict[str, Any]] = []
    for run_id in ids:
        fn = root / 'runs' / (run_id + '.json')
        validate_provenance_metadata(fn.with_name(fn.name + '.meta.json'),
                                     expected_mode=ExecutionMode.REAL)
        p = json.loads(fn.read_text(encoding='utf-8'))
        if (p.get('run_id') != run_id or p.get('scenario') != scenario
                or p.get('benchmark_name') != expected):
            raise ValueError(f'Atomic run identity/benchmark mismatch: {fn}')
        out.append(p)
    return out


def select_runs(runs: list[dict[str, Any]], *, scenario: str,
                ratio: str | None = None) -> dict[str, dict[int, dict[str, Any]]]:
    """Strict selection on scenario, Axis identity, target class, ratio, hierarchy.

    Return {strategy: {seed: run}} and reject ambiguous duplicates.
    """
    methods = S1 if scenario == 'scenario1' else S2
    expected = 'tier1_axis1_samplers' if scenario == 'scenario1' else 'tier1_scenario2_curriculum'
    by: dict[str, dict[int, dict[str, Any]]] = {s: {} for s in methods}
    for run in runs:
        if run.get('scenario') != scenario or run.get('benchmark_name') != expected:
            continue
        if scenario == 'scenario1' and (run.get('target_class') != 0 or run.get('imbalance_ratio') != ratio):
            continue
        strat = run.get('strategy')
        if strat not in by:
            continue
        if strat.startswith('mcts') and run.get('tree_type') != 'semantic':
            continue
        seed = int(run['seed'])
        if seed in by[strat]:
            raise ValueError(f'Duplicate {scenario}/{ratio}/{strat}/seed={seed}')
        by[strat][seed] = run
    return by


def paired_seeds(by: dict[str, dict[int, Any]], methods: tuple[str, ...],
                 minimum: int = 2, expected: int | None = None) -> list[int]:
    common = sorted(set.intersection(*(set(by[m]) for m in methods)))
    if len(common) < minimum or (expected is not None and len(common) != expected):
        target = f'exactly {expected}' if expected is not None else f'>= {minimum}'
        raise MissingPublicationEvidence(f'Need {target} paired seeds for {methods}, got {common}')
    return common


def aggregate_validation_history(runs: list[dict[str, Any]], *, ratio: str = '1:50') -> dict[str, Any]:
    """Aggregate measured validation histories on common (strategy, seed, step) keys."""
    by = select_runs(runs, scenario='scenario1', ratio=ratio)
    seeds = paired_seeds(by, S1, expected=5)
    recorded: dict[str, dict[int, dict[int, tuple[float, float]]]] = {}
    common_steps: set[int] | None = None
    for strategy in S1:
        recorded[strategy] = {}
        for seed in seeds:
            history = by[strategy][seed].get('history') or []
            vals: dict[int, tuple[float, float]] = {}
            for row in history:
                if 'step' not in row or 'f1' not in row or 'loss' not in row:
                    continue
                step = int(row['step'])
                if step in vals:
                    raise ValueError(f'Duplicate validation step for {strategy}/{seed}/{step}')
                f1, loss = float(row['f1']), float(row['loss'])
                if not (0 <= f1 <= 1 and loss >= 0 and math.isfinite(f1) and math.isfinite(loss)):
                    raise ValueError(f'Invalid validation measurement {strategy}/{seed}/{step}')
                vals[step] = (f1, loss)
            if len(vals) < 2:
                raise MissingPublicationEvidence(f'Incomplete validation history for {strategy}/{seed}')
            recorded[strategy][seed] = vals
            common_steps = set(vals) if common_steps is None else common_steps.intersection(vals)
    steps = sorted(common_steps or set())
    if len(steps) < 2:
        raise MissingPublicationEvidence('No >=2 shared validation steps across strategies/seeds')
    result: dict[str, Any] = {'steps': steps, 'imbalance_ratio': ratio,
                              'source_split': 'validation', 'n_paired_seeds': len(seeds),
                              'paired_seeds': seeds}
    for strat in S1:
        sample = np.asarray([[recorded[strat][sd][step] for step in steps] for sd in seeds], dtype=float)
        for i, key in enumerate(('f1', 'loss')):
            result[f'{strat}_{key}'] = sample[:, :, i].mean(axis=0).tolist()
            result[f'{strat}_{key}_sd'] = sample[:, :, i].std(axis=0, ddof=1).tolist()
    return result


def _run_test_f1_cost(run: dict[str, Any], protocol: str = 'last') -> tuple[float, float]:
    if protocol not in ('last', 'validation_selected'):
        raise ValueError(f'Unsupported test checkpoint protocol: {protocol}')
    metrics = run.get('last_metrics' if protocol == 'last' else 'best_validation_metrics')
    costs = run.get('computational_costs')
    if not isinstance(metrics, dict) or not isinstance(costs, dict):
        raise MissingPublicationEvidence(f'Missing last-checkpoint evidence: {run.get("run_id")}')
    f1 = float(metrics['f1'])
    seconds = float(costs['elapsed_seconds'])
    if not (math.isfinite(f1) and 0 <= f1 <= 1 and math.isfinite(seconds) and seconds > 0):
        raise ValueError(f'Invalid test F1 or elapsed_seconds in {run.get("run_id")}')
    return f1, seconds


def pareto_summary(runs: list[dict[str, Any]], *, ratio: str,
                   protocol: str = 'last') -> dict[str, dict[str, Any]]:
    """Measured last-checkpoint test F1 and walltime, SAME ratio and paired seeds."""
    by = select_runs(runs, scenario='scenario1', ratio=ratio)
    seeds = paired_seeds(by, S1, expected=5)
    out = {}
    for method in S1:
        arr = np.asarray([_run_test_f1_cost(by[method][sd], protocol=protocol) for sd in seeds], dtype=float)
        out[method] = {'f1_mean': float(arr[:, 0].mean()), 'f1_sd': float(arr[:, 0].std(ddof=1)),
                       'wall_mean': float(arr[:, 1].mean()), 'wall_sd': float(arr[:, 1].std(ddof=1)),
                       'seeds': seeds, 'protocol': protocol + '_checkpoint_test_f1'}
    return out


def plot_pareto_ratio(runs: list[dict[str, Any]], output_dir: str | Path,
                      *, ratio: str = '1:50', file_prefix: str = 'fig_pareto_cost_vs_accuracy',
                      protocol: str = 'last', watermark: str | None = None) -> dict[str, Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = pareto_summary(runs, ratio=ratio, protocol=protocol)
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    nondominated = []
    for m, s in summary.items():
        dominated = any((t['wall_mean'] <= s['wall_mean'] and t['f1_mean'] >= s['f1_mean']
                         and (t['wall_mean'] < s['wall_mean'] or t['f1_mean'] > s['f1_mean']))
                        for other, t in summary.items() if other != m)
        if not dominated:
            nondominated.append(m)
        ax.errorbar(s['wall_mean'], s['f1_mean'], xerr=s['wall_sd'], yerr=s['f1_sd'],
                    marker='o', capsize=3, linestyle='none', label=f'{m} (N={len(s["seeds"])})')
    nondominated.sort(key=lambda m: summary[m]['wall_mean'])
    if len(nondominated) >= 2:
        ax.plot([summary[m]['wall_mean'] for m in nondominated],
                [summary[m]['f1_mean'] for m in nondominated],
                color='gray', linestyle='--', alpha=.8, label='Nondominated means')
    protocol_label = ('last checkpoint' if protocol == 'last' else 'validation-selected checkpoint')
    ax.set(xlabel='Total measured wall-clock (s)',
           ylabel=f'{protocol_label.title()} TEST minority F1',
           title=f'Pareto, positive-retention {ratio}, {protocol_label} (mean ± SD)')
    ax.grid(alpha=.2)
    ax.legend(fontsize=7)
    if watermark:
        fig.text(.5, .5, watermark, ha='center', va='center', rotation=20, alpha=.15)
    fig.tight_layout()
    outdir = Path(output_dir); outdir.mkdir(parents=True, exist_ok=True)
    pdf, png = outdir / f'{file_prefix}.pdf', outdir / f'{file_prefix}.png'
    fig.savefig(pdf, bbox_inches='tight'); fig.savefig(png, dpi=220, bbox_inches='tight'); plt.close(fig)
    return {'pdf': pdf, 'png': png}


def exposure_concentration(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Cross-seed summaries of measured FINAL Scenario-2 exposure distributions.

    This is not a historical per-step series; it describes class selection
    concentration over the entire training run.
    """
    by = select_runs(runs, scenario='scenario2')
    seeds = paired_seeds(by, S2, expected=5)
    result: dict[str, dict[str, Any]] = {}
    for strat in S2:
        values = []
        for seed in seeds:
            counts = np.asarray(by[strat][seed].get('exposure_diagnostics', {}).get('class_exposure_counts'),
                                dtype=float)
            if counts.shape != (100,) or not np.isfinite(counts).all() or (counts < 0).any() or counts.sum() <= 0:
                raise MissingPublicationEvidence(f'Missing valid exposure counts {strat}/{seed}')
            p = counts / counts.sum()
            entropy = -np.sum(p[p > 0] * np.log(p[p > 0]))
            effective_k = float(np.exp(entropy))
            gini = float(np.abs(counts[:, None] - counts[None, :]).sum() / (2 * len(counts) * counts.sum()))
            values.append((effective_k, gini, float(counts.max() / counts.mean())))
        a = np.asarray(values, dtype=float)
        result[strat] = {'effective_classes_mean': float(a[:, 0].mean()),
                         'effective_classes_sd': float(a[:, 0].std(ddof=1)),
                         'gini_mean': float(a[:, 1].mean()),
                         'gini_sd': float(a[:, 1].std(ddof=1)),
                         'max_to_mean_mean': float(a[:, 2].mean()),
                         'n_seeds': len(seeds)}
    return result


def plot_exposure_concentration(runs: list[dict[str, Any]], output_dir: str | Path,
                                watermark: str | None = None) -> dict[str, Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m = exposure_concentration(runs)
    names = list(m)
    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    ax1.bar(x, [m[n]['effective_classes_mean'] for n in names],
            yerr=[m[n]['effective_classes_sd'] for n in names], capsize=3)
    ax1.set(ylabel='Effective number of classes (exp Shannon H)', ylim=(0, 105),
            title='Whole-run class exposure diversity')
    ax2.bar(x, [m[n]['gini_mean'] for n in names],
            yerr=[m[n]['gini_sd'] for n in names], capsize=3)
    ax2.set(ylabel='Gini coefficient of class exposure', ylim=(0, 1),
            title='Whole-run exposure inequality')
    for ax in (ax1, ax2):
        ax.set_xticks(x, names, rotation=20, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=.2)
    if watermark:
        fig.text(.5, .5, watermark, ha='center', va='center', rotation=20, alpha=.15)
    fig.tight_layout()
    outdir = Path(output_dir); outdir.mkdir(parents=True, exist_ok=True)
    pdf, png = outdir / 'fig_class_exposure_concentration.pdf', outdir / 'fig_class_exposure_concentration.png'
    fig.savefig(pdf, bbox_inches='tight'); fig.savefig(png, dpi=220, bbox_inches='tight'); plt.close(fig)
    return {'pdf': pdf, 'png': png}


def _holm(p_values: list[float]) -> list[float]:
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in p_values):
        raise ValueError('Holm input p-values must lie in [0, 1]')
    n = len(p_values)
    adjusted = [1.0] * n
    running = 0.0
    for j, idx in enumerate(sorted(range(n), key=lambda i: p_values[i])):
        running = max(running, (n-j) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def _effect(a: list[float], b: list[float]) -> dict[str, float]:
    """Pairing must be done by the same seed at the caller, never by array order."""
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if delta.ndim != 1 or len(delta) < 2 or not np.isfinite(delta).all():
        raise MissingPublicationEvidence('At least two finite paired differences required')
    n = len(delta)
    mean = float(delta.mean())
    sd = float(delta.std(ddof=1))
    if sd == 0:
        # A constant observed sample cannot justify claiming zero uncertainty.
        return {'effect': mean, 'ci_low': mean, 'ci_high': mean, 'p_raw': 1.0,
                'n': n, 'degenerate': True}
    tstat = mean / (sd / np.sqrt(n))
    p = float(2 * stats.t.sf(abs(tstat), df=n-1))
    margin = float(stats.t.ppf(.975, df=n-1) * sd / np.sqrt(n))
    return {'effect': mean, 'ci_low': mean-margin, 'ci_high': mean+margin,
            'p_raw': p, 'n': n, 'degenerate': False}


def paired_publication_effects(s1_runs: list[dict[str, Any]],
                               s2_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """12 prespecified comparisons: nine binary F1 and three 100-class Top-1.

    A single Holm family is applied across ALL available comparisons. If one
    of the three predeclared ratios is absent, no fabricated result is added.
    """
    records: list[dict[str, Any]] = []
    for ratio in RATIOS:
        by = select_runs(s1_runs, scenario='scenario1', ratio=ratio)
        if not any(by.values()):
            raise MissingPublicationEvidence(f'Missing prespecified positive-retention condition {ratio}')
        seeds = paired_seeds(by, S1, expected=5)
        for method in S1:
            if method == 'mcts_enhanced':
                continue
            a = [_run_test_f1_cost(by[method][sd])[0] for sd in seeds]
            b = [_run_test_f1_cost(by['mcts_enhanced'][sd])[0] for sd in seeds]
            records.append(dict(scenario='S1', ratio=ratio, method=method,
                                metric='Minority F1', **_effect(a, b)))
    by = select_runs(s2_runs, scenario='scenario2')
    if not any(by.values()):
        raise MissingPublicationEvidence('Missing prespecified Scenario-2 comparators')
    if any(by.values()):
        seeds = paired_seeds(by, S2, expected=5)
        for method in S2:
            if method == 'mcts_enhanced':
                continue
            def val(st, sd):
                r = by[st][sd]
                d = r.get('last_metrics') or r.get('final_metrics') or {}
                v = float(d['top1_accuracy'])
                if not 0 <= v <= 1 or not math.isfinite(v):
                    raise ValueError(f'Invalid top1_accuracy {st}/{sd}')
                return v
            a = [val(method, sd) for sd in seeds]
            b = [val('mcts_enhanced', sd) for sd in seeds]
            records.append(dict(scenario='S2', ratio='—', method=method,
                                metric='Top-1', **_effect(a, b)))
    if len(records) != 12:
        raise MissingPublicationEvidence(f'Incomplete prespecified 12-comparison inferential family: {len(records)}')
    corrected = _holm([x['p_raw'] for x in records])
    for r, value in zip(records, corrected):
        r['p_holm'] = value
        r['family_size'] = len(records)
    return records


def write_paired_effects_table(s1_runs: list[dict[str, Any]], s2_runs: list[dict[str, Any]],
                               output_dir: str | Path) -> Path:
    """Measured effect size, 95% paired-t CI and Holm-adjusted p, never inferred from aggregate SD."""
    rows = paired_publication_effects(s1_runs, s2_runs)
    lines = [
        '% Auto-generated from paired REAL atomic runs; not a superiority claim.',
        r'\begin{table*}[t]', r'\centering', r'\small',
        r'\caption{Prespecified paired comparisons against Enhanced MCTS-HNM. Effects are the',
        r'comparator minus Enhanced in percentage points; 95\% paired Student-$t$ confidence',
        r'intervals are descriptive with $N=5$ and Holm correction controls the family-wise',
        rf'error across {len(rows)} reported tests (both scenarios). Raw $p$-values in the primary tables are exploratory.}}',
        r'\label{tab:paired_effects}',
        r'\begin{tabular}{llllrrrr}', r'\toprule',
        r'Scenario & Setting & Comparator & Metric & $N$ & $\Delta$ (pp) [95\% CI] & Raw $p$ & Holm $p$ \\',
        r'\midrule',
    ]
    for r in rows:
        d, low, high = (100*r[k] for k in ('effect', 'ci_low', 'ci_high'))
        suffix = r'\textsuperscript{*}' if r['degenerate'] else ''
        ci = f'{d:+.1f} [{low:+.1f}, {high:+.1f}]{suffix}'
        label = {'uniform_grouped': 'Uniform Grouped',
                 'mcts_coverage_balanced': 'Coverage-Balanced MCTS',
                 'mcts_strict': 'Strict MCTS', 'uniform': 'Uniform',
                 'ohem': 'OHEM'}.get(r['method'], r['method'].replace('_', r'\_'))
        lines.append(f'{r["scenario"]} & {r["ratio"]} & {label} & {r["metric"]} & '
                     f'{r["n"]} & {ci} & {r["p_raw"]:.3g} & {r["p_holm"]:.3g} \\\\')
    if any(r['degenerate'] for r in rows):
        lines.append(r'\multicolumn{8}{l}{\footnotesize * Zero between-seed variance in observed paired differences; inferential $p$ set conservatively to 1.} \\')
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\end{table*}', '']
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / 'tab_paired_effects.tex'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path
