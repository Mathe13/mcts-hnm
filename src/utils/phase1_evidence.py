"""Recover existing measurements for MCTS-HNM publication artifacts (Phase 1).

No training is started by the plotters. Rebuilding a confusion matrix is an
explicit CLI operation over cached real or smoke embeddings, never a fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


class EvidenceUnavailable(ValueError):
    """Required measured and provenance-verified information is unavailable."""


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _confusion_path(paths: Any) -> Path:
    return paths.embeddings_dir / 'trees' / 'validation_confusion_matrix_100.npz'


def _checked_confusion(matrix: Any) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (100, 100) or not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError('Expected a finite nonnegative 100x100 measured validation confusion matrix')
    if np.any(arr.sum(axis=1) <= 0):
        raise ValueError('Validation confusion matrix contains an empty class row')
    return arr


def save_validation_confusion_matrix(matrix: Any, *, config: Any, repo_root: str | Path,
                                     source: str = 'notebook02:validation',
                                     cache_file: str | Path | None = None,
                                     reconstructed: bool = False,
                                     overwrite: bool = False) -> Path:
    """Persist measured confusion data and sidecar within the selected mode.

    `reconstructed` means refitted from the ORIGINAL cached embeddings, not a
    verbatim copy of a previously persisted confusion matrix.
    """
    from src.utils.experiment_config import get_isolated_paths, save_provenance_metadata
    paths = get_isolated_paths(config, repo_root=repo_root)
    dest = _confusion_path(paths)
    arr = _checked_confusion(matrix)
    if cache_file is not None and not Path(cache_file).is_file():
        raise FileNotFoundError(f'Embedding cache missing for provenance: {cache_file}')
    bio = io.BytesIO()
    np.savez_compressed(bio, confusion_matrix=arr)
    payload = bio.getvalue()
    if dest.exists():
        with np.load(dest, allow_pickle=False) as d:
            old = _checked_confusion(d['confusion_matrix'])
        if np.array_equal(old, arr):
            from src.utils.experiment_config import validate_provenance_metadata
            sidecar = dest.with_name(dest.name + '.meta.json')
            existing_meta = validate_provenance_metadata(sidecar, expected_mode=config.execution_mode)
            if (existing_meta.get('evidence_kind') != 'validation_confusion_matrix'
                    or existing_meta.get('matrix_sha256') != _hash_file(dest)):
                raise ValueError(f'Existing confusion matrix lacks matching evidence metadata: {dest}')
            return dest
        if not overwrite:
            raise FileExistsError(f'Existing confusion matrix differs: {dest}; explicit overwrite needed')
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix('.pending.npz')
    temp.write_bytes(payload)
    temp.replace(dest)
    extra = {'evidence_kind': 'validation_confusion_matrix', 'origin': source,
             'reconstructed_from_cached_embeddings': bool(reconstructed),
             'matrix_sha256': _hash_file(dest), 'num_classes': 100}
    if cache_file is not None:
        cp = Path(cache_file)
        if not cp.is_file():
            raise FileNotFoundError(f'Embedding cache missing for provenance: {cp}')
        extra['embedding_cache_sha256'] = _hash_file(cp)
    save_provenance_metadata(dest, config, extra=extra)
    return dest


def load_verified_validation_confusion(*, config: Any, repo_root: str | Path) -> tuple[np.ndarray, Path]:
    from src.utils.experiment_config import get_isolated_paths, validate_provenance_metadata
    paths = get_isolated_paths(config, repo_root=repo_root)
    dest = _confusion_path(paths)
    if not dest.is_file():
        raise EvidenceUnavailable(f'No persisted validation confusion matrix: {dest}. Run Notebook 02 save cell or explicit recovery command.')
    sidecar = dest.with_name(dest.name + '.meta.json')
    meta = validate_provenance_metadata(sidecar, expected_mode=config.execution_mode)
    if meta.get('evidence_kind') != 'validation_confusion_matrix' or meta.get('matrix_sha256') != _hash_file(dest):
        raise ValueError(f'Confusion matrix provenance/hash mismatch: {dest}')
    with np.load(dest, allow_pickle=False) as d:
        cm = _checked_confusion(d['confusion_matrix'])
    return cm, dest


def rebuild_confusion_explicitly(*, config: Any, repo_root: str | Path) -> Path:
    """Optional, deliberate baseline logistic refit from verified cached embeddings.

    Does NOT rerun any Tier-1 sampler or Tier-2 neural training.
    """
    from src.data.extract_embeddings import load_embedding_cache
    from src.trees.confusion_tree import estimate_validation_confusion_matrix
    from src.utils.experiment_config import get_isolated_paths, validate_provenance_metadata
    paths = get_isolated_paths(config, repo_root=repo_root)
    cache = paths.embeddings_dir / 'cifar100_resnet18.pt'
    validate_provenance_metadata(cache.with_name(cache.name + '.meta.json'), expected_mode=config.execution_mode)
    bundle = load_embedding_cache(cache)
    cm = estimate_validation_confusion_matrix(
        train_embeddings=bundle['train']['embeddings'],
        train_labels=bundle['train']['fine_labels'],
        val_embeddings=bundle['val']['embeddings'],
        val_labels=bundle['val']['fine_labels'],
        num_classes=100, seed=int(config.seed) if isinstance(config.seed, int) else 42,
    )
    return save_validation_confusion_matrix(cm, config=config, repo_root=repo_root,
         source='explicit_logistic_refit_from_cached_embeddings',
         cache_file=cache, reconstructed=True)


def _payloads(run_items: dict[str, dict[str, Any]], *, scenario: str) -> list[dict[str, Any]]:
    """Atomic paths are supplied by the provenance-validated notebook-05 loader."""
    out: list[dict[str, Any]] = []
    for run_id, info in run_items.items():
        if info.get('scenario') != scenario:
            continue
        fn = Path(info['run_file'])
        data = json.loads(fn.read_text(encoding='utf-8'))
        if (data.get('run_id') != run_id or data.get('scenario') != scenario or
            data.get('strategy') != info.get('strategy') or data.get('seed') != info.get('seed')):
            raise ValueError(f'Inconsistent atomic run identity: {fn}')
        out.append(data)
    return out


def measured_entropy_histories(run_items: dict[str, dict[str, Any]]) -> dict[str, dict[int, dict[int, float]]]:
    """Measured visitation entropy per strategy -> seed -> step -> value."""
    result: dict[str, dict[int, dict[int, float]]] = defaultdict(dict)
    for p in _payloads(run_items, scenario='scenario2'):
        strat = str(p['strategy'])
        seed = int(p['seed'])
        hist = {}
        for e in p.get('history', []):
            if not isinstance(e, dict) or 'normalized_entropy' not in e or 'step' not in e:
                continue
            val = float(e['normalized_entropy'])
            if not np.isfinite(val) or not 0 <= val <= 1:
                raise ValueError(f'Invalid normalized entropy in run {p["run_id"]}')
            hist[int(e['step'])] = val
        if hist:
            result[strat][seed] = hist
    return dict(result)


def plot_measured_entropy(run_items: dict[str, dict[str, Any]], output_dir: str | Path,
                          watermark: str | None = None) -> dict[str, Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curves = measured_entropy_histories(run_items)
    # At least two separately measured seeds per strategy for a dispersion band.
    curves = {k: v for k, v in curves.items() if len(v) >= 2}
    if not curves:
        raise EvidenceUnavailable('No per-step normalized visitation entropy for >=2 seeds in atomic Scenario-2 records')
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for strat, seed_curves in sorted(curves.items()):
        common_steps = sorted(set.intersection(*(set(v) for v in seed_curves.values())))
        if len(common_steps) < 2:
            continue
        data = np.array([[seed_curves[s][step] for step in common_steps]
                         for s in sorted(seed_curves)], dtype=np.float64)
        means = data.mean(axis=0)
        sd = data.std(axis=0, ddof=1)
        line, = ax.plot(common_steps, means, label=f'{strat} (N={len(seed_curves)})')
        ax.fill_between(common_steps, np.clip(means-sd, 0, 1), np.clip(means+sd, 0, 1),
                        alpha=.13, color=line.get_color())
    if not ax.lines:
        plt.close(fig)
        raise EvidenceUnavailable('No common measured entropy checkpoints between seeds')
    ax.set(xlabel='Training step', ylabel='Normalized MCTS visitation entropy',
           title='Measured visitation entropy (mean ± SD; Scenario 2)', ylim=(0, 1.04))
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    if watermark:
        fig.text(.5,.5,watermark,ha='center',va='center',alpha=.16,rotation=20)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    pdf, png = out/'fig_measured_visitation_entropy.pdf',out/'fig_measured_visitation_entropy.png'
    fig.tight_layout(); fig.savefig(pdf, bbox_inches='tight'); fig.savefig(png,dpi=220,bbox_inches='tight'); plt.close(fig)
    return {'pdf':pdf,'png':png}


def _validated_class_pair(p: dict[str, Any]) -> tuple[np.ndarray,np.ndarray]:
    counts = p.get('exposure_diagnostics',{}).get('class_exposure_counts')
    recalls = p.get('per_class_recall')
    if counts is None or recalls is None:
        raise EvidenceUnavailable('Atomic run lacks per-class exposure/recall; older runs cannot be treated as paired measurements')
    c,r=np.asarray(counts,dtype=float),np.asarray(recalls,dtype=float)
    if c.shape!=(100,) or r.shape!=(100,) or (c<0).any() or not np.isfinite(c).all() or not np.isfinite(r).all() or (r<0).any() or (r>1).any():
        raise ValueError(f'Invalid class-level arrays for atomic run {p.get("run_id")}')
    if c.sum()<=0:
        raise ValueError(f'Empty exposure count array for {p.get("run_id")}')
    return c,r


def measured_exposure_pairs(run_items: dict[str,dict[str,Any]], *, baseline: str='uniform',
                            candidate: str='mcts_enhanced') -> dict[str, Any]:
    """Require both methods under the SAME seeds and a unique atomic run per seed."""
    by: dict[str, dict[int, tuple[np.ndarray,np.ndarray]]] = {baseline:{},candidate:{}}
    for p in _payloads(run_items, scenario='scenario2'):
        if p['strategy'] not in by:
            continue
        if p['strategy'].startswith('mcts') and p.get('tree_type') != 'semantic':
            continue
        seed=int(p['seed']); strat=p['strategy']
        if seed in by[strat]:
            raise ValueError(f'Ambiguous duplicate strategy/seed for {strat}/{seed}')
        by[strat][seed]=_validated_class_pair(p)
    seeds=sorted(set(by[baseline]) & set(by[candidate]))
    if len(seeds)<2:
        raise EvidenceUnavailable('Per-seed paired recall and exposure not available for >=2 matching seeds')
    fractions=[]; deltas=[]
    for seed in seeds:
        c, r=by[candidate][seed]
        _, rb=by[baseline][seed]
        fractions.append(c/c.sum())
        deltas.append(r-rb)
    return {'seeds':seeds, 'mean_exposure':np.mean(fractions,axis=0),
            'mean_delta_recall':np.mean(deltas,axis=0),
            'std_delta_recall':np.std(deltas,axis=0,ddof=1)}


def plot_measured_exposure_recall(run_items: dict[str,dict[str,Any]], output_dir: str | Path,
                                   watermark: str | None = None) -> dict[str,Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=measured_exposure_pairs(run_items)
    fig,ax=plt.subplots(figsize=(7,4.7))
    ax.scatter(d['mean_exposure'],d['mean_delta_recall'],s=15,alpha=.7)
    ax.axhline(0,color='gray',linestyle='--',linewidth=1)
    ax.set(xlabel='Mean MCTS sampling exposure fraction per class',
           ylabel='Recall(MCTS) − Recall(Uniform)',
           title=f'Paired class-level differences (N={len(d["seeds"])} seeds)')
    ax.grid(alpha=.2)
    if watermark:
        fig.text(.5,.5,watermark,ha='center',va='center',alpha=.16,rotation=20)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    pdf,png=out/'fig_class_exposure_vs_delta_recall.pdf',out/'fig_class_exposure_vs_delta_recall.png'
    fig.tight_layout();fig.savefig(pdf,bbox_inches='tight');fig.savefig(png,dpi=220,bbox_inches='tight');plt.close(fig)
    return {'pdf':pdf,'png':png}


def extract_four_quadrants(last: dict[str, Any] | None,
                           best: dict[str, Any] | None) -> dict[str, float] | None:
    """Four *measured test* evaluations, not interpolated performance gains."""
    if not isinstance(last,dict) or not isinstance(best,dict):
        return None
    keys=(('q1_last_fixed',last,'f1_thresh05'),
          ('q2_last_calibrated',last,'f1'),
          ('q3_best_fixed',best,'f1_thresh05'),
          ('q4_best_calibrated',best,'f1'))
    if any(key not in d or d[key] is None for _,d,key in keys):
        return None
    ret={label:float(d[key]) for label,d,key in keys}
    if not all(np.isfinite(x) and 0<=x<=1 for x in ret.values()):
        raise ValueError('Invalid four-quadrant test metrics')
    return ret


def measured_quadrants(run_items: dict[str, dict[str, Any]]) -> dict[str,dict[str, Any]]:
    strategies=('uniform','ohem','mcts_strict','mcts_enhanced')
    by={s:{} for s in strategies}
    for p in _payloads(run_items, scenario='scenario1'):
        st=p['strategy'];sd=int(p['seed'])
        if st not in by:continue
        if st.startswith('mcts') and p.get('tree_type') != 'semantic':
            continue
        q=p.get('evaluation_quadrants') or extract_four_quadrants(p.get('last_metrics'), p.get('best_validation_metrics'))
        if q is None:continue
        if sd in by[st]:
            raise ValueError(f'Duplicate quadrants for strategy/seed {st}/{sd}')
        if set(q)!=set(('q1_last_fixed','q2_last_calibrated','q3_best_fixed','q4_best_calibrated')):
            raise ValueError('Incomplete four quadrant fields')
        vals=np.asarray([q[k] for k in ('q1_last_fixed','q2_last_calibrated',
                'q3_best_fixed','q4_best_calibrated')],dtype=float)
        if not np.isfinite(vals).all() or np.any((vals<0)|(vals>1)):
            raise ValueError(f'Invalid measured quadrant values for {st}/{sd}')
        by[st][sd]=q
    common=sorted(set.intersection(*(set(by[s]) for s in strategies)))
    if len(common)<2:
        raise EvidenceUnavailable('Four actual test quadrants per strategy/seed not available in historical atomic runs')
    return {s:{'seeds':common,'values':np.array([[by[s][seed][k] for k in
       ('q1_last_fixed','q2_last_calibrated','q3_best_fixed','q4_best_calibrated')] for seed in common])}
       for s in strategies}


def plot_measured_quadrants(run_items: dict[str,dict[str,Any]], output_dir: str | Path,
                            watermark: str | None = None) -> dict[str,Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=measured_quadrants(run_items)
    fig,ax=plt.subplots(figsize=(9.0,5.0))
    x=np.arange(4);width=.19
    for i,(name,one) in enumerate(d.items()):
        z=one['values'];ax.bar(x+(i-1.5)*width,z.mean(axis=0),width,
           yerr=z.std(axis=0,ddof=1),capsize=3,label=f'{name} (N={len(one["seeds"])})')
    ax.set_xticks(x,['Q1 Last fixed','Q2 Last cal.','Q3 Best fixed','Q4 Best cal.'])
    ax.set(ylabel='Test minority F1',title='Measured four evaluation protocols')
    ax.set_ylim(0,1);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    if watermark:fig.text(.5,.5,watermark,ha='center',va='center',alpha=.16,rotation=20)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    pdf,png=out/'fig_evaluation_4quadrants_comparison.pdf',out/'fig_evaluation_4quadrants_comparison.png'
    fig.tight_layout();fig.savefig(pdf,bbox_inches='tight');fig.savefig(png,dpi=220,bbox_inches='tight');plt.close(fig)
    return {'pdf':pdf,'png':png}


def inspect_existing_diagnostics(results_dir: str | Path) -> dict[str, Any]:
    """Report *availability* without silently upgrading aggregated snapshots to per-seed data."""
    p=Path(results_dir)/'tier1_scenario2_curriculum.json'
    if not p.is_file():return {'present':False,'reason':'Scenario-2 JSON not found'}
    d=json.loads(p.read_text(encoding='utf-8'))
    snap=d.get('diagnostics',{})
    return {'present':True,'strategies_with_final_snapshot':sorted(snap),
        'per_seed_diagnostics_available':any(isinstance(v,dict) and 'seed' in v for v in snap.values()),
        'note':'Snapshot lacks per-seed identifiers; DO NOT treat as paired multi-seed evidence.'}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['inspect','rebuild-confusion'])
    parser.add_argument('--mode',choices=['real','smoke'],default='real')
    parser.add_argument('--repo-root',default='.')
    args=parser.parse_args()
    from src.utils.experiment_config import ExperimentConfig,ExecutionMode,get_isolated_paths
    config=ExperimentConfig(execution_mode=ExecutionMode(args.mode))
    root=Path(args.repo_root).resolve()
    if args.command=='inspect':
        paths=get_isolated_paths(config,repo_root=root)
        print(json.dumps(inspect_existing_diagnostics(paths.results_dir),indent=2))
        try:
            _,p=load_verified_validation_confusion(config=config,repo_root=root)
            print(f'Verified confusion: {p}')
        except EvidenceUnavailable as exc:
            print(f'No verified confusion yet: {exc}')
    else:
        print('Explicitly refitting validation confusion from verified cached embeddings (no Tier-1 training).')
        print(rebuild_confusion_explicitly(config=config,repo_root=root))


if __name__=='__main__':
    main()
