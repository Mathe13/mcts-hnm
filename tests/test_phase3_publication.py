"""Phase 3 publication integrity: purely measured post-run analysis tests."""
from __future__ import annotations
import json
import numpy as np
import pytest

from src.utils.phase3_publication import (
    MissingPublicationEvidence, RATIOS, S1, S2,
    _holm, _effect, select_runs, paired_seeds, aggregate_validation_history,
    pareto_summary, exposure_concentration, paired_publication_effects,
    plot_pareto_ratio, plot_exposure_concentration, write_paired_effects_table,
)


def make_run(scenario: str, strat: str, seed: int, ratio: str | None = None,
             f1: float = .6, top1: float = .5) -> dict:
    expected = 'tier1_axis1_samplers' if scenario == 'scenario1' else 'tier1_scenario2_curriculum'
    run = {'run_id': f'{expected}_{ratio}_{strat}_{seed}', 'benchmark_name': expected,
           'scenario': scenario, 'seed': seed, 'target_class': 0 if ratio else None,
           'imbalance_ratio': ratio, 'strategy': strat,
           'tree_type': 'semantic' if strat.startswith('mcts') else 'none',
           'last_metrics': {'f1': f1, 'top1_accuracy': top1},
           'best_validation_metrics': {'f1': min(1.0, f1+.05)},
           'computational_costs': {'elapsed_seconds': 9+seed/100 + len(strat)/10},
           'history': [dict(step=k, f1=f1-.1 + k*.001, loss=1-f1+.01-k*.001)
                       for k in (10,20,30)]}
    if scenario == 'scenario2':
        counts = np.full(100, 10, dtype=int)
        counts[0] = 50 if strat == 'mcts_enhanced' else 10
        run['exposure_diagnostics'] = {'class_exposure_counts': counts.tolist()}
    return run


def campaign():
    s1 = [make_run('scenario1', strat, seed, ratio, .55+.03*i+.01*j-.003*(seed-42))
          for j,ratio in enumerate(RATIOS) for i,strat in enumerate(S1) for seed in (42,43,44,45,46)]
    s2 = [make_run('scenario2', strat, seed, top1=.30+.08*i+.001*(seed-42))
          for i,strat in enumerate(S2) for seed in (42,43,44,45,46)]
    return s1, s2


def test_convergence_is_paired_mean_sd_validation_and_has_correct_ratio():
    s1,_ = campaign()
    d = aggregate_validation_history(s1, ratio='1:50')
    assert d['imbalance_ratio'] == '1:50'
    assert d['source_split'] == 'validation'
    assert d['paired_seeds'] == [42,43,44,45,46]
    assert d['steps'] == [10,20,30]
    assert len(d['uniform_f1_sd']) == 3
    assert all(x > 0 for x in d['uniform_f1_sd'])
    # Historical accidental first-seed selection is not mean aggregation.
    first = next(x for x in s1 if x['imbalance_ratio']=='1:50' and x['strategy']=='uniform')
    assert d['uniform_f1'][0] != first['history'][0]['f1']


def test_convergence_refuses_missing_seed_and_duplicate_identity():
    s1,_ = campaign()
    with pytest.raises(ValueError, match='Duplicate'):
        aggregate_validation_history(s1 + [s1[0]], ratio='1:10')
    reduced = [x for x in s1 if not (x['imbalance_ratio']=='1:50' and
                x['strategy']=='ohem' and x['seed']!=42)]
    with pytest.raises(MissingPublicationEvidence):
        # Force >=2 common paired seeds, so this must not plot a one-seed band.
        aggregate_validation_history(reduced, ratio='1:50')


def test_pareto_strict_ratio_last_checkpoint_and_walltime(tmp_path):
    s1,_ = campaign()
    d = pareto_summary(s1, ratio='1:50')
    strict = [r for r in s1 if r['imbalance_ratio']=='1:50' and r['strategy']=='mcts_strict']
    assert d['mcts_strict']['f1_mean'] == pytest.approx(np.mean([r['last_metrics']['f1'] for r in strict]))
    assert d['mcts_strict']['f1_mean'] != pytest.approx(
        np.mean([r['best_validation_metrics']['f1'] for r in strict]))
    assert d['mcts_strict']['protocol'] == 'last_checkpoint_test_f1'
    best=pareto_summary(s1, ratio='1:50',protocol='validation_selected')
    assert best['mcts_strict']['protocol']=='validation_selected_checkpoint_test_f1'
    assert best['mcts_strict']['f1_mean'] > d['mcts_strict']['f1_mean']
    out=plot_pareto_ratio(s1,tmp_path,ratio='1:50')
    assert out['pdf'].exists() and out['png'].exists()
    assert not (tmp_path/'fig_pareto_cost_vs_accuracy_r1_10.pdf').exists()


def test_pareto_fails_if_benchmark_or_ratio_is_missing():
    s1,_ = campaign()
    other=[x for x in s1 if x['imbalance_ratio']=='1:50']
    with pytest.raises(MissingPublicationEvidence):
        pareto_summary(other,ratio='1:100')


def test_final_exposure_concentration_measured_not_temporal(tmp_path):
    _,s2=campaign()
    data=exposure_concentration(s2)
    assert data['uniform']['effective_classes_mean']==pytest.approx(100)
    assert data['mcts_enhanced']['effective_classes_mean'] < 100
    assert data['mcts_enhanced']['gini_mean'] > data['uniform']['gini_mean']
    out=plot_exposure_concentration(s2,tmp_path)
    assert out['png'].exists()


def test_holm_monotone_and_conservative():
    raw=[.04,.01,.03]
    adj=_holm(raw)
    assert adj==pytest.approx([.06,.03,.06])
    assert all(a>=b for a,b in zip(adj,raw))
    with pytest.raises(ValueError):_holm([float('nan')])


def test_effects_paired_and_multiple_testing_corrected(tmp_path):
    s1,s2=campaign()
    rows=paired_publication_effects(s1,s2)
    assert len(rows)==12
    assert all(row['n']==5 and row['family_size']==12 for row in rows)
    assert all(row['p_holm']>=row['p_raw'] for row in rows)
    assert set((r['scenario'],r['ratio']) for r in rows)=={
        ('S1','1:10'),('S1','1:50'),('S1','1:100'),('S2','—')}
    p=write_paired_effects_table(s1,s2,tmp_path)
    t=p.read_text()
    assert 'Holm' in t and '95\\%' in t and 'Top-1' in t
    assert '\\begin{table*}' in t and len(t.splitlines())>15


def test_missing_true_seed_pairs_never_receives_an_inferential_result():
    s1,s2=campaign()
    missing=[r for r in s1 if not (r['imbalance_ratio']=='1:50' and r['strategy']=='uniform' and r['seed']!=42)]
    with pytest.raises(MissingPublicationEvidence):paired_publication_effects(missing,s2)


def test_incomplete_ratio_cannot_shrink_holm_family():
    s1,s2=campaign()
    less=[r for r in s1 if r['imbalance_ratio']!='1:100']
    with pytest.raises(MissingPublicationEvidence, match='Missing prespecified'):
        paired_publication_effects(less,s2)


def test_axis2_duplicate_strategy_does_not_contaminate_axis1():
    s1,_=campaign()
    wrong=make_run('scenario1','mcts_enhanced',42,'1:50',f1=.99)
    wrong['benchmark_name']='tier1_axis2_tree_ablation'
    out=pareto_summary(s1+[wrong],ratio='1:50')
    assert out['mcts_enhanced']['f1_mean']<.99


def test_suite_loader_uses_exact_ids_and_rejects_identity_mismatch(tmp_path, monkeypatch):
    from src.utils import phase3_publication as pub
    # Validator is unit-mocked here; production still requires REAL sidecars.
    monkeypatch.setattr(pub, 'validate_provenance_metadata', lambda *a, **kw: {'execution_mode':'real'})
    runs = tmp_path / 'runs'; runs.mkdir()
    r = make_run('scenario1', 'uniform', 42, '1:50')
    report = tmp_path / 'tier1_axis1_samplers.json'
    report.write_text(json.dumps({'runs': [{'run_id': r['run_id']}]}))
    (runs / (r['run_id']+'.json')).write_text(json.dumps(r))
    loaded = pub.load_campaign_runs(tmp_path, 'scenario1')
    assert [x['run_id'] for x in loaded] == [r['run_id']]
    r['benchmark_name'] = 'tier1_axis2_tree_ablation'
    (runs / (r['run_id']+'.json')).write_text(json.dumps(r))
    with pytest.raises(ValueError, match='identity/benchmark mismatch'):
        pub.load_campaign_runs(tmp_path, 'scenario1')
