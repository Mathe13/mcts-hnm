"""Regression tests for Phase-1 scientific-evidence recovery.

No CIFAR-100 download or heavyweight benchmark execution is required.
"""
import json

import numpy as np
import pytest

from src.utils.phase1_evidence import (
    EvidenceUnavailable,
    _checked_confusion,
    extract_four_quadrants,
    measured_entropy_histories,
    measured_exposure_pairs,
    measured_quadrants,
    inspect_existing_diagnostics,
)


def _make_run(tmp_path, scenario, strat, seed, **extra):
    run_id=f'{scenario}_{strat}_seed{seed}'
    f=tmp_path/f'{run_id}.json'
    data={'scenario':scenario,'strategy':strat,'seed':seed,'run_id':run_id,
          'tree_type':'semantic' if strat.startswith('mcts') else 'none',**extra}
    f.write_text(json.dumps(data),encoding='utf-8')
    return run_id,dict(run_file=str(f),scenario=scenario,strategy=strat,seed=seed)


def test_confusion_matrix_requires_actual_100_square():
    assert _checked_confusion(np.eye(100)).shape==(100,100)
    for x in [np.ones((20,20)), -np.eye(100), np.full((100,100),np.nan)]:
        with pytest.raises(ValueError):_checked_confusion(x)


def test_entropy_is_preserved_from_real_checkpoints(tmp_path):
    items={}
    for sd in (42,43):
        run,info=_make_run(tmp_path,'scenario2','mcts_enhanced',sd,
          history=[{'step':25,'normalized_entropy':0.3},{'step':50,'normalized_entropy':0.6}])
        items[run]=info
    hist=measured_entropy_histories(items)
    assert hist['mcts_enhanced'][42][25]==pytest.approx(0.3)
    assert hist['mcts_enhanced'][43][50]==pytest.approx(0.6)


def test_historical_scenario2_snapshots_are_not_multiseed_pairs(tmp_path):
    f=tmp_path/'tier1_scenario2_curriculum.json'
    f.write_text(json.dumps({'diagnostics':{'uniform':{'class_exposure_counts':[3]*100},
                                            'mcts_enhanced':{'per_class_recall':[0.1]*100}}}))
    x=inspect_existing_diagnostics(tmp_path)
    assert x['present'] is True
    assert x['per_seed_diagnostics_available'] is False
    assert 'DO NOT' in x['note']


def test_exposure_delta_requires_explicit_paired_seed_data(tmp_path):
    items={}
    # The old atomic run fields do not contain per-class arrays.
    for st in ('uniform','mcts_enhanced'):
        run,info=_make_run(tmp_path,'scenario2',st,42,final_metrics={'macro_f1':0.5})
        items[run]=info
    with pytest.raises(EvidenceUnavailable):measured_exposure_pairs(items)


def test_exposure_delta_calculated_on_paired_seeds(tmp_path):
    items={}
    for seed in (42,43):
        for strat in ('uniform','mcts_enhanced'):
            rec=([0.2]*100 if strat=='uniform' else [0.3]*100)
            run,info=_make_run(tmp_path,'scenario2',strat,seed,
                 per_class_recall=rec,exposure_diagnostics={'class_exposure_counts':[4]*100})
            items[run]=info
    d=measured_exposure_pairs(items)
    assert d['seeds']==[42,43]
    np.testing.assert_allclose(d['mean_delta_recall'],0.1)
    np.testing.assert_allclose(d['mean_exposure'],0.01)


def test_quadrants_are_computed_from_actual_test_metrics():
    assert extract_four_quadrants({'f1_thresh05':0.3,'f1':0.4},
                                  {'f1_thresh05':0.5,'f1':0.6})=={
        'q1_last_fixed':0.3,'q2_last_calibrated':0.4,
        'q3_best_fixed':0.5,'q4_best_calibrated':0.6,
    }
    assert extract_four_quadrants({'f1':0.5},{'f1':0.7}) is None


def test_quadrants_require_all_methods_and_paired_seeds(tmp_path):
    items={}
    for seed in (42,43):
        for st in ('uniform','ohem','mcts_strict','mcts_enhanced'):
            run,info=_make_run(tmp_path,'scenario1',st,seed,
                 last_metrics={'f1_thresh05':0.4,'f1':0.5},
                 best_validation_metrics={'f1_thresh05':0.6,'f1':0.7})
            items[run]=info
    d=measured_quadrants(items)
    assert d['uniform']['values'].shape==(2,4)
    items.pop('scenario1_mcts_enhanced_seed43')
    with pytest.raises(EvidenceUnavailable):measured_quadrants(items)


def test_identity_mismatch_is_never_accepted(tmp_path):
    run,info=_make_run(tmp_path,'scenario2','uniform',42)
    info['seed']=43
    with pytest.raises(ValueError):measured_entropy_histories({run:info})
