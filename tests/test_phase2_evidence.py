"""Tests for measured-only Phase-2 publication data. No CIFAR downloads or training."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score

from src.utils.phase1_evidence import EvidenceUnavailable
from src.utils.phase2_evidence import (
    capture_class_coverage,
    coverage_curves,
    measured_pr_by_seed,
    measure_binary_pr,
    plot_coverage_entropy,
    plot_measured_pr,
)


def _write_run(tmp_path: Path, *, scenario: str, strategy: str, seed: int,
               history=None, binary_test_pr=None, run_id=None, **kw):
    run_id = run_id or f'{scenario}_{strategy}_{seed}'
    payload = {'scenario': scenario, 'strategy': strategy, 'seed': seed,
               'run_id': run_id, 'tree_type': 'semantic' if strategy.startswith('mcts') else 'none',
               'target_class': 0 if scenario == 'scenario1' else None,
               'imbalance_ratio': '1:50' if scenario == 'scenario1' else None,
               'benchmark_name': 'tier1_axis1_samplers' if scenario == 'scenario1' else 'tier1_scenario2_curriculum',
               'history': history or [], 'final_metrics': {}, **kw}
    if binary_test_pr is not None:
        payload['binary_test_pr'] = binary_test_pr
        payload['final_metrics']['auprc'] = binary_test_pr['average_precision']
    fp = tmp_path / f'{run_id}.json'
    fp.write_text(json.dumps(payload), encoding='utf-8')
    return run_id, {'run_file': str(fp), 'scenario': scenario, 'strategy': strategy, 'seed': seed}


def test_coverage_is_empirical_and_monotonic():
    assert capture_class_coverage(np.array([0, 1, 4, 5, 10])) == {
        'class_coverage_k1': 0.8, 'class_coverage_k5': 0.4,
    }
    with pytest.raises(ValueError):
        capture_class_coverage(np.array([2, -1, 4]))
    with pytest.raises(ValueError):
        capture_class_coverage(np.array([0.1, 0.7]))


def test_measured_pr_matches_sklearn_ap_and_restores_mode():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2., -1.]]))
    model.train()
    features = torch.tensor([[1., 0.], [0., 1.], [.7, .3], [-1., 0.], [2., 0.]])
    y = np.array([1, 0, 1, 0, 1])
    result = measure_binary_pr(model, features, y, batch_size=2)
    assert model.training
    with torch.no_grad():
        expected = torch.sigmoid(model(features).reshape(-1)).numpy()
    assert np.isclose(result['average_precision'], average_precision_score(y, expected))
    assert result['n_positive'] == 3 and result['n_negative'] == 2
    assert all(np.diff(result['recall']) <= 1e-10)
    assert result['checkpoint'] == 'final' and result['split'] == 'test'


def test_measured_pr_rejects_missing_and_inconsistent_atomic_runs(tmp_path):
    items = {}
    # No REAL or synthetic fallback for absent PR points.
    for strat in ('uniform', 'ohem', 'mcts_strict', 'mcts_enhanced'):
        for seed in (42, 43):
            key, val = _write_run(tmp_path, scenario='scenario1', strategy=strat, seed=seed)
            items[key] = val
    with pytest.raises(EvidenceUnavailable):
        measured_pr_by_seed(items)

    measured = {'split': 'test', 'checkpoint': 'final', 'metric': 'average_precision',
                'precision': [0.25, 0.5, 1.0], 'recall': [1.0, 0.5, 0.0],
                'average_precision': .55, 'n_positive': 3, 'n_negative': 8}
    items.clear()
    for strat in ('uniform', 'ohem', 'mcts_strict', 'mcts_enhanced'):
        for seed in (42, 43):
            key, val = _write_run(tmp_path, scenario='scenario1', strategy=strat, seed=seed,
                                  binary_test_pr=measured)
            items[key] = val
    curves = measured_pr_by_seed(items)
    assert all(len(x) == 2 for x in curves.values())
    assert plot_measured_pr(items, tmp_path)['pdf'].exists()
    item = tmp_path / 'scenario1_uniform_42.json'
    payload = json.loads(item.read_text())
    payload['binary_test_pr']['average_precision'] = .99
    item.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='inconsistent'):
        measured_pr_by_seed(items)


def test_coverage_requires_real_per_step_arrays(tmp_path):
    items = {}
    for strat in ('uniform', 'mcts_enhanced'):
        for seed in (42, 43):
            records = []
            for step, k1, k5 in ((10, .6, .2), (20, .8, .4), (30, 1., .6)):
                row = {'step': step, 'class_coverage_k1': k1, 'class_coverage_k5': k5}
                if strat.startswith('mcts'): row['normalized_entropy'] = .9
                records.append(row)
            key, value = _write_run(tmp_path, scenario='scenario2', strategy=strat, seed=seed, history=records)
            items[key] = value
    data = coverage_curves(items)
    assert np.isclose(data['uniform'][42][20][0], .8)
    assert plot_coverage_entropy(items, tmp_path)['png'].exists()
    bad = tmp_path / 'scenario2_uniform_42.json'
    d = json.loads(bad.read_text()); d['history'][1]['class_coverage_k5'] = .9
    bad.write_text(json.dumps(d))
    with pytest.raises(ValueError): coverage_curves(items)


def test_coverage_requires_two_seeds_and_checkpoints(tmp_path):
    key, value = _write_run(tmp_path, scenario='scenario2', strategy='mcts_enhanced', seed=42,
                             history=[{'step': 10, 'class_coverage_k1': .5, 'class_coverage_k5': .1}])
    with pytest.raises(EvidenceUnavailable): plot_coverage_entropy({key: value}, tmp_path)


def test_unmeasured_pr_cannot_be_inferred_from_auprc(tmp_path):
    key, val = _write_run(tmp_path, scenario='scenario1', strategy='uniform', seed=42)
    doc = json.loads(Path(val['run_file']).read_text())
    doc['final_metrics'] = {'auprc': 0.80}
    Path(val['run_file']).write_text(json.dumps(doc))
    with pytest.raises(EvidenceUnavailable): measured_pr_by_seed({key: val})


def test_four_quadrants_use_canonical_ratio_only_after_multi_ratio_run(tmp_path):
    """Regression: extra 1:10/1:100 runs cannot masquerade as extra seeds."""
    from src.utils.phase1_evidence import measured_quadrants
    items = {}
    for strat in ('uniform', 'ohem', 'mcts_strict', 'mcts_enhanced'):
        for seed in (42, 43):
            for ratio in ('1:10', '1:50', '1:100'):
                rid = f'axis1_{strat}_{seed}_{ratio.replace(":", "_")}'
                key, item = _write_run(tmp_path, scenario='scenario1', strategy=strat,
                                       seed=seed, run_id=rid)
                raw = json.loads(Path(item['run_file']).read_text())
                raw['imbalance_ratio'] = ratio
                raw['evaluation_quadrants'] = dict(
                    q1_last_fixed=.65, q2_last_calibrated=.7,
                    q3_best_fixed=.68, q4_best_calibrated=.72,
                )
                Path(item['run_file']).write_text(json.dumps(raw))
                items[key] = item
    result = measured_quadrants(items)
    assert result['uniform']['values'].shape == (2, 4)
    assert result['mcts_enhanced']['seeds'] == [42, 43]


def test_two_step_binary_trainer_persists_measured_test_pr():
    """Small real-tensor integration, not a scientific benchmark or synthetic fallback."""
    trainer = pytest.importorskip('src.training.trainer')
    rng = np.random.default_rng(42)
    features = torch.as_tensor(rng.normal(size=(30, 8)), dtype=torch.float32)
    train_y = np.array([1] * 6 + [0] * 24, dtype=np.int64)
    val = torch.as_tensor(rng.normal(size=(12, 8)), dtype=torch.float32)
    val_y = np.array([1] * 4 + [0] * 8, dtype=np.int64)
    res = trainer.train_scenario1(
        train_features=features, train_binary_labels=train_y,
        positive_indices=np.flatnonzero(train_y == 1),
        negative_indices=np.flatnonzero(train_y == 0),
        val_features=val, val_binary_labels=val_y,
        test_features=val.clone(), test_binary_labels=val_y.copy(),
        strategy='uniform', num_steps=2, eval_every=1,
        pos_batch_size=2, neg_batch_size=4, device=torch.device('cpu'),
    )
    assert isinstance(res['binary_test_pr'], dict)
    assert res['binary_test_pr']['split'] == 'test'
    assert np.isclose(res['binary_test_pr']['average_precision'], res['last_metrics']['auprc'])


def test_two_step_multiclass_trainer_records_coverage():
    trainer = pytest.importorskip('src.training.trainer')
    rng = np.random.default_rng(123)
    tr_x = torch.as_tensor(rng.normal(size=(30, 8)), dtype=torch.float32)
    tr_y = np.arange(30, dtype=np.int64) % 3
    va_x = torch.as_tensor(rng.normal(size=(12, 8)), dtype=torch.float32)
    va_y = np.arange(12, dtype=np.int64) % 3
    result = trainer.train_scenario2(
        train_features=tr_x, train_labels=tr_y,
        val_features=va_x, val_labels=va_y,
        test_features=va_x.clone(), test_labels=va_y.copy(),
        strategy='uniform', num_classes=3, num_steps=2, eval_every=1,
        batch_size=6, device=torch.device('cpu'),
    )
    hist = result['history']
    assert len(hist) == 2
    assert all('class_coverage_k1' in point and 'class_coverage_k5' in point for point in hist)
    assert hist[-1]['class_coverage_k1'] >= hist[0]['class_coverage_k1']
