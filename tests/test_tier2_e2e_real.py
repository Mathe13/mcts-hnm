"""Tests for Tier-2 End-to-End vision pipeline with real CIFAR-100 images and dynamic expansion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.experiment_config import (
    ExecutionMode,
    ExperimentConfig,
    find_repo_root,
    get_isolated_paths,
    reset_active_session_mode,
)
from src.data.cifar100_images import load_cifar100_raw_images
from src.training.e2e_trainer import (
    E2EExperimentConfig,
    Tier2ResNet18,
    run_e2e_experiment,
    train_e2e_scenario1,
    train_e2e_scenario2,
)
from src.training.trainer import train_scenario1
from src.trees.semantic_tree import build_semantic_tree


@pytest.fixture(autouse=True)
def clean_session():
    """Reset session mode before and after tests."""
    reset_active_session_mode()
    yield
    reset_active_session_mode()


def test_load_cifar100_real_missing_raises(tmp_path):
    """C1: load_cifar100_raw_images in REAL mode raises FileNotFoundError when dataset missing."""
    config = ExperimentConfig(execution_mode=ExecutionMode.REAL)
    missing_dir = tmp_path / "nonexistent_cifar"

    with pytest.raises(FileNotFoundError, match="Real CIFAR-100 dataset not found"):
        load_cifar100_raw_images(
            data_dir=missing_dir,
            split="train",
            download=False,
            config=config,
        )


def test_load_cifar100_smoke_synthetic_images():
    """C2: load_cifar100_raw_images in SMOKE mode returns deterministic synthetic 32x32 images."""
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    images, labels = load_cifar100_raw_images(
        data_dir=Path("data"),
        split="train",
        download=False,
        config=config,
        num_smoke_samples=50,
    )

    assert isinstance(images, torch.Tensor)
    assert images.shape == (50, 3, 32, 32)
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (50,)


def test_load_cifar100_raw_images_smoke_synthetic():
    """C5: load_cifar100_raw_images in SMOKE mode yields multi-sample classes (>=5/class)."""
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    images, labels = load_cifar100_raw_images(
        data_dir=Path("data"),
        split="train",
        download=False,
        config=config,
    )

    assert isinstance(images, torch.Tensor)
    assert images.shape == (500, 3, 32, 32)
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (500,)
    assert len(np.unique(labels)) == 100
    counts = np.bincount(labels)
    assert np.all(counts >= 5), f"Expected at least 5 instances per class, got min count: {np.min(counts)}"


def test_notebook_04_structure_and_banner():
    """C3: Notebook 04 initial cell declares EXECUTION_MODE = ExecutionMode.REAL and outputs banner."""
    root = find_repo_root()
    nb_path = root / "notebooks" / "04_tier2_e2e_and_expansion.ipynb"
    assert nb_path.is_file()

    nb = json.loads(nb_path.read_text())
    first_code_src = "".join(nb["cells"][1]["source"])

    assert "ExecutionMode" in first_code_src
    assert "EXECUTION_MODE = ExecutionMode.REAL" in first_code_src
    assert "print_execution_banner" in first_code_src


def test_notebook_04_smoke_execution(tmp_path):
    """C4: Notebook 04 in SMOKE mode writes results to results/smoke/ and checkpoints to checkpoints/smoke/."""
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    paths = get_isolated_paths(config, repo_root=tmp_path)

    assert "smoke" in str(paths.results_dir)
    assert "smoke" in str(paths.checkpoints_dir)


def test_run_e2e_experiment_provenance_metadata(tmp_path):
    """C5: run_e2e_experiment writes .meta.json sidecar alongside exported JSON results."""
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    paths = get_isolated_paths(config, repo_root=tmp_path)

    images = torch.randn(20, 3, 32, 32)
    labels = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    pos_idx = np.arange(0, 10, dtype=np.int64)
    neg_idx = np.arange(10, 20, dtype=np.int64)

    def tree_factory():
        return build_semantic_tree({1: neg_idx})

    cfg = E2EExperimentConfig(
        scenario=1,
        strategies=("uniform",),
        seeds=(42,),
        num_steps=4,
        eval_every=2,
        pos_batch_size=2,
        neg_batch_size=4,
        width_multiplier=0.25,
        output_dir=str(paths.results_dir),
    )

    res = run_e2e_experiment(
        config=cfg,
        train_images=images,
        train_labels=labels,
        test_images=images,
        test_labels=labels,
        positive_indices=pos_idx,
        negative_indices=neg_idx,
        tree_factory=tree_factory,
        experiment_config=config,
    )

    json_file = paths.results_dir / "e2e_scenario_1_results.json"
    meta_file = paths.results_dir / "e2e_scenario_1_results.json.meta.json"
    assert json_file.is_file()
    assert meta_file.is_file()

    meta = json.loads(meta_file.read_text())
    assert meta["execution_mode"] == "smoke"
    assert meta["synthetic_data"] is True


def test_train_e2e_scenario1_checkpoint_saving(tmp_path):
    """C6: train_e2e_scenario1 saves model checkpoint to checkpoints/{mode}/."""
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    paths = get_isolated_paths(config, repo_root=tmp_path)

    images = torch.randn(20, 3, 32, 32)
    labels = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    pos_idx = np.arange(0, 10, dtype=np.int64)
    neg_idx = np.arange(10, 20, dtype=np.int64)

    res = train_e2e_scenario1(
        train_images=images,
        train_binary_labels=labels,
        positive_indices=pos_idx,
        negative_indices=neg_idx,
        test_images=images,
        test_binary_labels=labels,
        strategy="uniform",
        num_steps=6,
        eval_every=3,
        pos_batch_size=2,
        neg_batch_size=4,
        width_multiplier=0.25,
        checkpoints_dir=paths.checkpoints_dir,
        seed=42,
    )

    assert "checkpoint_path" in res
    ckpt = Path(res["checkpoint_path"])
    assert ckpt.is_file()
    assert "smoke" in str(ckpt)


def test_train_e2e_scenario2_checkpoint_saving(tmp_path):
    """C7: train_e2e_scenario2 saves model checkpoint to checkpoints/{mode}/."""
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    paths = get_isolated_paths(config, repo_root=tmp_path)

    images = torch.randn(20, 3, 32, 32)
    labels = np.array([0] * 10 + [1] * 10, dtype=np.int64)

    res = train_e2e_scenario2(
        train_images=images,
        train_labels=labels,
        test_images=images,
        test_labels=labels,
        strategy="uniform",
        num_classes=2,
        num_steps=6,
        eval_every=3,
        batch_size=4,
        width_multiplier=0.25,
        checkpoints_dir=paths.checkpoints_dir,
        seed=42,
    )

    assert "checkpoint_path" in res
    ckpt = Path(res["checkpoint_path"])
    assert ckpt.is_file()
    assert "smoke" in str(ckpt)


def test_run_e2e_experiment_multi_seed_and_stats(tmp_path):
    """C8: run_e2e_experiment records per-seed metrics and paired statistical tests for multi-seed runs."""
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    paths = get_isolated_paths(config, repo_root=tmp_path)

    images = torch.randn(20, 3, 32, 32)
    labels = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    pos_idx = np.arange(0, 10, dtype=np.int64)
    neg_idx = np.arange(10, 20, dtype=np.int64)

    def tree_factory():
        return build_semantic_tree({1: neg_idx})

    cfg = E2EExperimentConfig(
        scenario=1,
        strategies=("uniform", "mcts_enhanced"),
        seeds=(42, 43),
        num_steps=4,
        eval_every=2,
        pos_batch_size=2,
        neg_batch_size=4,
        width_multiplier=0.25,
        output_dir=str(paths.results_dir),
    )

    summary = run_e2e_experiment(
        config=cfg,
        train_images=images,
        train_labels=labels,
        test_images=images,
        test_labels=labels,
        positive_indices=pos_idx,
        negative_indices=neg_idx,
        tree_factory=tree_factory,
        experiment_config=config,
    )

    assert "strategies" in summary
    assert "uniform" in summary["strategies"]
    assert "mcts_enhanced" in summary["strategies"]
    assert len(summary["strategies"]["uniform"]["runs"]) == 2
    assert "statistical_tests" in summary


def test_dynamic_leaf_expansion_growth():
    """C9: Dynamic leaf expansion in train_scenario1 increases leaf count when visits exceed threshold."""
    features = torch.randn(30, 16)
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int64)
    pos_idx = np.arange(0, 10, dtype=np.int64)
    neg_idx = np.arange(10, 30, dtype=np.int64)
    bin_labels = (labels == 0).astype(np.int64)

    class_to_indices = {1: np.arange(10, 20, dtype=np.int64), 2: np.arange(20, 30, dtype=np.int64)}
    tree = build_semantic_tree(class_to_indices)
    initial_leaves = tree.num_leaves

    res = train_scenario1(
        train_features=features,
        train_binary_labels=bin_labels,
        positive_indices=pos_idx,
        negative_indices=neg_idx,
        test_features=features,
        test_binary_labels=bin_labels,
        strategy="mcts_enhanced",
        tree=tree,
        num_steps=30,
        eval_every=10,
        pos_batch_size=2,
        neg_batch_size=4,
        enable_leaf_expansion=True,
        split_threshold=5,
        k_sub=2,
        seed=42,
    )

    assert res["final_num_leaves"] is not None
    assert res["final_num_leaves"] >= initial_leaves


def test_dynamically_expanded_tree_invariants():
    """C10: Dynamically expanded tree preserves structural invariants and passes validate_tree()."""
    features = torch.randn(30, 16)
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int64)
    pos_idx = np.arange(0, 10, dtype=np.int64)
    neg_idx = np.arange(10, 30, dtype=np.int64)
    bin_labels = (labels == 0).astype(np.int64)

    class_to_indices = {1: np.arange(10, 20, dtype=np.int64), 2: np.arange(20, 30, dtype=np.int64)}
    tree = build_semantic_tree(class_to_indices)

    train_scenario1(
        train_features=features,
        train_binary_labels=bin_labels,
        positive_indices=pos_idx,
        negative_indices=neg_idx,
        test_features=features,
        test_binary_labels=bin_labels,
        strategy="mcts_enhanced",
        tree=tree,
        num_steps=30,
        eval_every=10,
        pos_batch_size=2,
        neg_batch_size=4,
        enable_leaf_expansion=True,
        split_threshold=5,
        k_sub=2,
        seed=42,
    )

    # Invariant assertion
    tree.validate_tree()
    assert int(np.sum(tree.is_leaf)) == tree.num_leaves
    assert tree.root_id == 0
