"""Tests for Feature F5: Dynamic entropy leaf expansion and Tier-2 end-to-end vision pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data.cifar100_taxonomy import get_cifar100_taxonomy
from src.mcts.agent import MCTSHNMAgent
from src.mcts.diagnostics import (
    MCTSDiagnosticsTracker,
    compute_superclass_visit_distribution,
    compute_visitation_entropy,
)
from src.mcts.expansion import (
    SUBCLUSTER_ID_STRIDE,
    compute_prediction_entropy,
    expand_leaf_by_entropy,
    maybe_expand_leaf,
    resolve_base_class_id,
)
from src.mcts.sampler import DualStreamBatchSampler
from src.trees.base_tree import FastHierarchyTree
from src.trees.semantic_tree import build_semantic_tree


@pytest.fixture
def toy_tree() -> FastHierarchyTree:
    """Create a toy 3-class tree with Root -> 3 Leaves."""
    parents = np.array([-1, 0, 0, 0], dtype=np.int64)
    children = np.array([[1, 2, 3], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]], dtype=np.int64)
    child_counts = np.array([3, 0, 0, 0], dtype=np.int64)
    is_leaf = np.array([False, True, True, True], dtype=bool)
    depths = np.array([0, 1, 1, 1], dtype=np.int64)
    leaf_to_class_id = np.array([-1, 0, 1, 2], dtype=np.int64)
    instance_counts = np.array([30, 10, 10, 10], dtype=np.int64)
    visits = np.zeros(4, dtype=np.int64)
    q_values = np.zeros(4, dtype=np.float64)
    node_names = ("root", "class:0", "class:1", "class:2")
    class_to_indices = {
        0: np.arange(0, 10, dtype=np.int64),
        1: np.arange(10, 20, dtype=np.int64),
        2: np.arange(20, 30, dtype=np.int64),
    }
    return FastHierarchyTree(
        tree_type="toy",
        root_id=0,
        parents=parents,
        children=children,
        child_counts=child_counts,
        is_leaf=is_leaf,
        depths=depths,
        leaf_to_class_id=leaf_to_class_id,
        instance_counts=instance_counts,
        visits=visits,
        q_values=q_values,
        node_names=node_names,
        class_to_indices=class_to_indices,
    )


# --- C1 Tests ---
def test_compute_prediction_entropy_binary_and_multiclass() -> None:
    # Binary logits: z=0 -> p=0.5 -> H = ln(2) ~= 0.693147
    logits_bin = np.array([0.0, 10.0, -10.0], dtype=np.float32)
    ent_bin = compute_prediction_entropy(logits_bin, is_binary=True)
    assert ent_bin.shape == (3,)
    assert np.isclose(ent_bin[0], np.log(2.0), atol=1e-4)
    assert ent_bin[1] < 1e-3  # high confidence positive -> low entropy
    assert ent_bin[2] < 1e-3  # high confidence negative -> low entropy
    assert np.all(ent_bin >= 0.0)

    # Multi-class logits: 4 uniform classes -> H = ln(4)
    logits_multi = torch.tensor([[1.0, 1.0, 1.0, 1.0], [15.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    ent_multi = compute_prediction_entropy(logits_multi, is_binary=False)
    assert ent_multi.shape == (2,)
    assert np.isclose(ent_multi[0], np.log(4.0), atol=1e-4)
    assert ent_multi[1] < 1e-3
    assert np.all(ent_multi >= 0.0)


# --- C2 Tests ---
def test_expand_leaf_by_entropy_splits_and_preserves_invariants(toy_tree: FastHierarchyTree) -> None:
    tree = toy_tree
    leaf_id = 1  # class 0 with 10 instances: indices 0..9
    entropies = np.linspace(0.1, 0.9, 10)  # increasing entropy
    new_children = expand_leaf_by_entropy(tree, leaf_node_id=leaf_id, entropies=entropies, k_sub=3)

    assert len(new_children) == 3
    assert not tree.is_leaf[leaf_id]
    assert tree.leaf_to_class_id[leaf_id] == -1
    assert tree.child_counts[leaf_id] == 3

    # Check children
    sub_classes = []
    total_child_instances = 0
    for cid in new_children:
        assert tree.is_leaf[cid]
        v_cls = tree.leaf_to_class_id[cid]
        assert v_cls >= SUBCLUSTER_ID_STRIDE
        assert resolve_base_class_id(v_cls) == 0
        sub_classes.append(v_cls)
        total_child_instances += tree.instance_counts[cid]

    assert total_child_instances == 10
    assert tree.instance_counts[leaf_id] == 10
    # Tree validation passes
    tree.validate_tree()


# --- C3 Tests ---
def test_mcts_agent_sampler_and_diagnostics_on_expanded_tree(toy_tree: FastHierarchyTree) -> None:
    tree = toy_tree
    leaf_id = 1
    entropies = np.linspace(0.1, 0.9, 10)
    new_children = expand_leaf_by_entropy(tree, leaf_node_id=leaf_id, entropies=entropies, k_sub=2)

    agent = MCTSHNMAgent(tree=tree, exploration_c=2.0, momentum_alpha=0.3, q_norm=True)
    sampler = DualStreamBatchSampler(tree=tree, neg_batch_size=4, dual_stream_ratio=0.5, seed=42)

    # Select leaf should visit unvisited children
    selected_leaf, path = agent.select_leaf()
    assert selected_leaf in tree.leaf_node_ids()
    batch = sampler.sample_batch(selected_leaf)
    assert batch.negative_indices.size == 4

    # Backpropagation
    agent.update(selected_leaf, reward=1.0)
    assert tree.visits[selected_leaf] == 1
    assert tree.visits[leaf_id] == 1
    assert tree.visits[tree.root_id] == 1

    # Diagnostics and superclass distribution
    dist = compute_superclass_visit_distribution(tree)
    assert len(dist) == 20
    assert np.isclose(sum(dist.values()), 1.0)
    tax = get_cifar100_taxonomy()
    expected_coarse = tax.coarse_names[tax.fine_to_coarse_idx[0]]
    assert dist[expected_coarse] == 1.0

    tracker = MCTSDiagnosticsTracker(tree=tree)
    snap = tracker.record_step(1, selected_leaf, reward=1.0)
    assert snap.leaf_class_id == tree.leaf_to_class_id[selected_leaf]
    summary = tracker.summary()
    assert summary["total_leaf_visits"] == 1


# --- C4 Tests ---
def test_expansion_invalid_arguments_raise(toy_tree: FastHierarchyTree) -> None:
    tree = toy_tree
    # Out of range leaf_node_id
    with pytest.raises(ValueError):
        expand_leaf_by_entropy(tree, leaf_node_id=99, entropies=np.zeros(10))

    # Internal node
    with pytest.raises(ValueError):
        expand_leaf_by_entropy(tree, leaf_node_id=0, entropies=np.zeros(30))

    # k_sub < 2
    with pytest.raises(ValueError):
        expand_leaf_by_entropy(tree, leaf_node_id=1, entropies=np.zeros(10), k_sub=1)

    # Mismatched entropies size
    with pytest.raises(ValueError):
        expand_leaf_by_entropy(tree, leaf_node_id=1, entropies=np.zeros(5), k_sub=2)


# --- C7 Tests ---
def test_tier2_resnet18_forward_and_features() -> None:
    from src.training.e2e_trainer import Tier2ResNet18

    # Width multiplier 0.25 for quick execution
    model_bin = Tier2ResNet18(num_outputs=1, width_multiplier=0.25, squeeze_binary=True)
    images = torch.randn(4, 3, 32, 32)
    out_bin = model_bin(images)
    assert out_bin.shape == (4,)

    feats = model_bin.forward_features(images)
    assert feats.ndim == 2
    assert feats.shape[0] == 4

    model_multi = Tier2ResNet18(num_outputs=10, width_multiplier=0.25, squeeze_binary=False)
    out_multi = model_multi(images)
    assert out_multi.shape == (4, 10)


# --- C8 Tests ---
def test_train_e2e_scenario1_and_scenario2_with_refresh_and_expansion(toy_tree: FastHierarchyTree) -> None:
    from src.training.e2e_trainer import train_e2e_scenario1, train_e2e_scenario2

    images = torch.randn(30, 3, 32, 32)
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int64)
    pos_idx = np.arange(0, 10, dtype=np.int64)
    neg_idx = np.arange(10, 30, dtype=np.int64)
    bin_labels = (labels == 0).astype(np.int64)

    tree = FastHierarchyTree(
        tree_type="confusion_ward",
        root_id=0,
        parents=toy_tree.parents.copy(),
        children=toy_tree.children.copy(),
        child_counts=toy_tree.child_counts.copy(),
        is_leaf=toy_tree.is_leaf.copy(),
        depths=toy_tree.depths.copy(),
        leaf_to_class_id=toy_tree.leaf_to_class_id.copy(),
        instance_counts=toy_tree.instance_counts.copy(),
        visits=toy_tree.visits.copy(),
        q_values=toy_tree.q_values.copy(),
        node_names=toy_tree.node_names,
        class_to_indices={k: v.copy() for k, v in toy_tree.class_to_indices.items()},
    )

    res1 = train_e2e_scenario1(
        train_images=images,
        train_binary_labels=bin_labels,
        positive_indices=pos_idx,
        negative_indices=neg_idx,
        test_images=images,
        test_binary_labels=bin_labels,
        strategy="mcts_enhanced",
        tree=tree,
        val_images=images,
        val_binary_labels=bin_labels,
        num_steps=10,
        eval_every=5,
        refresh_tree_every=5,
        pos_batch_size=2,
        neg_batch_size=4,
        width_multiplier=0.25,
        enable_leaf_expansion=True,
        split_threshold=3,
        k_sub=2,
        seed=42,
    )
    assert res1["final_metrics"]["loss"] is not None

    res2 = train_e2e_scenario2(
        train_images=images,
        train_labels=labels,
        test_images=images,
        test_labels=labels,
        strategy="mcts_enhanced",
        tree=tree,
        val_images=images,
        val_labels=labels,
        num_classes=3,
        num_steps=10,
        eval_every=5,
        refresh_tree_every=5,
        batch_size=4,
        width_multiplier=0.25,
        enable_leaf_expansion=True,
        split_threshold=3,
        k_sub=2,
        seed=42,
    )
    assert res2["final_metrics"]["loss"] is not None


# --- C9 Tests ---
def test_run_e2e_experiment_multi_seed_and_export(tmp_path: Path, toy_tree: FastHierarchyTree) -> None:
    from src.training.e2e_trainer import E2EExperimentConfig, run_e2e_experiment

    images = torch.randn(30, 3, 32, 32)
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int64)
    pos_idx = np.arange(0, 10, dtype=np.int64)
    neg_idx = np.arange(10, 30, dtype=np.int64)
    bin_labels = (labels == 0).astype(np.int64)

    def tree_factory() -> FastHierarchyTree:
        return FastHierarchyTree(
            tree_type="toy",
            root_id=0,
            parents=toy_tree.parents.copy(),
            children=toy_tree.children.copy(),
            child_counts=toy_tree.child_counts.copy(),
            is_leaf=toy_tree.is_leaf.copy(),
            depths=toy_tree.depths.copy(),
            leaf_to_class_id=toy_tree.leaf_to_class_id.copy(),
            instance_counts=toy_tree.instance_counts.copy(),
            visits=toy_tree.visits.copy(),
            q_values=toy_tree.q_values.copy(),
            node_names=toy_tree.node_names,
            class_to_indices={k: v.copy() for k, v in toy_tree.class_to_indices.items()},
        )

    cfg = E2EExperimentConfig(
        scenario=1,
        strategies=("uniform", "mcts_enhanced"),
        seeds=(42, 43),
        num_steps=6,
        eval_every=3,
        pos_batch_size=2,
        neg_batch_size=4,
        width_multiplier=0.25,
        output_dir=str(tmp_path),
    )

    summary = run_e2e_experiment(
        config=cfg,
        train_images=images,
        train_labels=bin_labels,
        test_images=images,
        test_labels=bin_labels,
        positive_indices=pos_idx,
        negative_indices=neg_idx,
        tree_factory=tree_factory,
    )

    assert "strategies" in summary
    assert "statistical_tests" in summary
    assert (tmp_path / "e2e_scenario_1_results.json").exists()
    assert (tmp_path / "e2e_scenario_1_results.csv").exists()


# --- C11 Tests ---
def test_notebook_04_execution() -> None:
    import ast
    import nbformat

    repo_root = Path(__file__).resolve().parent.parent
    nb_path = repo_root / "notebooks" / "04_tier2_e2e_and_expansion.ipynb"
    assert nb_path.is_file(), "notebooks/04_tier2_e2e_and_expansion.ipynb must exist"

    nb = nbformat.read(nb_path, as_version=4)
    nbformat.validate(nb)

    md_cells = [c for c in nb.cells if c.cell_type == "markdown"]
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(md_cells) >= 3
    assert len(code_cells) >= 3

    joined_code = "\n\n".join(c.source for c in code_cells)
    for cell in code_cells:
        ast.parse(cell.source)

    for required_ref in (
        "src.training.e2e_trainer",
        "Tier2ResNet18",
        "run_e2e_experiment",
        "enable_leaf_expansion",
    ):
        assert required_ref in joined_code, f"Notebook 04 missing reference to {required_ref}"
def test_e2e_trainer_invalid_arguments_raise() -> None:
    from src.training.e2e_trainer import E2EExperimentConfig, Tier2ResNet18, run_e2e_experiment, train_e2e_scenario1

    # Tier2ResNet18 invalid args
    with pytest.raises(ValueError):
        Tier2ResNet18(num_outputs=0)
    with pytest.raises(ValueError):
        Tier2ResNet18(width_multiplier=-1.0)
    with pytest.raises(ValueError):
        Tier2ResNet18(dropout=1.0)

    # Invalid image shape
    model = Tier2ResNet18(width_multiplier=0.25)
    with pytest.raises(ValueError):
        model.forward_features(torch.randn(2, 4, 32, 32))  # 4 channels

    # Unknown strategy
    with pytest.raises(ValueError):
        train_e2e_scenario1(
            train_images=torch.randn(10, 3, 32, 32),
            train_binary_labels=np.zeros(10),
            positive_indices=np.array([0]),
            negative_indices=np.array([1]),
            test_images=torch.randn(10, 3, 32, 32),
            test_binary_labels=np.zeros(10),
            strategy="invalid_strat",
        )

    # Invalid scenario in config
    with pytest.raises(ValueError):
        run_e2e_experiment(
            config=E2EExperimentConfig(scenario=3),
            train_images=torch.randn(10, 3, 32, 32),
            train_labels=np.zeros(10),
            test_images=torch.randn(10, 3, 32, 32),
            test_labels=np.zeros(10),
        )
def test_tier1_trainers_with_dynamic_leaf_expansion(toy_tree: FastHierarchyTree) -> None:
    from src.training.trainer import train_scenario1, train_scenario2

    torch.manual_seed(42)
    features = torch.randn(30, 16)
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int64)

    # Scenario 1 (binary: class 0 is positive, classes 1 and 2 are negatives)
    pos_idx = np.arange(0, 10, dtype=np.int64)
    neg_idx = np.arange(10, 30, dtype=np.int64)
    bin_labels = (labels == 0).astype(np.int64)

    tree1 = FastHierarchyTree(
        tree_type="toy1",
        root_id=0,
        parents=toy_tree.parents.copy(),
        children=toy_tree.children.copy(),
        child_counts=toy_tree.child_counts.copy(),
        is_leaf=toy_tree.is_leaf.copy(),
        depths=toy_tree.depths.copy(),
        leaf_to_class_id=toy_tree.leaf_to_class_id.copy(),
        instance_counts=toy_tree.instance_counts.copy(),
        visits=toy_tree.visits.copy(),
        q_values=toy_tree.q_values.copy(),
        node_names=toy_tree.node_names,
        class_to_indices={k: v.copy() for k, v in toy_tree.class_to_indices.items()},
    )

    res1 = train_scenario1(
        train_features=features,
        train_binary_labels=bin_labels,
        positive_indices=pos_idx,
        negative_indices=neg_idx,
        val_features=features,
        val_binary_labels=bin_labels,
        test_features=features,
        test_binary_labels=bin_labels,
        strategy="mcts_enhanced",
        tree=tree1,
        num_steps=30,
        eval_every=10,
        pos_batch_size=2,
        neg_batch_size=4,
        enable_leaf_expansion=True,
        split_threshold=5,
        k_sub=2,
        seed=42,
    )
    assert res1["final_num_leaves"] is not None
    assert res1["final_num_leaves"] >= 3  # at least initial 3 leaves

    # Scenario 2 (multi-class: 3 classes)
    tree2 = FastHierarchyTree(
        tree_type="toy2",
        root_id=0,
        parents=toy_tree.parents.copy(),
        children=toy_tree.children.copy(),
        child_counts=toy_tree.child_counts.copy(),
        is_leaf=toy_tree.is_leaf.copy(),
        depths=toy_tree.depths.copy(),
        leaf_to_class_id=toy_tree.leaf_to_class_id.copy(),
        instance_counts=toy_tree.instance_counts.copy(),
        visits=toy_tree.visits.copy(),
        q_values=toy_tree.q_values.copy(),
        node_names=toy_tree.node_names,
        class_to_indices={k: v.copy() for k, v in toy_tree.class_to_indices.items()},
    )

    res2 = train_scenario2(
        train_features=features,
        train_labels=labels,
        val_features=features,
        val_labels=labels,
        test_features=features,
        test_labels=labels,
        strategy="mcts_enhanced",
        tree=tree2,
        num_classes=3,
        num_steps=30,
        eval_every=10,
        batch_size=4,
        enable_leaf_expansion=True,
        split_threshold=5,
        k_sub=2,
        seed=42,
    )
    assert res2["final_num_leaves"] is not None
    assert res2["final_num_leaves"] >= 3
def test_expansion_edge_cases_noop_bmax_and_ties(toy_tree: FastHierarchyTree) -> None:
    tree = toy_tree
    # Dummy model
    model = nn.Linear(5, 1)
    inputs = torch.randn(30, 5)

    # visits < split_threshold -> returns None
    res = maybe_expand_leaf(tree, leaf_node_id=1, model=model, train_inputs=inputs, split_threshold=50)
    assert res is None

    # tied entropies
    entropies = np.full(10, 0.5)
    new_children = expand_leaf_by_entropy(tree, leaf_node_id=1, entropies=entropies, k_sub=4)
    assert len(new_children) == 4
    tree.validate_tree()

    # calling on already expanded sub-cluster
    res2 = maybe_expand_leaf(tree, leaf_node_id=new_children[0], model=model, train_inputs=inputs, split_threshold=0)
    assert res2 is None
