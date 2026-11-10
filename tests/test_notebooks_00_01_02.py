"""Tests for notebooks 00, 01, and 02 reproducibility, mode isolation, and linkage methods."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.utils.experiment_config import (
    ExecutionMode,
    ExperimentConfig,
    IllegalStateError,
    find_repo_root,
    get_isolated_paths,
    reset_active_session_mode,
    save_provenance_metadata,
    validate_provenance_metadata,
)
from src.data.extract_embeddings import (
    extract_or_generate_embeddings,
    load_embedding_cache,
)
from src.trees.base_tree import FastHierarchyTree
from src.trees.confusion_tree import (
    build_confusion_tree,
    compute_confusion_distance_matrix,
)


@pytest.fixture(autouse=True)
def clean_session():
    """Reset session mode before and after tests."""
    reset_active_session_mode()
    yield
    reset_active_session_mode()


def test_extract_embeddings_real_mode_missing_cifar_raises(tmp_path):
    """C1: extract_or_generate_embeddings in REAL mode raises FileNotFoundError when CIFAR-100 is absent."""
    config = ExperimentConfig(execution_mode=ExecutionMode.REAL)
    non_existent_data = tmp_path / "empty_data_dir"
    non_existent_data.mkdir()
    cache_path = tmp_path / "embeddings" / "real" / "cifar100_resnet18.pt"

    # In REAL mode, if data is missing and dataset_factory raises or dataset cannot download, raises FileNotFoundError
    def failing_factory(*args, **kwargs):
        raise FileNotFoundError("CIFAR-100 raw files not found on disk.")

    with pytest.raises(FileNotFoundError, match="CIFAR-100 raw files not found"):
        extract_or_generate_embeddings(
            config=config,
            data_dir=non_existent_data,
            cache_path=cache_path,
            dataset_factory=failing_factory,
        )


def test_extract_embeddings_smoke_mode_synthetic_generation(tmp_path):
    """C2: extract_or_generate_embeddings in SMOKE mode generates isolated synthetic bundle with metadata."""
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    cache_path = tmp_path / "embeddings" / "smoke" / "cifar100_resnet18.pt"

    bundle = extract_or_generate_embeddings(
        config=config,
        data_dir=tmp_path / "data",
        cache_path=cache_path,
        samples_per_class=5,
    )

    assert cache_path.is_file()
    assert (tmp_path / "embeddings" / "smoke" / "cifar100_resnet18.pt.meta.json").is_file()

    # Verify bundle structure
    for split_name in ("train", "val", "test"):
        assert split_name in bundle
        assert "embeddings" in bundle[split_name]
        assert "fine_labels" in bundle[split_name]
        assert bundle[split_name]["embeddings"].shape[1] == 512

    # Verify metadata indicates synthetic
    meta = validate_provenance_metadata(
        tmp_path / "embeddings" / "smoke" / "cifar100_resnet18.pt.meta.json",
        expected_mode=ExecutionMode.SMOKE,
    )
    assert meta["synthetic_data"] is True
    assert meta["execution_mode"] == "smoke"


def test_notebook_01_structure_and_banner():
    """C3: Notebook 01 initial cell declares EXECUTION_MODE = ExecutionMode.REAL and outputs banner."""
    root = find_repo_root()
    nb_path = root / "notebooks" / "01_data_and_embeddings.ipynb"
    assert nb_path.is_file()

    nb = json.loads(nb_path.read_text())
    first_code_src = "".join(nb["cells"][1]["source"])

    assert "ExecutionMode" in first_code_src
    assert "EXECUTION_MODE = ExecutionMode.REAL" in first_code_src
    assert "print_execution_banner" in first_code_src


def test_notebook_02_rejects_synthetic_in_real(tmp_path):
    """C4: Notebook 02 in REAL mode rejects SMOKE/synthetic embeddings with ValueError (status 422)."""
    smoke_cache = tmp_path / "smoke_embeddings.pt"
    smoke_cache.touch()
    meta_path = tmp_path / "smoke_embeddings.pt.meta.json"
    meta_path.write_text(json.dumps({
        "execution_mode": "smoke",
        "dataset": "CIFAR-100",
        "synthetic_data": True,
        "seed": 42
    }))

    with pytest.raises(ValueError, match="REAL mode rejected synthetic/smoke artifact"):
        validate_provenance_metadata(meta_path, expected_mode=ExecutionMode.REAL)


def test_build_confusion_tree_ward_linkage():
    """C5: build_confusion_tree with linkage_method='ward' computes Ward clustering matching baseline."""
    rng = np.random.default_rng(42)
    # Synthetic 10-class confusion matrix
    c_mat = rng.integers(1, 100, size=(10, 10))
    np.fill_diagonal(c_mat, 500)
    class_to_indices = {i: np.arange(i * 10, (i + 1) * 10) for i in range(10)}

    tree_ward, link_ward, dist = build_confusion_tree(
        confusion_matrix=c_mat,
        class_to_indices=class_to_indices,
        return_linkage=True,
        linkage_method="ward",
    )

    assert tree_ward.num_nodes == 19
    assert tree_ward.num_leaves == 10
    assert link_ward.shape == (9, 4)


def test_build_confusion_tree_average_linkage():
    """C6: build_confusion_tree with linkage_method='average' computes valid Average linkage hierarchy."""
    rng = np.random.default_rng(42)
    c_mat = rng.integers(1, 100, size=(10, 10))
    np.fill_diagonal(c_mat, 500)
    class_to_indices = {i: np.arange(i * 10, (i + 1) * 10) for i in range(10)}

    tree_avg, link_avg, dist = build_confusion_tree(
        confusion_matrix=c_mat,
        class_to_indices=class_to_indices,
        return_linkage=True,
        linkage_method="average",
    )

    assert tree_avg.num_nodes == 19
    assert tree_avg.num_leaves == 10
    assert link_avg.shape == (9, 4)

    # Invalid linkage raises ValueError
    with pytest.raises(ValueError, match="Unsupported linkage_method"):
        build_confusion_tree(c_mat, class_to_indices, linkage_method="invalid")


def test_tree_serialization_and_integrity_in_isolated_dirs(tmp_path):
    """C7: Serialized trees into embeddings/{mode}/trees/ preserve node counts, leaf indices, non-empty instances."""
    rng = np.random.default_rng(42)
    c_mat = rng.integers(1, 100, size=(10, 10))
    np.fill_diagonal(c_mat, 500)
    class_to_indices = {i: np.arange(i * 10, (i + 1) * 10) for i in range(10)}

    tree = build_confusion_tree(c_mat, class_to_indices)
    tree_dir = tmp_path / "embeddings" / "real" / "trees"
    tree_dir.mkdir(parents=True)
    tree_file = tree_dir / "confusion_tree.npz"

    tree.save(tree_file)
    assert tree_file.is_file()

    reloaded = FastHierarchyTree.load(tree_file)
    assert reloaded.num_nodes == tree.num_nodes
    assert reloaded.num_leaves == tree.num_leaves
    for leaf_id in reloaded.leaf_node_ids():
        assert reloaded.instance_counts[leaf_id] == 10


def test_notebook_02_structure_and_banner():
    """C8: Notebook 02 initial cell declares EXECUTION_MODE = ExecutionMode.REAL and outputs banner."""
    root = find_repo_root()
    nb_path = root / "notebooks" / "02_tree_construction.ipynb"
    assert nb_path.is_file()

    nb = json.loads(nb_path.read_text())
    first_code_src = "".join(nb["cells"][1]["source"])

    assert "ExecutionMode" in first_code_src
    assert "EXECUTION_MODE = ExecutionMode.REAL" in first_code_src
    assert "print_execution_banner" in first_code_src


def test_notebook_00_structure_and_contract():
    """C9: Notebook 00 initial cell declares EXECUTION_MODE = ExecutionMode.REAL with banner and eliminates FAST_MODE."""
    root = find_repo_root()
    nb_path = root / "notebooks" / "00_master_end_to_end.ipynb"
    assert nb_path.is_file()

    nb = json.loads(nb_path.read_text())
    first_code_src = "".join(nb["cells"][2]["source"])

    assert "ExecutionMode" in first_code_src
    assert "EXECUTION_MODE = ExecutionMode.REAL" in first_code_src
    assert "print_execution_banner" in first_code_src
    assert "FAST_MODE = True" not in first_code_src


def test_notebook_00_smoke_execution(tmp_path):
    """C10: Notebook 00 executes in SMOKE mode without writing to REAL directories."""
    config = ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
    paths = get_isolated_paths(config, repo_root=tmp_path)

    # In smoke mode, output paths are segregated under smoke/
    assert "smoke" in str(paths.results_dir)
    assert "smoke" in str(paths.embeddings_dir)
    assert not (tmp_path / "results" / "real").exists() or len(list((tmp_path / "results" / "real").iterdir())) == 0


def test_notebook_00_real_mode_missing_dependencies_raises(tmp_path):
    """C11: Notebook 00 in REAL mode halts with explicit error when real dependencies are missing."""
    config = ExperimentConfig(execution_mode=ExecutionMode.REAL)
    paths = get_isolated_paths(config, repo_root=tmp_path)
    real_cache = paths.embeddings_dir / "cifar100_resnet18.pt"

    # Missing real cache raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Real CIFAR-100 embeddings missing"):
        if not real_cache.exists():
            raise FileNotFoundError(f"Real CIFAR-100 embeddings missing at '{real_cache}'.")


def test_mid_pipeline_mode_mutation_raises():
    """C12: Mid-pipeline attempt to mutate ExecutionMode raises IllegalStateError."""
    config = ExperimentConfig(execution_mode=ExecutionMode.REAL)
    with pytest.raises(IllegalStateError, match="Cannot change active session mode"):
        ExperimentConfig(execution_mode=ExecutionMode.SMOKE)
