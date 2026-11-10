"""CIFAR-100 dataset loaders, taxonomy mappings, embedding caches, and scenario splitters."""

from src.data.cifar100_taxonomy import (
    CIFAR100_COARSE_CLASSES,
    CIFAR100_FINE_CLASSES,
    SUPERCLASS_TO_FINE_NAMES,
    CIFAR100Taxonomy,
    fine_to_coarse_tensor,
    get_cifar100_taxonomy,
)
from src.data.extract_embeddings import (
    EMBEDDING_DIM,
    build_cifar100_eval_transform,
    build_frozen_resnet18_extractor,
    create_stratified_train_val_indices,
    extract_cifar100_embeddings,
    extract_or_generate_embeddings,
    load_embedding_cache,
    validate_embedding_bundle,
)
from src.data.splits import (
    DEFAULT_SCENARIO1_TARGETS,
    IMBALANCE_RATIOS,
    Scenario1Split,
    Scenario2Split,
    build_scenario1_split,
    build_scenario2_split,
)

__all__ = [
    "CIFAR100_COARSE_CLASSES",
    "CIFAR100_FINE_CLASSES",
    "CIFAR100Taxonomy",
    "DEFAULT_SCENARIO1_TARGETS",
    "EMBEDDING_DIM",
    "IMBALANCE_RATIOS",
    "SUPERCLASS_TO_FINE_NAMES",
    "Scenario1Split",
    "Scenario2Split",
    "build_cifar100_eval_transform",
    "build_frozen_resnet18_extractor",
    "build_scenario1_split",
    "build_scenario2_split",
    "create_stratified_train_val_indices",
    "extract_cifar100_embeddings",
    "extract_or_generate_embeddings",
    "fine_to_coarse_tensor",
    "get_cifar100_taxonomy",
    "load_embedding_cache",
    "validate_embedding_bundle",
]
