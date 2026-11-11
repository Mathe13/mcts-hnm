"""Hierarchical tree representations and builders for MCTS-HNM."""

from src.trees.base_tree import FastHierarchyTree
from src.trees.confusion_tree import (
    build_confusion_tree,
    compute_confusion_distance_matrix,
    estimate_validation_confusion_matrix,
    validate_confusion_matrix,
)
from src.trees.hierarchy_metrics import compute_tree_structural_metrics
from src.trees.random_tree import build_random_tree
from src.trees.semantic_tree import build_semantic_tree

__all__ = [
    "FastHierarchyTree",
    "build_confusion_tree",
    "build_random_tree",
    "build_semantic_tree",
    "compute_confusion_distance_matrix",
    "compute_tree_structural_metrics",
    "estimate_validation_confusion_matrix",
    "validate_confusion_matrix",
]
