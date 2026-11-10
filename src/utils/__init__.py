"""Shared utilities for reproducibility, metrics, and artifact export."""

from src.utils.paper_artifacts import (
    generate_ablation_table,
    generate_all_paper_artifacts,
    generate_convergence_plot,
    generate_hyperparameter_sensitivity_plot,
    generate_scenario1_table,
    generate_scenario2_table,
    generate_tree_visitation_heatmap,
    load_or_synthesize_benchmark_results,
    load_paper_metadata,
)
from src.utils.seed import get_torch_generator, set_global_seed
from src.utils.statistics import compute_paired_statistical_tests

__all__ = [
    "get_torch_generator",
    "set_global_seed",
    "compute_paired_statistical_tests",
    "load_paper_metadata",
    "load_or_synthesize_benchmark_results",
    "generate_convergence_plot",
    "generate_tree_visitation_heatmap",
    "generate_hyperparameter_sensitivity_plot",
    "generate_scenario1_table",
    "generate_scenario2_table",
    "generate_ablation_table",
    "generate_all_paper_artifacts",
]
