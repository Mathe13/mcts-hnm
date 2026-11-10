"""Training package for supervised classifiers, baselines, and MCTS training loops."""

from src.training.baselines import OHEMSampler, UniformSampler
from src.training.benchmark_runner import (
    BenchmarkRunResult,
    audit_scenario1_imbalance,
    run_benchmark,
    run_sensitivity_ablation,
)
from src.training.trainer import (
    evaluate_scenario1_metrics,
    evaluate_scenario2_metrics,
    train_scenario1,
    train_scenario2,
)

__all__ = [
    "UniformSampler",
    "OHEMSampler",
    "evaluate_scenario1_metrics",
    "evaluate_scenario2_metrics",
    "train_scenario1",
    "train_scenario2",
    "BenchmarkRunResult",
    "audit_scenario1_imbalance",
    "run_benchmark",
    "run_sensitivity_ablation",
]
