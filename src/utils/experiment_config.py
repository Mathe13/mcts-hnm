"""Centralized experiment configuration, execution mode enforcement, banners, and provenance isolation."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ExecutionMode(str, Enum):
    """Global execution mode for all notebooks and experimental pipelines."""
    SMOKE = "smoke"
    REAL = "real"


class IllegalStateError(RuntimeError):
    """Raised when an illegal state or mode transition is attempted during an execution session."""
    pass


_ACTIVE_SESSION_MODE: Optional[ExecutionMode] = None


def set_active_session_mode(mode: Union[ExecutionMode, str]) -> ExecutionMode:
    """Register or validate the active session execution mode."""
    global _ACTIVE_SESSION_MODE
    if isinstance(mode, str):
        try:
            mode = ExecutionMode(mode.lower())
        except ValueError:
            raise ValueError(f"Invalid execution mode: '{mode}'. Must be 'smoke' or 'real'.")

    if _ACTIVE_SESSION_MODE is not None and _ACTIVE_SESSION_MODE != mode:
        raise IllegalStateError(
            f"Cannot change active session mode from {_ACTIVE_SESSION_MODE.value.upper()} "
            f"to {mode.value.upper()} during execution. Execution mode must be defined once per process."
        )

    _ACTIVE_SESSION_MODE = mode
    return _ACTIVE_SESSION_MODE


def get_active_session_mode() -> Optional[ExecutionMode]:
    """Retrieve the currently active session execution mode, if set."""
    return _ACTIVE_SESSION_MODE


def reset_active_session_mode() -> None:
    """Reset the active session execution mode (used in test fixtures)."""
    global _ACTIVE_SESSION_MODE
    _ACTIVE_SESSION_MODE = None


def find_repo_root(start_dir: Optional[Union[str, Path]] = None) -> Path:
    """Locate repository root by ascending from start_dir looking for pyproject.toml or .git."""
    cur = Path(start_dir).resolve() if start_dir is not None else Path(__file__).resolve().parent
    while cur != cur.parent:
        if (cur / "pyproject.toml").is_file() or (cur / ".git").is_dir():
            return cur
        cur = cur.parent
    # Fallback to current working directory
    return Path.cwd().resolve()


@dataclass
class ExperimentConfig:
    """Validated configuration for scientific experiments and smoke tests."""
    execution_mode: ExecutionMode = ExecutionMode.REAL
    seed: Union[int, List[int]] = 42
    dataset_name: str = "CIFAR-100"
    dataset_source: str = "cifar100_resnet18.pt"
    dataset_fingerprint: Optional[str] = None
    num_steps: int = 100
    eval_every: int = 10
    output_dir: Union[str, Path] = "results"
    cache_dir: Union[str, Path] = "embeddings"
    device: str = "cpu"
    config_version: str = "1.0.0"
    synthetic_fallback_enabled: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.execution_mode, str):
            try:
                self.execution_mode = ExecutionMode(self.execution_mode.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid execution mode: '{self.execution_mode}'. Must be 'smoke' or 'real'."
                )

        if not isinstance(self.execution_mode, ExecutionMode):
            raise ValueError(
                f"Invalid execution mode: '{self.execution_mode}'. Must be an ExecutionMode enum."
            )

        if self.execution_mode == ExecutionMode.REAL:
            if self.synthetic_fallback_enabled:
                raise ValueError("Synthetic fallback cannot be enabled in REAL mode.")
            self.synthetic_fallback_enabled = False

        # Register with process session
        set_active_session_mode(self.execution_mode)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentConfig:
        """Construct validated ExperimentConfig from a dictionary."""
        config_data = dict(data)
        if "execution_mode" in config_data:
            mode_val = config_data["execution_mode"]
            if isinstance(mode_val, str):
                try:
                    config_data["execution_mode"] = ExecutionMode(mode_val.lower())
                except ValueError:
                    raise ValueError(
                        f"Invalid execution mode: '{mode_val}'. Must be 'smoke' or 'real'."
                    )
        return cls(**config_data)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a serializable dictionary."""
        res = asdict(self)
        res["execution_mode"] = self.execution_mode.value
        res["output_dir"] = str(self.output_dir)
        res["cache_dir"] = str(self.cache_dir)
        return res


@dataclass
class IsolatedPaths:
    """Directory paths segregated strictly by execution mode."""
    mode: ExecutionMode
    repo_root: Path
    embeddings_dir: Path
    results_dir: Path
    checkpoints_dir: Path
    paper_figures_dir: Path
    paper_tables_dir: Path
    legacy_dir: Path


def migrate_legacy_results(repo_root: Optional[Union[str, Path]] = None) -> int:
    """Explicitly move unsegregated legacy files from results/ to results/legacy/."""
    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()
    base_results = root / "results"
    legacy_dir = base_results / "legacy"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    moved_count = 0
    if base_results.is_dir():
        for item in base_results.iterdir():
            if item.is_file():
                dest = legacy_dir / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
                    moved_count += 1
    return moved_count


def get_isolated_paths(
    config: ExperimentConfig,
    repo_root: Optional[Union[str, Path]] = None,
    migrate_legacy: bool = False,
) -> IsolatedPaths:
    """Resolve and create isolated directory paths anchored to repository root."""
    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()
    mode = config.execution_mode

    embeddings_dir = root / "embeddings" / mode.value
    results_dir = root / "results" / mode.value
    checkpoints_dir = root / "checkpoints" / mode.value
    legacy_dir = root / "results" / "legacy"

    if mode == ExecutionMode.REAL:
        paper_figures_dir = root / "paper" / "figures"
        paper_tables_dir = root / "paper" / "tables"
    else:
        paper_figures_dir = root / "results" / "smoke" / "paper" / "figures"
        paper_tables_dir = root / "results" / "smoke" / "paper" / "tables"

    # Create directories
    for d in (
        embeddings_dir,
        results_dir,
        checkpoints_dir,
        legacy_dir,
        paper_figures_dir,
        paper_tables_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)

    # Migrate any legacy files located directly under results/ to results/legacy/
    if migrate_legacy:
        migrate_legacy_results(root)

    return IsolatedPaths(
        mode=mode,
        repo_root=root,
        embeddings_dir=embeddings_dir,
        results_dir=results_dir,
        checkpoints_dir=checkpoints_dir,
        paper_figures_dir=paper_figures_dir,
        paper_tables_dir=paper_tables_dir,
        legacy_dir=legacy_dir,
    )


def validate_mode_isolated_path(
    path: Union[str, Path],
    mode: ExecutionMode,
    root: Optional[Path] = None,
) -> Path:
    """Validate that path is properly segregated and does not cross execution mode boundaries.

    Raises:
        ValueError: If a SMOKE run attempts to access root paper/ or REAL paths, or if unisolated results/ is targeted.
    """
    repo = root or find_repo_root()
    resolved = Path(path).resolve()

    # Reject unsegregated root results/ (must be results/smoke or results/real or results/legacy)
    if resolved == (repo / "results").resolve() or resolved.name == "results":
        raise ValueError(
            f"Direct unsegregated I/O at '{path}' is prohibited. Use mode-segregated paths ('results/{mode.value}')."
        )

    # In SMOKE mode, reject root paper/ directory and REAL results
    if mode == ExecutionMode.SMOKE:
        is_smoke_paper = any(
            resolved.parts[i] == "results" and resolved.parts[i + 1] == "smoke" and resolved.parts[i + 2] == "paper"
            for i in range(len(resolved.parts) - 2)
        )
        if not is_smoke_paper and (
            resolved == (repo / "paper").resolve()
            or (repo / "paper").resolve() in resolved.parents
            or "paper" in resolved.parts
        ):
            raise ValueError(
                f"SMOKE mode prohibited from reading or writing root paper/ directory: '{path}'. Use 'results/smoke/paper/'."
            )
        if (
            resolved == (repo / "results" / "real").resolve()
            or (repo / "results" / "real").resolve() in resolved.parents
            or any(resolved.parts[i] == "results" and resolved.parts[i + 1] == "real" for i in range(len(resolved.parts) - 1))
        ):
            raise ValueError(
                f"SMOKE mode prohibited from accessing REAL results directory: '{path}'."
            )

    # In REAL mode, reject smoke results directory
    if mode == ExecutionMode.REAL:
        if (
            resolved == (repo / "results" / "smoke").resolve()
            or (repo / "results" / "smoke").resolve() in resolved.parents
            or any(resolved.parts[i] == "results" and resolved.parts[i + 1] == "smoke" for i in range(len(resolved.parts) - 1))
        ):
            raise ValueError(
                f"REAL mode prohibited from accessing SMOKE results directory: '{path}'."
            )

    # Any path located under results/ must be segregated into the active mode or legacy
    for i in range(len(resolved.parts) - 1):
        if resolved.parts[i] == "results":
            child = resolved.parts[i + 1]
            if child not in (mode.value, "legacy"):
                raise ValueError(
                    f"Path '{path}' inside results/ is not properly segregated. Must be under 'results/{mode.value}' or 'results/legacy'."
                )

    return resolved


def print_execution_banner(config: ExperimentConfig, notebook_name: str = "") -> str:
    """Print and return a standardized prominent warning banner."""
    paths = get_isolated_paths(config)
    lines: List[str] = []

    if config.execution_mode == ExecutionMode.SMOKE:
        lines = [
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
            "!!! SMOKE TEST — SYNTHETIC/TEST EXECUTION !!!",
            "!!! NOT VALID FOR SCIENTIFIC REPORTING    !!!",
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
            f"Notebook: {notebook_name}" if notebook_name else "Execution: Pipeline SMOKE",
            "Mode: SMOKE",
            f"Output directory: {paths.results_dir}",
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
        ]
    else:
        lines = [
            "==============================================",
            "EXPERIMENT MODE: REAL",
            f"Dataset: {config.dataset_name}",
            "Synthetic fallback: DISABLED",
            "Scientific results: ENABLED",
            "==============================================",
            f"Notebook: {notebook_name}" if notebook_name else "Execution: Pipeline REAL",
            "Mode: REAL",
            f"Output directory: {paths.results_dir}",
            "==============================================",
        ]

    banner = "\n".join(lines)
    print(banner)
    return banner


def get_git_commit(repo_root: Optional[Path] = None) -> str:
    """Retrieve current git commit hash if available."""
    try:
        root = repo_root or find_repo_root()
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "uncommitted"


def save_provenance_metadata(
    target_file_or_dir: Union[str, Path],
    config: ExperimentConfig,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist JSON provenance sidecar metadata for a generated artifact."""
    target_path = Path(target_file_or_dir).resolve()
    if target_path.is_dir():
        meta_path = target_path / "metadata.json"
    else:
        meta_path = target_path.parent / f"{target_path.name}.meta.json"

    meta_path.parent.mkdir(parents=True, exist_ok=True)

    root = find_repo_root()
    metadata: Dict[str, Any] = {
        "execution_mode": config.execution_mode.value,
        "dataset": config.dataset_name,
        "data_source": config.dataset_source,
        "seed": config.seed,
        "synthetic_data": (config.execution_mode == ExecutionMode.SMOKE),
        "num_steps": config.num_steps,
        "git_commit": get_git_commit(root),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_version": config.config_version,
    }

    if extra:
        metadata.update(extra)

    # In REAL mode, synthetic_data cannot be true under any circumstances
    if config.execution_mode == ExecutionMode.REAL and metadata.get("synthetic_data") is True:
        raise ValueError("Cannot persist metadata with synthetic_data=True in REAL mode.")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return meta_path


def validate_provenance_metadata(
    metadata_path: Union[str, Path],
    expected_mode: Optional[Union[ExecutionMode, str]] = None,
) -> Dict[str, Any]:
    """Validate that artifact metadata matches expectations and prohibits synthetic data in REAL mode."""
    path = Path(metadata_path).resolve()
    if not path.is_file():
        mode_suffix = (
            f" in {expected_mode.value.upper()} mode"
            if expected_mode and isinstance(expected_mode, ExecutionMode)
            else (f" in {str(expected_mode).upper()} mode" if expected_mode else "")
        )
        raise FileNotFoundError(f"Missing mandatory provenance metadata{mode_suffix}: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid provenance metadata content (expected dict): {path}")

    if "execution_mode" not in data:
        raise ValueError(f"Provenance metadata missing 'execution_mode': {path}")

    if expected_mode is not None:
        exp_mode_str = expected_mode.value if isinstance(expected_mode, ExecutionMode) else str(expected_mode).lower()
        actual_mode_str = str(data.get("execution_mode", "")).lower()

        if exp_mode_str == ExecutionMode.REAL.value:
            if data.get("synthetic_data") is True or actual_mode_str == ExecutionMode.SMOKE.value:
                raise ValueError(
                    f"REAL mode rejected synthetic/smoke artifact: {path} (synthetic_data={data.get('synthetic_data')}, mode={actual_mode_str})"
                )
            if actual_mode_str != ExecutionMode.REAL.value:
                raise ValueError(f"Expected REAL mode in metadata but found: {actual_mode_str}")
        elif exp_mode_str == ExecutionMode.SMOKE.value:
            if actual_mode_str == ExecutionMode.REAL.value:
                raise ValueError(
                    f"SMOKE mode rejected REAL artifact: {path} (mode={actual_mode_str})"
                )
            if actual_mode_str != ExecutionMode.SMOKE.value:
                raise ValueError(f"Expected SMOKE mode in metadata but found: {actual_mode_str}")

    return data
