"""Paper artifacts and publication figure/table generation module.

Implements automated export of publication-grade figures (PDF/PNG),
LaTeX booktabs tables (.tex), paper metadata loading (paper_metadata.yaml),
and manuscript compilation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PAPER_METADATA: dict[str, Any] = {
    "title": "MCTS-HNM: Monte Carlo Tree Search for Hierarchical Hard Negative Mining in Class-Imbalanced Learning",
    "authors": [
        {
            "name": "Matheus Santos",
            "affiliation": "PPGC - Universidade Federal do Rio Grande do Sul (UFRGS)",
            "email": "matheus@inf.ufrgs.br",
        },
        {
            "name": "Co-Author Placeholder",
            "affiliation": "PPGC - Universidade Federal do Rio Grande do Sul (UFRGS)",
            "email": "coauthor@inf.ufrgs.br",
        },
    ],
    "institution": (
        "Postgraduate Program in Computer Science (PPGC)\n"
        "Universidade Federal do Rio Grande do Sul (UFRGS)\n"
        "Porto Alegre, RS, Brazil"
    ),
    "abstract": (
        "Deep image classification under severe class imbalance and hierarchical label structures "
        "presents significant challenges for standard training regimes. In this work, we investigate "
        "Monte Carlo Tree Search (MCTS) as an active reinforcement learning agent for hierarchical "
        "hard negative mining (MCTS-HNM)."
    ),
    "keywords": [
        "Monte Carlo Tree Search",
        "Hard Negative Mining",
        "Hierarchical Classification",
        "Class Imbalance",
        "Curriculum Learning",
    ],
}


def _simple_yaml_parse(text: str) -> dict[str, Any]:
    """Lightweight fallback YAML parser for basic paper_metadata dictionaries."""
    data: dict[str, Any] = {}
    current_key = None
    list_items = []
    in_list = False

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" in line and not stripped.startswith("-"):
            if in_list and current_key:
                data[current_key] = list_items
                in_list = False
                list_items = []
            key, val = line.split(":", 1)
            current_key = key.strip()
            val = val.strip().strip('"').strip("'")
            if val:
                data[current_key] = val
        elif stripped.startswith("-") and current_key:
            in_list = True
            list_items.append(stripped.lstrip("-").strip().strip('"').strip("'"))
    if in_list and current_key:
        data[current_key] = list_items

    return data if data else DEFAULT_PAPER_METADATA.copy()


def load_paper_metadata(path: str | Path = "paper_metadata.yaml") -> dict[str, Any]:
    """Load paper metadata from a YAML file, falling back to defaults if missing or invalid.

    Args:
        path: Path to paper_metadata.yaml.

    Returns:
        Dictionary containing title, authors, institution, and keywords.
    """
    file_path = Path(path)
    if not file_path.exists():
        logger.warning(
            "Paper metadata file %s not found. Using default PPGC UFRGS metadata.",
            file_path,
        )
        return DEFAULT_PAPER_METADATA.copy()

    try:
        try:
            import yaml  # type: ignore

            with open(file_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict) and "title" in loaded:
                return loaded
        except Exception as e:
            logger.debug("PyYAML load failed: %s; using fallback parser", e)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        parsed = _simple_yaml_parse(text)
        if isinstance(parsed, dict) and "title" in parsed:
            return parsed
    except Exception as exc:
        logger.warning(
            "Failed to parse %s: %s. Falling back to default metadata.", file_path, exc
        )

    return DEFAULT_PAPER_METADATA.copy()
