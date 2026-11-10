"""Unit and integration tests for paper artifacts, figure/table generation, and manuscript compilation."""

import os
from pathlib import Path
import pytest
import yaml

from src.utils.paper_artifacts import (
    DEFAULT_PAPER_METADATA,
    load_paper_metadata,
)


def test_load_paper_metadata(tmp_path: Path):
    # 1. Existing root yaml
    meta = load_paper_metadata("paper_metadata.yaml")
    assert isinstance(meta, dict)
    assert "MCTS-HNM" in meta["title"]
    assert "authors" in meta
    assert len(meta["authors"]) >= 1
    assert "PPGC" in meta["institution"] or "PPGC" in meta["authors"][0]["affiliation"]

    # 2. Custom path
    custom_yaml = tmp_path / "custom_meta.yaml"
    custom_data = {
        "title": "Custom Test Paper",
        "authors": [{"name": "Test Author", "affiliation": "Test Univ"}],
        "institution": "Test Inst",
        "keywords": ["Test", "RL"],
    }
    with open(custom_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(custom_data, f)

    loaded_custom = load_paper_metadata(custom_yaml)
    assert loaded_custom["title"] == "Custom Test Paper"
    assert loaded_custom["authors"][0]["name"] == "Test Author"

    # 3. Non-existent path falls back to default
    non_existent = tmp_path / "non_existent.yaml"
    fallback = load_paper_metadata(non_existent)
    assert fallback["title"] == DEFAULT_PAPER_METADATA["title"]
    assert "PPGC" in fallback["institution"]
