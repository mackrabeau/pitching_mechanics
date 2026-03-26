"""Centralized path and configuration management for PitchLens."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """Resolve the project root directory.

    Checks PITCHLENS_ROOT env var first, then walks up to find pyproject.toml.
    """
    env = os.environ.get("PITCHLENS_ROOT")
    if env:
        return Path(env).resolve()

    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent

    return current.parent


def get_obp_data_root(project_root: Path | None = None) -> Path:
    """Path to openbiomechanics/baseball_pitching/data/."""
    root = project_root or get_project_root()
    return root / "openbiomechanics" / "baseball_pitching" / "data"


def get_logs_dir(project_root: Path | None = None) -> Path:
    """Path to pitchlens/logs/, created lazily."""
    root = project_root or get_project_root()
    d = root / "pitchlens" / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_figures_dir(project_root: Path | None = None) -> Path:
    """Path to pitchlens/logs/figures/, created lazily."""
    d = get_logs_dir(project_root) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_models_dir(project_root: Path | None = None) -> Path:
    """Path to pitchlens/models/, created lazily."""
    root = project_root or get_project_root()
    d = root / "pitchlens" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d
