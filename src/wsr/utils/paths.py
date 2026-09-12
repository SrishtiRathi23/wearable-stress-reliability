"""Project path resolution.

All code refers to locations through this module so that the repository can be
moved or run from any working directory without editing scripts.
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root (the directory containing pyproject.toml).

    Resolved relative to this file rather than the current working directory,
    so it is stable for scripts, tests and notebooks alike.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Could not locate project root: no pyproject.toml above wsr/utils/paths.py")


ROOT = project_root()
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_INTERIM = DATA / "interim"
DATA_PROCESSED = DATA / "processed"
MANIFESTS = DATA / "manifests"
RESULTS = ROOT / "results"
RESULTS_TABLES = RESULTS / "tables"
RESULTS_FIGURES = RESULTS / "figures"
RESULTS_PREDICTIONS = RESULTS / "predictions"
RESULTS_MODELS = RESULTS / "models"
RESULTS_LOGS = RESULTS / "logs"


def raw_dataset_dir(name: str) -> Path:
    """Directory holding the unmodified raw release of a dataset."""
    return DATA_RAW / name
