"""YAML configuration loading with base + dataset override merging.

`load_config("wesad")` returns base.yaml deep-merged with wesad.yaml. Keys in the
dataset file override keys in base; nested mappings are merged recursively;
lists are replaced, not concatenated (so a dataset config can narrow a list).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from wsr.utils.paths import CONFIGS


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def load_config(dataset: str | None = None, configs_dir: Path = CONFIGS) -> dict[str, Any]:
    """Load configs/base.yaml, optionally merged with configs/<dataset>.yaml."""
    base = load_yaml(Path(configs_dir) / "base.yaml")
    if dataset is None:
        return base
    override_path = Path(configs_dir) / f"{dataset}.yaml"
    if not override_path.is_file():
        raise FileNotFoundError(f"No config for dataset {dataset!r} at {override_path}")
    return _deep_merge(base, load_yaml(override_path))


def find_todos(cfg: dict[str, Any], prefix: str = "") -> list[str]:
    """Return dotted paths of unresolved (None-valued) config entries.

    Used to fail loudly if an experiment is run while a scientific decision is
    still open, instead of silently defaulting.
    """
    todos: list[str] = []
    for key, value in cfg.items():
        path = f"{prefix}.{key}" if prefix else key
        if value is None:
            todos.append(path)
        elif isinstance(value, dict):
            todos.extend(find_todos(value, path))
    return todos
