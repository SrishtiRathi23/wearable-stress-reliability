"""Config loading: base + dataset merge, and TODO detection."""

from pathlib import Path

import pytest

from wsr.utils.config import _deep_merge, find_todos, load_config


def test_deep_merge_overrides_scalars_merges_dicts_replaces_lists():
    base = {"a": 1, "nest": {"x": 1, "y": 2}, "lst": [1, 2, 3]}
    over = {"a": 2, "nest": {"y": 20, "z": 30}, "lst": [9]}
    merged = _deep_merge(base, over)
    assert merged == {"a": 2, "nest": {"x": 1, "y": 20, "z": 30}, "lst": [9]}
    # inputs untouched
    assert base["nest"] == {"x": 1, "y": 2}


def test_repo_configs_load_and_merge():
    base = load_config()
    assert base["seed"] == 42
    assert base["windowing"]["length_s"] == 60
    wesad = load_config("wesad")
    assert wesad["dataset"]["name"] == "wesad"
    assert wesad["seed"] == 42  # inherited from base
    nurse = load_config("nurse")
    assert nurse["labels"]["unknown_is_negative"] is False


def test_missing_dataset_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("no_such_dataset")


def test_find_todos_reports_unresolved_paths():
    cfg = {"a": None, "b": {"c": None, "d": 1}, "e": 2}
    assert find_todos(cfg) == ["a", "b.c"]


def test_wesad_config_still_has_open_decisions():
    # Documents the current state: these must be resolved before modelling.
    todos = find_todos(load_config("wesad"))
    assert "device" in todos
    assert "labels.raw_codes" in todos
