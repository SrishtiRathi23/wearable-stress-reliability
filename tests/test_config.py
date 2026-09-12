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


def test_wesad_config_open_and_resolved_decisions():
    # Documents the current state: design TODOs still open, audit facts filled in.
    cfg = load_config("wesad")
    todos = find_todos(cfg)
    # still open
    assert "evaluation.outer_split" in todos  # T-04: no executable default may masquerade as agreed
    assert "episodes.definition" in todos  # T-05
    assert "observation_policies.budget_unit" in todos and "observation_policies.detector" in todos  # T-06, T-07
    # frozen by D-021
    assert cfg["device"] == "wrist" and "device" not in todos  # T-01
    assert cfg["labels"]["raw_codes"][2] == "stress" and cfg["labels"]["positive_code"] == 2 and cfg["labels"]["negative_code"] == 1
    assert cfg["labels"]["ineligible_codes"] == [0, 3, 4, 5, 6, 7] and cfg["labels"]["preserve_raw_label"] is True
    assert cfg["windowing"]["anchor"] == "pickle_t0" and cfg["windowing"]["grid_uses_labels"] is False  # invariant #19
    assert cfg["windowing"]["eligibility_rule"] == "homogeneous_full_window"
    assert cfg["windowing"]["mixed_window_policy"] == "keep_in_provenance_mark_ineligible"
    assert len(cfg["participants"]["all"]) == 15 and cfg["participants"]["excluded"] == []
    assert cfg["features"]["hrv"]["enabled"] is False
