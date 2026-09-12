"""Unit tests for schema-independent audit helpers (synthetic inputs only)."""

import numpy as np
import pytest

from wsr.data.audit_common import (
    Run,
    code_summary,
    contiguous_runs,
    duration_from_samples,
    finiteness,
    missing_codes,
    unknown_codes,
)


def test_contiguous_runs_basic_and_ordering():
    labels = np.array([0, 0, 1, 1, 1, 2, 1, 0])
    runs = contiguous_runs(labels)
    assert runs == [
        Run(0, 0, 2, 2),
        Run(1, 2, 5, 3),
        Run(2, 5, 6, 1),
        Run(1, 6, 7, 1),
        Run(0, 7, 8, 1),
    ]
    # runs tile the vector exactly
    assert sum(r.n_samples for r in runs) == labels.size
    assert runs[0].start_idx == 0 and runs[-1].end_idx == labels.size


def test_contiguous_runs_edge_cases():
    assert contiguous_runs(np.array([])) == []
    assert contiguous_runs(np.array([5])) == [Run(5, 0, 1, 1)]
    assert contiguous_runs(np.array([3, 3, 3])) == [Run(3, 0, 3, 3)]


def test_run_duration_and_dict():
    r = Run(2, 700, 1400, 700)
    assert r.duration_s(700.0) == 1.0
    d = r.to_dict(700.0)
    assert d["start_s"] == 1.0 and d["end_s"] == 2.0 and d["duration_s"] == 1.0
    assert "duration_s" not in r.to_dict()


def test_code_summary_counts_durations_runs():
    labels = np.array([1, 1, 2, 2, 2, 1, 0])
    s = code_summary(labels, rate_hz=2.0)
    assert list(s) == [0, 1, 2]  # sorted codes
    assert s[1] == {"n_samples": 3, "duration_s": 1.5, "n_runs": 2, "run_durations_s": [1.0, 0.5]}
    assert s[2]["n_runs"] == 1 and s[2]["duration_s"] == 1.5
    assert s[0]["n_samples"] == 1


def test_unknown_and_missing_codes():
    documented = {0: "transient", 1: "baseline", 2: "stress"}
    labels = np.array([0, 1, 1, 7, 2])
    assert unknown_codes(labels, documented) == [7]
    assert missing_codes(labels, expected=[1, 2, 3, 4]) == [3, 4]
    assert unknown_codes(np.array([1, 2]), documented) == []


def test_finiteness_counts():
    arr = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
    assert finiteness(arr) == {"n": 5, "n_nan": 1, "n_inf": 2, "numeric": True}
    assert finiteness(np.array([]))["n"] == 0
    assert finiteness(np.array(["a", "b"]))["numeric"] is False


def test_duration_from_samples_rejects_bad_rate():
    assert duration_from_samples(1400, 700) == 2.0
    with pytest.raises(ValueError):
        duration_from_samples(10, 0)
