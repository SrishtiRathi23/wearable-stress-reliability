"""Schema-independent helpers for structural dataset audits.

Pure functions over NumPy arrays; no dataset-specific assumptions. Used by
audit_wesad.py and intended for reuse by later audits (Nurse, Stress-Predict).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Run:
    """A maximal contiguous run of one label code in a label vector."""

    code: int
    start_idx: int  # inclusive
    end_idx: int  # exclusive
    n_samples: int

    def duration_s(self, rate_hz: float) -> float:
        return self.n_samples / rate_hz

    def to_dict(self, rate_hz: float | None = None) -> dict:
        d = asdict(self)
        if rate_hz is not None:
            d["start_s"] = self.start_idx / rate_hz
            d["end_s"] = self.end_idx / rate_hz
            d["duration_s"] = self.duration_s(rate_hz)
        return d


def contiguous_runs(labels: np.ndarray) -> list[Run]:
    """Split a 1-D integer label vector into maximal contiguous runs, in order.

    An empty vector yields an empty list. Runs are returned in index order so
    the output is deterministic for a given input.
    """
    labels = np.asarray(labels).ravel()
    if labels.size == 0:
        return []
    change = np.flatnonzero(np.diff(labels) != 0) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [labels.size]))
    return [Run(int(labels[s]), int(s), int(e), int(e - s)) for s, e in zip(starts, ends)]


def code_summary(labels: np.ndarray, rate_hz: float) -> dict[int, dict]:
    """Per-code sample counts, durations and run counts, keyed by raw code (sorted)."""
    labels = np.asarray(labels).ravel()
    runs = contiguous_runs(labels)
    out: dict[int, dict] = {}
    for code in sorted(int(c) for c in np.unique(labels)):
        n = int((labels == code).sum())
        code_runs = [r for r in runs if r.code == code]
        out[code] = {
            "n_samples": n,
            "duration_s": n / rate_hz,
            "n_runs": len(code_runs),
            "run_durations_s": [r.duration_s(rate_hz) for r in code_runs],
        }
    return out


def unknown_codes(labels: np.ndarray, documented: dict[int, str]) -> list[int]:
    """Codes present in `labels` that are not in the documented code table."""
    present = {int(c) for c in np.unique(np.asarray(labels).ravel())}
    return sorted(present - set(documented))


def missing_codes(labels: np.ndarray, expected: list[int]) -> list[int]:
    """Expected codes that do not occur in `labels`."""
    present = {int(c) for c in np.unique(np.asarray(labels).ravel())}
    return sorted(set(expected) - present)


def finiteness(arr: np.ndarray) -> dict:
    """Counts of NaN / Inf and the total element count for a numeric array."""
    arr = np.asarray(arr)
    if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
        return {"n": int(arr.size), "n_nan": 0, "n_inf": 0, "numeric": bool(np.issubdtype(arr.dtype, np.number))}
    return {
        "n": int(arr.size),
        "n_nan": int(np.isnan(arr).sum()),
        "n_inf": int(np.isinf(arr).sum()),
        "numeric": True,
    }


def duration_from_samples(n_samples: int, rate_hz: float) -> float:
    if rate_hz <= 0:
        raise ValueError(f"rate_hz must be positive, got {rate_hz}")
    return n_samples / rate_hz
