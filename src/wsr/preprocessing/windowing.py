"""Deterministic, label-independent time windowing (Phase 2, D-021 / invariant #19).

Two strictly separated steps:

1. `build_window_frame` - the CANDIDATE FRAME. Built from the recording
   duration and each stream's sampling rate only. Windows are half-open
   intervals [start, end) in seconds, anchored at synchronised time t = 0,
   non-overlapping, and never restarted at label boundaries. Incomplete
   tails are dropped from the frame and their duration is reported. Nothing
   in this step reads a label value.

2. `annotate_reference_labels` - attaches reference-label information to an
   EXISTING frame: codes present, homogeneity, binary eligibility and the
   analysis label. Mixed windows are never majority-voted; ineligible windows
   are never removed.

Each modality is sliced at its own native rate: a 60-s window covers
ACC 1920 x 3, BVP 3840, EDA 240, TEMP 240 samples and 42 000 label samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Primary binary reference (D-021): raw code -> analysis label.
BINARY_REFERENCE: dict[int, int] = {1: 0, 2: 1}  # 1 = baseline reference -> 0 ; 2 = protocol-stress reference -> 1
CONDITION_NAMES: dict[int, str] = {0: "transient", 1: "baseline", 2: "stress", 3: "amusement", 4: "meditation", 5: "reading", 6: "reading", 7: "reading"}


@dataclass(frozen=True)
class WindowSpec:
    length_s: float = 60.0
    step_s: float = 60.0
    origin_s: float = 0.0

    def __post_init__(self) -> None:
        if self.length_s <= 0 or self.step_s <= 0:
            raise ValueError("length_s and step_s must be positive")
        if self.origin_s != 0.0:
            raise ValueError("Phase-2 contract: windows are anchored at synchronised t=0 (origin_s must be 0)")


def sample_slice(start_s: float, end_s: float, rate_hz: float) -> tuple[int, int]:
    """Half-open sample index range [start, end) for a time interval at a given rate.

    Rounded to the nearest sample; for the Phase-2 grid (integer seconds,
    integer rates) the products are exact integers.
    """
    return int(round(start_s * rate_hz)), int(round(end_s * rate_hz))


def n_complete_windows(duration_s: float, spec: WindowSpec) -> int:
    if duration_s < spec.length_s:
        return 0
    return int(np.floor((duration_s - spec.length_s) / spec.step_s + 1e-9)) + 1


def build_window_frame(participant_id: str, duration_s: float, rates_hz: dict[str, float], label_rate_hz: float, spec: WindowSpec = WindowSpec(), dataset: str = "wesad") -> pd.DataFrame:
    """Candidate frame: one row per COMPLETE window, from time and rates only.

    Columns: provenance (dataset, participant_id, window_id, window_index,
    start_seconds, end_seconds, label sample range, per-modality sample
    ranges). No label value is consulted.
    """
    n = n_complete_windows(duration_s, spec)
    idx = np.arange(n)
    start = spec.origin_s + idx * spec.step_s
    end = start + spec.length_s
    frame = pd.DataFrame(
        {
            "dataset": dataset,
            "participant_id": participant_id,
            "window_id": [f"{dataset}:{participant_id}:w{int(i):05d}" for i in idx],
            "window_index": idx.astype(np.int32),
            "start_seconds": start.astype(np.float64),
            "end_seconds": end.astype(np.float64),
        }
    )
    ls, le = zip(*(sample_slice(s, e, label_rate_hz) for s, e in zip(start, end))) if n else ((), ())
    frame["start_label_sample"] = np.asarray(ls, dtype=np.int64)
    frame["end_label_sample"] = np.asarray(le, dtype=np.int64)
    for mod, rate in rates_hz.items():
        ms, me = zip(*(sample_slice(s, e, rate) for s, e in zip(start, end))) if n else ((), ())
        frame[f"{mod.lower()}_start_sample"] = np.asarray(ms, dtype=np.int64)
        frame[f"{mod.lower()}_end_sample"] = np.asarray(me, dtype=np.int64)
    return frame


def omitted_tail_s(duration_s: float, spec: WindowSpec) -> float:
    n = n_complete_windows(duration_s, spec)
    covered = (spec.origin_s + (n - 1) * spec.step_s + spec.length_s) if n else spec.origin_s
    return float(max(duration_s - covered, 0.0))


def annotate_reference_labels(frame: pd.DataFrame, label: np.ndarray, binary_reference: dict[int, int] = BINARY_REFERENCE) -> pd.DataFrame:
    """Attach reference-label information to an existing frame (boundaries untouched).

    Per window: raw_label_codes_present (sorted, comma-joined), n_label_samples,
    is_label_homogeneous, homogeneous_raw_label (nullable Int16), condition_name,
    binary_eligible, analysis_label (nullable Int8), ineligibility_reason.
    """
    label = np.asarray(label).ravel()
    out = frame.copy()
    codes_present, n_samp, homog, hlabel, cond, elig, alabel, reason = [], [], [], [], [], [], [], []
    for s, e in zip(out["start_label_sample"].to_numpy(), out["end_label_sample"].to_numpy()):
        seg = label[s:e]
        uniq = np.unique(seg)
        codes_present.append(",".join(str(int(c)) for c in uniq))
        n_samp.append(int(seg.size))
        is_h = uniq.size == 1 and seg.size == (e - s)
        homog.append(bool(is_h))
        h = int(uniq[0]) if is_h else None
        hlabel.append(h)
        cond.append(CONDITION_NAMES.get(h, "unknown") if is_h else "mixed")
        if is_h and h in binary_reference:
            elig.append(True)
            alabel.append(binary_reference[h])
            reason.append("")
        elif is_h:
            elig.append(False)
            alabel.append(None)
            reason.append(f"homogeneous_code_{h}_not_in_binary_reference")
        else:
            elig.append(False)
            alabel.append(None)
            reason.append("mixed_label_codes")
    out["raw_label_codes_present"] = codes_present
    out["n_label_samples"] = np.asarray(n_samp, dtype=np.int64)
    out["is_label_homogeneous"] = np.asarray(homog, dtype=bool)
    out["homogeneous_raw_label"] = pd.array(hlabel, dtype="Int16")
    out["condition_name"] = cond
    out["binary_eligible"] = np.asarray(elig, dtype=bool)
    out["analysis_label"] = pd.array(alabel, dtype="Int8")
    out["ineligibility_reason"] = reason
    return out


#: Columns produced by this module that must never be model features.
PROVENANCE_COLUMNS = (
    "dataset",
    "participant_id",
    "window_id",
    "window_index",
    "start_seconds",
    "end_seconds",
    "start_label_sample",
    "end_label_sample",
)
REFERENCE_COLUMNS = (
    "raw_label_codes_present",
    "n_label_samples",
    "is_label_homogeneous",
    "homogeneous_raw_label",
    "condition_name",
    "binary_eligible",
    "analysis_label",
    "ineligibility_reason",
)
