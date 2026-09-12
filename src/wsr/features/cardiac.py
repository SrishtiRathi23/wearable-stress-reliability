"""Wrist BVP (64 Hz) window features with a conservative in-house pulse-peak detector.

HRV is DISABLED (D-021). Only raw BVP statistics and simple heart-rate
summaries from plausible inter-beat intervals are produced.

Recording-level step (`detect_pulse_peaks`, once per participant, label-free):
zero-phase Butterworth band-pass (default 0.5-8 Hz, order 3) then
scipy.signal.find_peaks with a minimum distance (default 0.33 s ~ 180 bpm)
and a prominence floor relative to the recording's robust amplitude
(default 0.2 x MAD-based scale). This is a simple, transparent detector -
NOT a validated PPG beat algorithm; its outputs are provisional and subject
to feature-quality review before any modelling use.

PROVISIONAL STATUS (Phase-2 diagnostic, docs/known_issues.md KI-21): on the
real release the detector yields ~30-45 % successive beat-to-beat HR changes
> 20 bpm, so beat/HR columns are written with role "feature_provisional"
(model_feature = False) until a validated detector or an IBI-consistency gate
is approved. Raw BVP statistics remain ordinary features.

Window step (`compute_bvp_features`): raw statistics always; HR summaries
only from inter-beat intervals whose both beats lie inside the window and
whose length is within ibi_valid_range_s; when fewer than min_beats_for_hr
valid beats or coverage below min_beat_coverage, HR features are null and
q_bvp_hr_available is False. Values are never fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from wsr.features.schema import ColumnDef, feature


@dataclass(frozen=True)
class BvpParams:
    bandpass_hz: tuple[float, float] = (0.5, 8.0)
    filter_order: int = 3
    min_peak_distance_s: float = 0.33
    prominence_scale: float = 0.2  # provisional: fraction of robust amplitude scale
    ibi_valid_range_s: tuple[float, float] = (0.33, 2.0)  # 30-180 bpm
    min_beats_for_hr: int = 10  # provisional
    min_beat_coverage: float = 0.5  # provisional: fraction of window spanned by valid IBIs


@dataclass
class PulsePeaks:
    peak_idx: np.ndarray  # sample indices (recording timeline)
    ok: bool
    error: str = ""
    prominence_threshold: float = float("nan")


def detect_pulse_peaks(bvp: np.ndarray, rate_hz: float, params: BvpParams = BvpParams()) -> PulsePeaks:
    x = np.asarray(bvp, dtype=np.float64)
    try:
        if x.ndim != 1 or x.size < 8 * params.filter_order + 1:
            raise ValueError(f"BVP recording too short for filtering: {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("BVP contains non-finite values")
        sos = signal.butter(params.filter_order, params.bandpass_hz, btype="band", fs=rate_hz, output="sos")
        y = signal.sosfiltfilt(sos, x)
        mad = float(np.median(np.abs(y - np.median(y))))
        scale = 1.4826 * mad
        if scale <= 0:
            raise ValueError("BVP has zero robust amplitude (constant signal)")
        thr = params.prominence_scale * scale
        peaks, _ = signal.find_peaks(y, distance=max(1, int(round(params.min_peak_distance_s * rate_hz))), prominence=thr)
        return PulsePeaks(peaks.astype(np.int64), True, "", thr)
    except Exception as exc:  # noqa: BLE001
        return PulsePeaks(np.zeros(0, dtype=np.int64), False, repr(exc))


def _iqr(v: np.ndarray) -> float:
    q75, q25 = np.percentile(v, [75, 25])
    return float(q75 - q25)


FEATURES: list[ColumnDef] = [
    feature("bvp_mean", "bvp", "Mean raw BVP", "a.u."),
    feature("bvp_std", "bvp", "Std raw BVP", "a.u."),
    feature("bvp_median", "bvp", "Median raw BVP", "a.u."),
    feature("bvp_iqr", "bvp", "Interquartile range raw BVP", "a.u."),
    feature("bvp_min", "bvp", "Min raw BVP", "a.u."),
    feature("bvp_max", "bvp", "Max raw BVP", "a.u."),
    feature("bvp_range", "bvp", "Max - min raw BVP", "a.u."),
    feature("bvp_beat_count", "bvp", "Detected pulse peaks whose index lies in the window", "count", nullable=True, missing_reason="pulse detection failed for the participant", provisional=True),
    feature("bvp_valid_ibi_count", "bvp", "Inter-beat intervals with both beats in the window and length within ibi_valid_range_s", "count", nullable=True, missing_reason="pulse detection failed", provisional=True),
    feature("bvp_beat_coverage", "bvp", "Sum of valid IBIs / window length", "fraction", nullable=True, missing_reason="pulse detection failed", provisional=True),
    feature("hr_mean", "bvp", "Mean of 60/IBI over valid IBIs", "bpm", nullable=True, missing_reason="too few valid beats or coverage (q_bvp_hr_available False)", provisional=True),
    feature("hr_median", "bvp", "Median instantaneous HR", "bpm", nullable=True, missing_reason="q_bvp_hr_available False", provisional=True),
    feature("hr_std", "bvp", "Std instantaneous HR", "bpm", nullable=True, missing_reason="q_bvp_hr_available False", provisional=True),
    feature("hr_min", "bvp", "Min instantaneous HR", "bpm", nullable=True, missing_reason="q_bvp_hr_available False", provisional=True),
    feature("hr_max", "bvp", "Max instantaneous HR", "bpm", nullable=True, missing_reason="q_bvp_hr_available False", provisional=True),
]


def compute_bvp_features(bvp_win: np.ndarray, rate_hz: float, peaks: PulsePeaks | None, start: int, end: int, params: BvpParams = BvpParams()) -> tuple[dict[str, float], bool]:
    """Returns (features, hr_available)."""
    x = np.asarray(bvp_win, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError(f"BVP window must be 1-D with >= 2 samples; got {x.shape}")
    out: dict[str, float] = {
        "bvp_mean": float(x.mean()),
        "bvp_std": float(x.std()),
        "bvp_median": float(np.median(x)),
        "bvp_iqr": _iqr(x),
        "bvp_min": float(x.min()),
        "bvp_max": float(x.max()),
        "bvp_range": float(x.max() - x.min()),
    }
    nan = float("nan")
    hr_keys = ("hr_mean", "hr_median", "hr_std", "hr_min", "hr_max")
    if peaks is None or not peaks.ok:
        for k in ("bvp_beat_count", "bvp_valid_ibi_count", "bvp_beat_coverage", *hr_keys):
            out[k] = nan
        return out, False
    sel = peaks.peak_idx[(peaks.peak_idx >= start) & (peaks.peak_idx < end)]
    out["bvp_beat_count"] = float(sel.size)
    ibi = np.diff(sel) / rate_hz if sel.size >= 2 else np.zeros(0)
    lo, hi = params.ibi_valid_range_s
    valid = ibi[(ibi >= lo) & (ibi <= hi)]
    out["bvp_valid_ibi_count"] = float(valid.size)
    window_s = (end - start) / rate_hz
    coverage = float(valid.sum() / window_s) if window_s > 0 else 0.0
    out["bvp_beat_coverage"] = coverage
    available = valid.size >= params.min_beats_for_hr and coverage >= params.min_beat_coverage
    if available:
        hr = 60.0 / valid
        out["hr_mean"] = float(hr.mean())
        out["hr_median"] = float(np.median(hr))
        out["hr_std"] = float(hr.std())
        out["hr_min"] = float(hr.min())
        out["hr_max"] = float(hr.max())
    else:
        for k in hr_keys:
            out[k] = nan
    return out, bool(available)
