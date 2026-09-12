"""Wrist EDA (4 Hz) processing and window features.

Two stages, both deterministic and label-free:

1. `decompose_eda` runs ONCE per participant over the whole synchronised
   recording: tonic = zero-phase Butterworth low-pass (default 0.05 Hz,
   order 2, scipy.signal.sosfiltfilt); phasic = raw - tonic; SCR-like peaks
   on the phasic component with scipy.signal.find_peaks (prominence >=
   scr_min_amplitude_us, distance >= scr_min_distance_s). Recording-level
   filtering avoids 60-s edge artefacts; consequently a window's tonic/
   phasic/SCR values depend on a few seconds of neighbouring signal (a
   filter effect, not a label effect). This is a simple, transparent
   decomposition - NOT a validated cvxEDA/Ledalab implementation. All
   parameters are provisional and recorded in the schema.

2. `compute_eda_features` summarises one window from the raw slice and the
   corresponding slices of the decomposition.

SCR amplitude summaries are null (not zero) when a window contains no peak;
`eda_scr_count` and `eda_scr_amp_sum` are 0 in that case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from wsr.features.schema import ColumnDef, feature


@dataclass(frozen=True)
class EdaParams:
    tonic_lowpass_hz: float = 0.05  # provisional
    filter_order: int = 2
    scr_min_amplitude_us: float = 0.01  # provisional (prominence on the phasic component)
    scr_min_distance_s: float = 1.0


@dataclass
class EdaDecomposition:
    tonic: np.ndarray
    phasic: np.ndarray
    peak_idx: np.ndarray  # sample indices of SCR-like peaks (recording timeline)
    peak_amp: np.ndarray  # prominence (uS) per peak
    ok: bool
    error: str = ""


def decompose_eda(eda: np.ndarray, rate_hz: float, params: EdaParams = EdaParams()) -> EdaDecomposition:
    x = np.asarray(eda, dtype=np.float64)
    try:
        if x.ndim != 1 or x.size < 8 * params.filter_order + 1:
            raise ValueError(f"EDA recording too short for filtering: {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("EDA contains non-finite values")
        sos = signal.butter(params.filter_order, params.tonic_lowpass_hz, btype="low", fs=rate_hz, output="sos")
        tonic = signal.sosfiltfilt(sos, x)
        phasic = x - tonic
        peaks, props = signal.find_peaks(phasic, prominence=params.scr_min_amplitude_us, distance=max(1, int(round(params.scr_min_distance_s * rate_hz))))
        return EdaDecomposition(tonic, phasic, peaks.astype(np.int64), np.asarray(props.get("prominences", []), dtype=np.float64), True)
    except Exception as exc:  # noqa: BLE001 - failure is recorded, never fabricated
        nan = np.full(x.shape, np.nan)
        return EdaDecomposition(nan, nan, np.zeros(0, dtype=np.int64), np.zeros(0), False, repr(exc))


def _slope_per_s(v: np.ndarray, rate_hz: float) -> float:
    t = np.arange(v.size, dtype=np.float64) / rate_hz
    return float(np.polyfit(t, v, 1)[0])


def _iqr(v: np.ndarray) -> float:
    q75, q25 = np.percentile(v, [75, 25])
    return float(q75 - q25)


FEATURES: list[ColumnDef] = [
    feature("eda_mean", "eda", "Mean raw EDA", "uS"),
    feature("eda_median", "eda", "Median raw EDA", "uS"),
    feature("eda_std", "eda", "Std raw EDA", "uS"),
    feature("eda_min", "eda", "Min raw EDA", "uS"),
    feature("eda_max", "eda", "Max raw EDA", "uS"),
    feature("eda_range", "eda", "Max - min raw EDA", "uS"),
    feature("eda_iqr", "eda", "Interquartile range raw EDA", "uS"),
    feature("eda_slope", "eda", "Least-squares linear slope of raw EDA over the window", "uS/s"),
    feature("eda_tonic_mean", "eda", "Mean tonic component (recording-level low-pass)", "uS", nullable=True, missing_reason="decomposition failed"),
    feature("eda_tonic_std", "eda", "Std tonic component", "uS", nullable=True, missing_reason="decomposition failed"),
    feature("eda_tonic_slope", "eda", "Linear slope of tonic component", "uS/s", nullable=True, missing_reason="decomposition failed"),
    feature("eda_phasic_mean", "eda", "Mean phasic component (raw - tonic)", "uS", nullable=True, missing_reason="decomposition failed"),
    feature("eda_phasic_std", "eda", "Std phasic component", "uS", nullable=True, missing_reason="decomposition failed"),
    feature("eda_scr_count", "eda", "Number of SCR-like peaks (phasic prominence >= threshold) whose index falls in the window", "count", nullable=True, missing_reason="decomposition failed"),
    feature("eda_scr_amp_sum", "eda", "Sum of SCR prominences in the window (0 if none)", "uS", nullable=True, missing_reason="decomposition failed"),
    feature("eda_scr_amp_mean", "eda", "Mean SCR prominence", "uS", nullable=True, missing_reason="no SCR in window, or decomposition failed"),
    feature("eda_scr_amp_max", "eda", "Max SCR prominence", "uS", nullable=True, missing_reason="no SCR in window, or decomposition failed"),
]


def compute_eda_features(eda_win: np.ndarray, rate_hz: float, decomp: EdaDecomposition | None, start: int, end: int) -> dict[str, float]:
    x = np.asarray(eda_win, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError(f"EDA window must be 1-D with >= 2 samples; got {x.shape}")
    out: dict[str, float] = {
        "eda_mean": float(x.mean()),
        "eda_median": float(np.median(x)),
        "eda_std": float(x.std()),
        "eda_min": float(x.min()),
        "eda_max": float(x.max()),
        "eda_range": float(x.max() - x.min()),
        "eda_iqr": _iqr(x),
        "eda_slope": _slope_per_s(x, rate_hz),
    }
    nan = float("nan")
    if decomp is None or not decomp.ok:
        for k in ("eda_tonic_mean", "eda_tonic_std", "eda_tonic_slope", "eda_phasic_mean", "eda_phasic_std", "eda_scr_count", "eda_scr_amp_sum", "eda_scr_amp_mean", "eda_scr_amp_max"):
            out[k] = nan
        return out
    ton = decomp.tonic[start:end]
    pha = decomp.phasic[start:end]
    out["eda_tonic_mean"] = float(ton.mean())
    out["eda_tonic_std"] = float(ton.std())
    out["eda_tonic_slope"] = _slope_per_s(ton, rate_hz)
    out["eda_phasic_mean"] = float(pha.mean())
    out["eda_phasic_std"] = float(pha.std())
    sel = (decomp.peak_idx >= start) & (decomp.peak_idx < end)
    amps = decomp.peak_amp[sel]
    out["eda_scr_count"] = float(amps.size)
    out["eda_scr_amp_sum"] = float(amps.sum()) if amps.size else 0.0
    out["eda_scr_amp_mean"] = float(amps.mean()) if amps.size else nan
    out["eda_scr_amp_max"] = float(amps.max()) if amps.size else nan
    return out
