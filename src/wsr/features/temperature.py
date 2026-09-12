"""Wrist skin-temperature (4 Hz) window features. Simple summaries only."""

from __future__ import annotations

import numpy as np

from wsr.features.schema import ColumnDef, feature

FEATURES: list[ColumnDef] = [
    feature("temp_mean", "temp", "Mean skin temperature", "degC"),
    feature("temp_median", "temp", "Median skin temperature", "degC"),
    feature("temp_std", "temp", "Std skin temperature", "degC"),
    feature("temp_min", "temp", "Min skin temperature", "degC"),
    feature("temp_max", "temp", "Max skin temperature", "degC"),
    feature("temp_range", "temp", "Max - min skin temperature", "degC"),
    feature("temp_iqr", "temp", "Interquartile range skin temperature", "degC"),
    feature("temp_slope", "temp", "Least-squares linear slope over the window", "degC/s"),
]


def compute_temp_features(temp_win: np.ndarray, rate_hz: float) -> dict[str, float]:
    x = np.asarray(temp_win, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError(f"TEMP window must be 1-D with >= 2 samples; got {x.shape}")
    t = np.arange(x.size, dtype=np.float64) / rate_hz
    q75, q25 = np.percentile(x, [75, 25])
    return {
        "temp_mean": float(x.mean()),
        "temp_median": float(np.median(x)),
        "temp_std": float(x.std()),
        "temp_min": float(x.min()),
        "temp_max": float(x.max()),
        "temp_range": float(x.max() - x.min()),
        "temp_iqr": float(q75 - q25),
        "temp_slope": float(np.polyfit(t, x, 1)[0]),
    }
