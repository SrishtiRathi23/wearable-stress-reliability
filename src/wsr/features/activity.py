"""Wrist accelerometer (32 Hz) window features.

Input: raw E4 counts (1/64 g, readme II.2), converted to g here. Deterministic,
no filtering. ACC is an activity/confound control, so the set is deliberately
small: per-axis summaries, magnitude summaries and first-difference summaries.
"""

from __future__ import annotations

import numpy as np

from wsr.features.schema import ColumnDef, feature

#: readme II.2: E4 ACC is shipped in 1/64 g. The config value must equal this;
#: any other value is rejected (the release cannot have a different scale).
COUNTS_PER_G = 64.0
_AXES = ("x", "y", "z")


def counts_per_g_from_config(acc_cfg: dict) -> float:
    v = float(acc_cfg.get("counts_per_g", COUNTS_PER_G))
    if v != COUNTS_PER_G:
        raise ValueError(f"features.acc.counts_per_g = {v} contradicts the WESAD release scale of {COUNTS_PER_G:g} counts/g (readme II.2)")
    return v


def _iqr(v: np.ndarray) -> float:
    q75, q25 = np.percentile(v, [75, 25])
    return float(q75 - q25)


FEATURES: list[ColumnDef] = (
    [feature(f"acc_{ax}_{st}", "acc", f"{st} of wrist ACC {ax}-axis in the window", "g") for ax in _AXES for st in ("mean", "std", "min", "max", "range")]
    + [feature(f"acc_mag_{st}", "acc", f"{st} of ACC magnitude sqrt(x^2+y^2+z^2)", "g") for st in ("mean", "std", "median", "iqr", "min", "max", "range", "rms", "p10", "p90")]
    + [
        feature("acc_mag_diff_mean_abs", "acc", "Mean |first difference| of magnitude (sample-to-sample at 32 Hz)", "g"),
        feature("acc_mag_diff_std", "acc", "Std of first difference of magnitude", "g"),
    ]
)


def compute_acc_features(acc_counts: np.ndarray, counts_per_g: float = COUNTS_PER_G) -> dict[str, float]:
    a = np.asarray(acc_counts, dtype=np.float64) / counts_per_g
    if a.ndim != 2 or a.shape[1] != 3 or a.shape[0] < 2:
        raise ValueError(f"ACC window must be (n>=2, 3); got {a.shape}")
    out: dict[str, float] = {}
    for i, ax in enumerate(_AXES):
        v = a[:, i]
        out[f"acc_{ax}_mean"] = float(v.mean())
        out[f"acc_{ax}_std"] = float(v.std())
        out[f"acc_{ax}_min"] = float(v.min())
        out[f"acc_{ax}_max"] = float(v.max())
        out[f"acc_{ax}_range"] = float(v.max() - v.min())
    mag = np.sqrt(np.sum(a * a, axis=1))
    out["acc_mag_mean"] = float(mag.mean())
    out["acc_mag_std"] = float(mag.std())
    out["acc_mag_median"] = float(np.median(mag))
    out["acc_mag_iqr"] = _iqr(mag)
    out["acc_mag_min"] = float(mag.min())
    out["acc_mag_max"] = float(mag.max())
    out["acc_mag_range"] = float(mag.max() - mag.min())
    out["acc_mag_rms"] = float(np.sqrt(np.mean(mag * mag)))
    out["acc_mag_p10"] = float(np.percentile(mag, 10))
    out["acc_mag_p90"] = float(np.percentile(mag, 90))
    d = np.diff(mag)
    out["acc_mag_diff_mean_abs"] = float(np.mean(np.abs(d)))
    out["acc_mag_diff_std"] = float(d.std())
    return out
