"""Window-level signal-quality flags (Phase 2).

Flags distinguish "a number could be computed" from "this window is
structurally sound". They never delete windows. Thresholds marked
provisional are configurable (configs/base.yaml: quality) and documented as
such; they are not evidence-based physiological cut-offs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QualityParams:
    constant_eps: float = 1e-9
    acc_clip_counts: float = 127.0  # provisional: |count| >= 127 treated as at the +-2 g rail
    eda_min_plausible_us: float = 0.01  # provisional
    temp_plausible_range_c: tuple[float, float] = (20.0, 45.0)  # provisional


EXPECTED_COUNTS_60S = {"acc": 1920, "bvp": 3840, "eda": 240, "temp": 240}


def expected_count(rate_hz: float, length_s: float) -> int:
    return int(round(rate_hz * length_s))


def modality_flags(mod: str, seg: np.ndarray, rate_hz: float, length_s: float, params: QualityParams) -> dict:
    """Generic per-modality flags. `seg` is (n,) or (n, k)."""
    seg = np.asarray(seg, dtype=np.float64)
    n = int(seg.shape[0])
    finite = np.isfinite(seg)
    finite_fraction = float(finite.mean()) if seg.size else 0.0
    if n and finite.any():
        vals = seg[finite] if seg.ndim == 1 else seg[np.all(finite, axis=1)] if seg.ndim == 2 else seg
        rng = float(np.max(vals) - np.min(vals)) if vals.size else 0.0
    else:
        rng = 0.0
    return {
        f"q_{mod}_n_samples": n,
        f"q_{mod}_count_ok": n == expected_count(rate_hz, length_s),
        f"q_{mod}_finite_fraction": finite_fraction,
        f"q_{mod}_constant": bool(n > 0 and rng < params.constant_eps),
    }


def acc_flags(acc_counts: np.ndarray, params: QualityParams) -> dict:
    a = np.asarray(acc_counts, dtype=np.float64)
    return {"q_acc_clipped": bool(a.size and np.any(np.abs(a[np.isfinite(a)]) >= params.acc_clip_counts))}


def eda_flags(eda: np.ndarray, params: QualityParams) -> dict:
    e = np.asarray(eda, dtype=np.float64)
    f = e[np.isfinite(e)]
    return {"q_eda_below_plausible": bool(f.size and np.any(f < params.eda_min_plausible_us))}


def temp_flags(temp: np.ndarray, params: QualityParams) -> dict:
    t = np.asarray(temp, dtype=np.float64)
    f = t[np.isfinite(t)]
    lo, hi = params.temp_plausible_range_c
    return {"q_temp_out_of_range": bool(f.size and np.any((f < lo) | (f > hi)))}


def structural_ok(flags: dict) -> bool:
    mods = ("acc", "bvp", "eda", "temp")
    return bool(
        all(flags.get(f"q_{m}_count_ok", False) for m in mods)
        and all(flags.get(f"q_{m}_finite_fraction", 0.0) == 1.0 for m in mods)
        and not any(flags.get(f"q_{m}_constant", False) for m in mods)
        and not flags.get("q_feature_error", "")
    )
