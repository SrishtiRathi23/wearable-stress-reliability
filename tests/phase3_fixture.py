"""Synthetic canonical-schema feature table for Phase-3 tests (no signal processing, no real data)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wsr.features.build_features import FEATURE_DEFS
from wsr.features.schema import all_column_defs, schema_document


def make_synthetic_table(n_participants: int = 6, n_windows: int = 30, seed: int = 0, n_baseline: int = 12, n_stress: int = 8) -> tuple[pd.DataFrame, dict]:
    """Rows follow the Phase-2 schema exactly. Per participant: n_baseline eligible
    baseline windows, n_stress eligible stress windows, remaining windows ineligible
    (transient/mixed). Features are random with a weak class signal; SCR amplitude
    columns carry nulls where scr_count == 0, like the real table."""
    rng = np.random.default_rng(seed)
    schema = schema_document(FEATURE_DEFS, {})
    defs = all_column_defs(FEATURE_DEFS)
    feats = [c.name for c in defs if c.role in ("feature", "feature_provisional")]
    rows = []
    for i in range(n_participants):
        pid = f"S{i + 2}"
        kinds = ["baseline"] * n_baseline + ["stress"] * n_stress + ["transient"] * ((n_windows - n_baseline - n_stress) // 2) + ["mixed"] * (n_windows - n_baseline - n_stress - (n_windows - n_baseline - n_stress) // 2)
        for w, kind in enumerate(kinds):
            elig = kind in ("baseline", "stress")
            lab = 1 if kind == "stress" else (0 if kind == "baseline" else None)
            r = {
                "dataset": "wesad", "participant_id": pid, "window_id": f"wesad:{pid}:w{w:05d}", "window_index": np.int32(w),
                "start_seconds": 60.0 * w, "end_seconds": 60.0 * (w + 1), "start_label_sample": 42000 * w, "end_label_sample": 42000 * (w + 1),
                "acc_start_sample": 1920 * w, "acc_end_sample": 1920 * (w + 1), "bvp_start_sample": 3840 * w, "bvp_end_sample": 3840 * (w + 1),
                "eda_start_sample": 240 * w, "eda_end_sample": 240 * (w + 1), "temp_start_sample": 240 * w, "temp_end_sample": 240 * (w + 1),
                "source_file": f"WESAD/{pid}/{pid}.pkl", "source_sha256": "0" * 64,
                "raw_label_codes_present": {"baseline": "1", "stress": "2", "transient": "0", "mixed": "1,2"}[kind], "n_label_samples": 42000,
                "is_label_homogeneous": kind != "mixed", "homogeneous_raw_label": {"baseline": 1, "stress": 2, "transient": 0, "mixed": None}[kind],
                "condition_name": kind, "binary_eligible": elig, "analysis_label": lab,
                "ineligibility_reason": "" if elig else ("mixed_label_codes" if kind == "mixed" else "homogeneous_code_0_not_in_binary_reference"),
            }
            for c in defs:
                if c.role == "quality":
                    r[c.name] = {"int64": 1920, "bool": False, "float64": 1.0, "string": ""}[c.dtype]
            r["q_structural_ok"] = True
            for m in ("acc", "bvp", "eda", "temp"):
                r[f"q_{m}_count_ok"] = True
            shift = 0.6 if lab == 1 else 0.0
            for f in feats:
                r[f] = float(rng.normal(shift, 1.0)) + i * 0.1
            r["eda_scr_count"] = float(rng.integers(0, 3))
            if r["eda_scr_count"] == 0:
                r["eda_scr_amp_mean"] = np.nan
                r["eda_scr_amp_max"] = np.nan
                r["eda_scr_amp_sum"] = 0.0
            rows.append(r)
    t = pd.DataFrame(rows)[[c.name for c in defs]]
    t["homogeneous_raw_label"] = t["homogeneous_raw_label"].astype("Int16")
    t["analysis_label"] = t["analysis_label"].astype("Int8")
    t["window_index"] = t["window_index"].astype("int32")
    return t, schema
