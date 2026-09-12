"""Build the canonical Phase-2 WESAD window-feature table.

Pipeline per participant (all deterministic, no labels used before step 3):

  1. verified load (wsr.data.load_wesad)                 -> streams + labels
  2. time-only 60-s grid at t=0 (windowing.build_window_frame)
  3. reference annotation on the existing grid (windowing.annotate_reference_labels)
  4. recording-level processing: EDA decomposition, BVP pulse peaks
  5. per-window quality flags + features for ACC / EDA / BVP / TEMP

Outputs (paths configurable via CLI):
  data/processed/wesad_windows_60s.parquet       canonical table (all complete windows, all participants)
  data/manifests/wesad_feature_schema.json       every column: role, definition, unit, nullable, model_feature
  data/manifests/wesad_windows_60s_manifest.json processing manifest (provenance, counts, hashes, versions)
  results/tables/wesad_windows_60s_summary.csv   per-participant counts (regenerable)

Usage:  python -m wsr.features.build_features [--participants S2 S3] [--out ...]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wsr.data.load_wesad import VerifiedRelease, WesadParticipant, load_participant
from wsr.features import activity, cardiac, eda as eda_mod, temperature
from wsr.features.schema import SCHEMA_VERSION, all_column_defs, schema_document
from wsr.preprocessing import quality as q
from wsr.preprocessing.windowing import WindowSpec, annotate_reference_labels, build_window_frame, omitted_tail_s
from wsr.utils import integrity
from wsr.utils.config import load_config
from wsr.utils.paths import DATA_PROCESSED, MANIFESTS, RESULTS_TABLES, ROOT

FEATURE_DEFS = activity.FEATURES + eda_mod.FEATURES + cardiac.FEATURES + temperature.FEATURES
FEATURE_NAMES = [f.name for f in FEATURE_DEFS]


def params_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    w = cfg["windowing"]
    f = cfg.get("features", {})
    qc = cfg.get("quality", {})
    return {
        "window": WindowSpec(length_s=float(w["length_s"]), step_s=float(w["step_s"]), origin_s=0.0),
        "eda": eda_mod.EdaParams(**f.get("eda", {})),
        "bvp": cardiac.BvpParams(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in f.get("bvp", {}).items()}),
        "quality": q.QualityParams(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in qc.items()}),
    }


def process_participant(part: WesadParticipant, params: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec: WindowSpec = params["window"]
    rates = part.rates_hz
    # 1-2. candidate frame from time + rates only
    frame = build_window_frame(part.participant_id, part.duration_s, rates, part.label_rate_hz, spec)
    tail = omitted_tail_s(part.duration_s, spec)
    # 3. reference annotation AFTER the grid exists
    frame = annotate_reference_labels(frame, part.label)
    frame["source_file"] = part.source_file
    frame["source_sha256"] = part.source_sha256
    # 4. recording-level, label-free processing
    decomp = eda_mod.decompose_eda(part.eda, rates["EDA"], params["eda"])
    peaks = cardiac.detect_pulse_peaks(part.bvp, rates["BVP"], params["bvp"])
    # 5. per-window quality + features
    rows: list[dict[str, Any]] = []
    qp: q.QualityParams = params["quality"]
    for r in frame.itertuples(index=False):
        acc = part.acc[r.acc_start_sample : r.acc_end_sample]
        bvp = part.bvp[r.bvp_start_sample : r.bvp_end_sample]
        eda = part.eda[r.eda_start_sample : r.eda_end_sample]
        temp = part.temp[r.temp_start_sample : r.temp_end_sample]
        row: dict[str, Any] = {}
        row.update(q.modality_flags("acc", acc, rates["ACC"], spec.length_s, qp))
        row.update(q.modality_flags("bvp", bvp, rates["BVP"], spec.length_s, qp))
        row.update(q.modality_flags("eda", eda, rates["EDA"], spec.length_s, qp))
        row.update(q.modality_flags("temp", temp, rates["TEMP"], spec.length_s, qp))
        row.update(q.acc_flags(acc, qp))
        row.update(q.eda_flags(eda, qp))
        row.update(q.temp_flags(temp, qp))
        row["q_eda_decomposition_ok"] = decomp.ok
        row["q_bvp_pulse_detection_ok"] = peaks.ok
        row["q_bvp_hr_available"] = False
        row["q_feature_error"] = ""
        feats: dict[str, float] = {name: np.nan for name in FEATURE_NAMES}
        try:
            feats.update(activity.compute_acc_features(acc))
            feats.update(eda_mod.compute_eda_features(eda, rates["EDA"], decomp, r.eda_start_sample, r.eda_end_sample))
            bvp_feats, hr_ok = cardiac.compute_bvp_features(bvp, rates["BVP"], peaks, r.bvp_start_sample, r.bvp_end_sample, params["bvp"])
            feats.update(bvp_feats)
            row["q_bvp_hr_available"] = hr_ok
            feats.update(temperature.compute_temp_features(temp, rates["TEMP"]))
        except Exception as exc:  # noqa: BLE001 - recorded per window, never fabricated
            row["q_feature_error"] = repr(exc)
        row["q_structural_ok"] = q.structural_ok(row)
        row.update(feats)
        rows.append(row)
    feat_df = pd.DataFrame(rows)
    table = pd.concat([frame.reset_index(drop=True), feat_df], axis=1)
    info = {
        "participant_id": part.participant_id,
        "duration_s": part.duration_s,
        "stream_durations_s": part.stream_durations_s,
        "n_windows": int(len(table)),
        "omitted_tail_s": tail,
        "eda_decomposition_ok": decomp.ok,
        "eda_decomposition_error": decomp.error,
        "bvp_pulse_detection_ok": peaks.ok,
        "bvp_pulse_detection_error": peaks.error,
        "bvp_prominence_threshold": peaks.prominence_threshold,
        "n_pulse_peaks_recording": int(peaks.peak_idx.size),
        "n_scr_peaks_recording": int(decomp.peak_idx.size),
    }
    return table, info


def order_columns(table: pd.DataFrame) -> pd.DataFrame:
    names = [c.name for c in all_column_defs(FEATURE_DEFS)]
    missing = [n for n in names if n not in table.columns]
    extra = [c for c in table.columns if c not in names]
    if missing or extra:
        raise RuntimeError(f"table/schema mismatch: missing={missing} extra={extra}")
    return table[names]


def count_summary(table: pd.DataFrame) -> dict[str, Any]:
    def per(df: pd.DataFrame) -> dict[str, int]:
        return {
            "complete_windows": int(len(df)),
            "eligible_baseline": int(((df["binary_eligible"]) & (df["analysis_label"] == 0)).sum()),
            "eligible_stress": int(((df["binary_eligible"]) & (df["analysis_label"] == 1)).sum()),
            "homogeneous_other_ineligible": int((df["is_label_homogeneous"] & ~df["binary_eligible"]).sum()),
            "mixed_boundary": int((~df["is_label_homogeneous"]).sum()),
            "structural_ok": int(df["q_structural_ok"].sum()),
        }

    out = {"total": per(table), "per_participant": {sid: per(g) for sid, g in table.groupby("participant_id", sort=False)}}
    out["homogeneous_other_by_code"] = {str(k): int(v) for k, v in table.loc[table["is_label_homogeneous"] & ~table["binary_eligible"], "homogeneous_raw_label"].value_counts().sort_index().items()}
    return out


def _git_commit() -> dict[str, Any]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": sha, "dirty_tree": dirty}
    except Exception as exc:  # noqa: BLE001
        return {"commit": None, "dirty_tree": None, "error": repr(exc)}


def build(release: VerifiedRelease, participants: list[str], params: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    tables, infos = [], []
    for sid in participants:
        part = load_participant(sid, release)
        t, info = process_participant(part, params)
        tables.append(t)
        infos.append(info)
        del part
    table = order_columns(pd.concat(tables, ignore_index=True))
    return table, infos


def write_outputs(table: pd.DataFrame, infos: list[dict[str, Any]], params: dict[str, Any], cfg: dict[str, Any], release: VerifiedRelease, out_parquet: Path, schema_path: Path, manifest_path: Path, summary_csv: Path) -> dict[str, Any]:
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_parquet, engine="pyarrow", index=False)
    param_doc = {k: asdict(v) for k, v in params.items()}
    schema = schema_document(FEATURE_DEFS, param_doc)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(schema, indent=1) + "\n", encoding="utf-8")
    import pyarrow
    import scipy

    counts = count_summary(table)
    manifest = {
        "dataset": "wesad",
        "table_path": str(out_parquet.relative_to(ROOT)) if out_parquet.is_relative_to(ROOT) else str(out_parquet),
        "table_sha256": integrity.sha256_file(out_parquet),
        "n_rows": int(len(table)),
        "n_columns": int(table.shape[1]),
        "n_model_features": schema["n_model_features"],
        "schema_version": SCHEMA_VERSION,
        "schema_path": str(schema_path.relative_to(ROOT)) if schema_path.is_relative_to(ROOT) else str(schema_path),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git": _git_commit(),
        "raw_checksum_manifest": {
            "path": str(integrity.manifest_path(release.raw_root, release.manifests_dir)),
            "sha256": integrity.sha256_file(integrity.manifest_path(release.raw_root, release.manifests_dir)),
            "n_files": len(release.expected_hashes),
        },
        "config_snapshot": {"windowing": cfg["windowing"], "features": cfg.get("features"), "quality": cfg.get("quality"), "device": cfg.get("device"), "labels": cfg.get("labels")},
        "parameters": param_doc,
        "window": {"length_s": params["window"].length_s, "step_s": params["window"].step_s, "origin": "synchronised pickle t=0", "interval": "half-open [start, end)", "grid_uses_labels": False},
        "participants": [i["participant_id"] for i in infos],
        "per_participant": infos,
        "counts": counts,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__},
        "stochastic_processing": "none",
    }
    manifest_path.write_text(json.dumps(manifest, indent=1, default=float) + "\n", encoding="utf-8")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"participant_id": sid, **c} for sid, c in counts["per_participant"].items()]).to_csv(summary_csv, index=False, lineterminator="\n")
    return manifest


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participants", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=DATA_PROCESSED / "wesad_windows_60s.parquet")
    ap.add_argument("--schema", type=Path, default=MANIFESTS / "wesad_feature_schema.json")
    ap.add_argument("--manifest", type=Path, default=MANIFESTS / "wesad_windows_60s_manifest.json")
    ap.add_argument("--summary", type=Path, default=RESULTS_TABLES / "wesad_windows_60s_summary.csv")
    args = ap.parse_args(argv)

    cfg = load_config("wesad")
    if cfg.get("device") != "wrist":
        print("configs/wesad.yaml device must be 'wrist' (D-021)", file=sys.stderr)
        return 2
    params = params_from_config(cfg)
    release = VerifiedRelease.open()  # verifies the whole raw tree; fails closed
    participants = args.participants or cfg["participants"]["all"]
    table, infos = build(release, participants, params)
    manifest = write_outputs(table, infos, params, cfg, release, args.out, args.schema, args.manifest, args.summary)
    c = manifest["counts"]["total"]
    print(f"rows={manifest['n_rows']} cols={manifest['n_columns']} features={manifest['n_model_features']} -> {args.out}")
    print(f"eligible baseline={c['eligible_baseline']} stress={c['eligible_stress']} other-homogeneous={c['homogeneous_other_ineligible']} mixed={c['mixed_boundary']}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
