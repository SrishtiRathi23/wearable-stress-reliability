"""Feature construction: provenance survival, schema safety, determinism, failure contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.wesad_fixture import T_S, make_fake_wesad
from wsr.data.load_wesad import VerifiedRelease, load_participant
from wsr.features import activity, cardiac, eda as eda_mod, temperature
from wsr.features.build_features import FEATURE_DEFS, FEATURE_NAMES, build, count_summary, params_from_config, process_participant, write_outputs
from wsr.features.schema import all_column_defs, schema_document
from wsr.preprocessing.windowing import PROVENANCE_COLUMNS, REFERENCE_COLUMNS
from wsr.utils import integrity
from wsr.utils.config import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config("wesad")


@pytest.fixture
def release(tmp_path: Path):
    raw = tmp_path / "raw" / "wesad"
    (raw / "WESAD").mkdir(parents=True)
    make_fake_wesad(raw / "WESAD", "S2")
    make_fake_wesad(raw / "WESAD", "S3")
    manifests = tmp_path / "manifests"
    integrity.snapshot(raw, manifests)
    return VerifiedRelease.open(raw, manifests)


# --- L: schema safety ------------------------------------------------------------


def test_schema_blocks_metadata_labels_quality_from_model_features():
    doc = schema_document(FEATURE_DEFS, {})
    model = set(doc["model_feature_columns"])
    forbidden = set(PROVENANCE_COLUMNS) | set(REFERENCE_COLUMNS) | {"source_file", "source_sha256"}
    forbidden |= {c.name for c in all_column_defs(FEATURE_DEFS) if c.role == "quality"}
    assert not (model & forbidden), model & forbidden
    for bad in ("participant_id", "window_id", "window_index", "start_seconds", "end_seconds", "analysis_label", "binary_eligible", "condition_name", "ineligibility_reason", "raw_label_codes_present", "homogeneous_raw_label"):
        assert bad not in model
    # provisional beat/HR columns are present in the table but not model features
    assert "hr_mean" in doc["provisional_feature_columns"] and "hr_mean" not in model
    assert "bvp_std" in model and "eda_mean" in model and "acc_mag_std" in model and "temp_mean" in model
    assert all(c["model_feature"] == (c["role"] == "feature") for c in doc["columns"])
    assert len({c["name"] for c in doc["columns"]}) == doc["n_columns"]


# --- K + counts on the fixture ----------------------------------------------------


def test_build_preserves_provenance_and_reference_and_counts(release, cfg):
    params = params_from_config(cfg)
    table, infos = build(release, ["S2", "S3"], params)
    # every schema column present, ordered, nothing else
    assert list(table.columns) == [c.name for c in all_column_defs(FEATURE_DEFS)]
    assert len(table) == 2 * (T_S // 60)
    assert table.participant_id.tolist() == ["S2", "S2", "S3", "S3"]
    assert table.window_id.tolist() == ["wesad:S2:w00000", "wesad:S2:w00001", "wesad:S3:w00000", "wesad:S3:w00001"]
    assert table.acc_start_sample.tolist() == [0, 1920, 0, 1920] and table.end_label_sample.tolist() == [42000, 84000, 42000, 84000]
    assert table.source_file.tolist()[0] == "WESAD/S2/S2.pkl" and table.source_sha256.str.len().eq(64).all()
    # fixture windows straddle baseline/stress boundaries -> mixed, ineligible, analysis_label null
    assert (~table.binary_eligible).all() and table.analysis_label.isna().all() and (table.condition_name == "mixed").all()
    c = count_summary(table)["total"]
    assert c == {"complete_windows": 4, "eligible_baseline": 0, "eligible_stress": 0, "homogeneous_other_ineligible": 0, "mixed_boundary": 4, "structural_ok": 4}
    assert all(i["omitted_tail_s"] == 0.0 for i in infos)
    # a window can be traced back to its raw samples: slice the source and recompute one feature
    p = load_participant("S2", release)
    r = table.iloc[1]
    seg = p.temp[r.temp_start_sample : r.temp_end_sample]
    assert np.isclose(r.temp_mean, seg.mean())


# --- M: determinism ----------------------------------------------------------------


def test_repeated_extraction_is_bit_identical(release, cfg):
    params = params_from_config(cfg)
    p = load_participant("S2", release)
    t1, _ = process_participant(p, params)
    t2, _ = process_participant(p, params)
    pd.testing.assert_frame_equal(t1, t2)
    assert t1[FEATURE_NAMES].to_numpy().tobytes() == t2[FEATURE_NAMES].to_numpy().tobytes()


# --- N: failure contract ----------------------------------------------------------


def test_recording_level_failures_produce_nulls_and_flags(release, cfg):
    params = params_from_config(cfg)
    p = load_participant("S2", release)
    # constant BVP -> detector fails (zero robust amplitude); short EDA -> decomposition fails
    bad = cardiac.detect_pulse_peaks(np.zeros(64 * T_S), 64.0, params["bvp"])
    assert bad.ok is False and "zero robust amplitude" in bad.error
    feats, hr_ok = cardiac.compute_bvp_features(p.bvp[:3840], 64.0, bad, 0, 3840, params["bvp"])
    assert hr_ok is False and np.isnan(feats["hr_mean"]) and np.isnan(feats["bvp_beat_count"]) and np.isfinite(feats["bvp_std"])
    dec = eda_mod.decompose_eda(np.ones(5), 4.0, params["eda"])
    assert dec.ok is False
    ef = eda_mod.compute_eda_features(p.eda[:240], 4.0, dec, 0, 240)
    assert np.isnan(ef["eda_tonic_mean"]) and np.isnan(ef["eda_scr_count"]) and np.isfinite(ef["eda_mean"])


def test_no_scr_in_window_gives_null_amplitude_not_zero(cfg):
    params = params_from_config(cfg)
    t = np.arange(4 * 600) / 4.0
    eda = 1.0 + 0.0001 * t  # slow drift, no SCR
    dec = eda_mod.decompose_eda(eda, 4.0, params["eda"])
    assert dec.ok and dec.peak_idx.size == 0
    f = eda_mod.compute_eda_features(eda[240:480], 4.0, dec, 240, 480)
    assert f["eda_scr_count"] == 0.0 and f["eda_scr_amp_sum"] == 0.0
    assert np.isnan(f["eda_scr_amp_mean"]) and np.isnan(f["eda_scr_amp_max"])


def test_per_window_feature_exception_is_flagged_not_raised(release, cfg, monkeypatch):
    params = params_from_config(cfg)
    p = load_participant("S2", release)

    def boom(*a, **k):
        raise RuntimeError("synthetic feature failure")

    monkeypatch.setattr(temperature, "compute_temp_features", boom)
    t, _ = process_participant(p, params)
    assert (t.q_feature_error.str.contains("synthetic feature failure")).all()
    assert (~t.q_structural_ok).all()
    assert t["temp_mean"].isna().all() and t["acc_mag_mean"].notna().all()


# --- feature-level sanity ---------------------------------------------------------------


def test_acc_features_units_and_magnitude():
    acc = np.zeros((1920, 3))
    acc[:, 2] = 64.0  # 1 g on z
    f = activity.compute_acc_features(acc)
    assert f["acc_z_mean"] == 1.0 and f["acc_mag_mean"] == 1.0 and f["acc_mag_std"] == 0.0 and f["acc_mag_diff_mean_abs"] == 0.0
    assert set(f) == {c.name for c in activity.FEATURES}


def test_temp_and_eda_slope_recover_linear_trend():
    t = np.arange(240) / 4.0
    f = temperature.compute_temp_features(30.0 + 0.01 * t, 4.0)
    assert np.isclose(f["temp_slope"], 0.01) and np.isclose(f["temp_range"], 0.01 * t[-1])
    e = eda_mod.compute_eda_features(2.0 - 0.005 * t, 4.0, None, 0, 240)
    assert np.isclose(e["eda_slope"], -0.005) and np.isnan(e["eda_tonic_mean"])


def test_write_outputs_roundtrip_and_manifest(release, cfg, tmp_path: Path):
    params = params_from_config(cfg)
    table, infos = build(release, ["S2"], params)
    out = tmp_path / "t.parquet"
    m = write_outputs(table, infos, params, cfg, release, out, tmp_path / "schema.json", tmp_path / "manifest.json", tmp_path / "summary.csv")
    back = pd.read_parquet(out)
    pd.testing.assert_frame_equal(back, table)
    assert back.analysis_label.dtype.name == "Int8" and back.homogeneous_raw_label.dtype.name == "Int16"
    assert m["table_sha256"] == integrity.sha256_file(out) and m["n_rows"] == len(table) and m["window"]["grid_uses_labels"] is False
    assert m["raw_checksum_manifest"]["n_files"] == len(release.expected_hashes)
    schema = json.loads((tmp_path / "schema.json").read_text())
    assert schema["n_model_features"] == m["n_model_features"] == len([c for c in FEATURE_DEFS if c.role == "feature"])


# --- real release (integration; requires data + committed manifests) -----------------


def test_real_table_matches_committed_manifest():
    from wsr.utils.paths import DATA_PROCESSED, MANIFESTS

    table_path = DATA_PROCESSED / "wesad_windows_60s.parquet"
    manifest_path = MANIFESTS / "wesad_windows_60s_manifest.json"
    if not table_path.is_file():
        pytest.skip("canonical WESAD feature table not built locally")
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert integrity.sha256_file(table_path) == m["table_sha256"], "local parquet differs from the committed manifest; rebuild or update the manifest"
    t = pd.read_parquet(table_path)
    assert len(t) == m["n_rows"] and list(t.columns) == [c.name for c in all_column_defs(FEATURE_DEFS)]
    assert t.participant_id.nunique() == 15
    c = count_summary(t)["total"]
    assert c == m["counts"]["total"]
    assert c["eligible_baseline"] + c["eligible_stress"] == 434  # independently reviewed expectation, recomputed here
    assert (t.q_structural_ok).all() and (t.q_feature_error == "").all()
