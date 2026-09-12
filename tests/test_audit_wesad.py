"""Tests for the WESAD structural auditor using a small synthetic release.

The fixture reproduces the documented release layout (wesad_readme.pdf I.1,
II.1-II.3, III.1) at a tiny scale: a 120-s recording with one baseline and
one stress run, a raw RespiBAN text file of which the pkl chest stream is an
exact crop, a raw E4 zip of which the pkl wrist ACC is an exact crop, and a
quest.csv whose intervals are the label runs widened by 10 s on each side in
raw-RespiBAN time (the relation observed on the real release).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from wsr.data import audit_wesad as aw
from wsr.utils import integrity
from wsr.utils.paths import DATA_RAW, MANIFESTS

from tests.wesad_fixture import BASE, CROP_S, E4_CROP_S, E4_TAIL_S, RESP_TAIL_S, STRESS, T_S, make_fake_wesad  # noqa: E402


@pytest.fixture
def fake_wesad(tmp_path: Path) -> Path:
    root = tmp_path / "WESAD"
    root.mkdir()
    (root / "wesad_readme.pdf").write_bytes(b"%PDF-fake")
    make_fake_wesad(root, "S2")
    return root


# --- parsers ----------------------------------------------------------------


def test_parse_min_sec_is_minutes_dot_seconds():
    assert aw.parse_min_sec("7.08") == 428.0
    assert aw.parse_min_sec("39.55") == 2395.0
    assert aw.parse_min_sec("50.3") == 3030.0  # trailing zero dropped by the csv writer
    assert aw.parse_min_sec("0") == 0.0
    with pytest.raises(ValueError):
        aw.parse_min_sec("1.60")


def test_discover_participants_numeric_order(tmp_path: Path):
    for name in ("S10", "S2", "S3", "notes", "SX"):
        (tmp_path / name).mkdir()
    assert aw.discover_participants(tmp_path) == ["S2", "S3", "S10"]
    with pytest.raises(ValueError):
        aw.subject_number("SX")


def test_parse_quest_schedule_drops_reading_blocks(fake_wesad: Path):
    sched = aw.parse_quest_schedule(fake_wesad / "S2" / "S2_quest.csv")
    assert [c["condition"] for c in sched] == ["Base", "TSST"]
    assert sched[0]["expected_code"] == 1 and sched[1]["expected_code"] == 2
    assert sched[0]["start_s"] == BASE[0] - 10 + CROP_S and sched[0]["end_s"] == BASE[1] + 10 + CROP_S


# --- pickle guard -------------------------------------------------------------


def test_load_subject_pickle_refuses_outside_trusted_root(fake_wesad: Path, tmp_path: Path):
    with pytest.raises(ValueError, match="refusing to unpickle"):
        aw.load_subject_pickle(fake_wesad / "S2" / "S2.pkl", trusted_root=tmp_path / "elsewhere")
    d = aw.load_subject_pickle(fake_wesad / "S2" / "S2.pkl", trusted_root=fake_wesad)
    assert d["subject"] == "S2"


# --- full subject audit on the clean fixture -----------------------------------


def test_audit_subject_clean_fixture(fake_wesad: Path):
    rec = aw.audit_subject(fake_wesad, "S2")
    assert rec["readable"] and rec["flags"] == [], rec["flags"]
    assert all(rec["files"].values())
    lab = rec["labels"]
    assert lab["duration_s"] == T_S and lab["codes_present"] == [0, 1, 2, 3, 4]
    assert lab["missing_condition_codes"] == [] and lab["undocumented_codes"] == []
    assert lab["per_code"]["1"]["duration_s"] == 40.0 and lab["per_code"]["1"]["n_runs"] == 1
    assert lab["per_code"]["2"]["duration_s"] == 30.0
    assert [r["code"] for r in lab["runs"]] == [0, 1, 0, 2, 0, 3, 0, 4, 0]
    # signals: all present, durations equal the label duration
    for dev, names in (("chest", ["ACC", "ECG", "EDA", "EMG", "Resp", "Temp"]), ("wrist", ["ACC", "BVP", "EDA", "TEMP"])):
        assert sorted(rec["signals"][dev]) == names
        for e in rec["signals"][dev].values():
            assert e["present"] and e["duration_minus_label_s"] == 0.0 and e["finite_n_nan"] == 0
    # alignment: pkl is a crop of raw RespiBAN at CROP_S; quest intervals are label runs +-10 s
    assert rec["pkl_vs_raw_respiban"]["full_length_match"] and rec["pkl_vs_raw_respiban"]["crop_offset_s"] == CROP_S
    assert rec["pkl_vs_raw_respiban"]["raw_s_after_pkl_end"] == RESP_TAIL_S
    assert rec["schedule_offset_summary"]["start_offsets_s"] == [10.0, 10.0]
    assert rec["schedule_offset_summary"]["end_offsets_s"] == [-10.0, -10.0]
    # wrist crop + device clocks
    assert rec["pkl_vs_raw_e4"]["full_length_match"] and rec["pkl_vs_raw_e4"]["crop_offset_s"] == E4_CROP_S
    assert rec["device_clock_check"]["residual_after_2h_s"] == 0.0
    assert rec["respiban_rows_minus_label_samples"] == 700 * (CROP_S + RESP_TAIL_S)
    assert rec["e4_raw_minus_pkl_wrist_s"]["ACC"] == E4_CROP_S + E4_TAIL_S
    assert aw.classify(rec) == "apparently_usable"


# --- defect detection -------------------------------------------------------


def test_missing_pkl_is_unusable(fake_wesad: Path):
    (fake_wesad / "S2" / "S2.pkl").unlink()
    rec = aw.audit_subject(fake_wesad, "S2")
    assert rec["readable"] is False and "pkl_missing" in rec["flags"]
    assert aw.classify(rec).startswith("unusable")


def test_undocumented_label_code_flagged(tmp_path: Path):
    root = tmp_path / "W"
    make_fake_wesad(root, "S2", label_override=lambda l: np.where(np.arange(l.size) < 700, 9, l).astype(np.int32))
    rec = aw.audit_subject(root, "S2")
    assert rec["labels"]["undocumented_codes"] == [9]
    assert "undocumented_label_codes" in rec["flags"]


def test_chest_length_mismatch_flagged(tmp_path: Path):
    root = tmp_path / "W"
    make_fake_wesad(root, "S2", chest_ecg_len=700 * (T_S - 5))
    rec = aw.audit_subject(root, "S2")
    assert "chest.ECG_length_ne_label" in rec["flags"] and "chest.ECG_duration_mismatch" in rec["flags"]


def test_wrist_duration_mismatch_flagged(tmp_path: Path):
    root = tmp_path / "W"
    make_fake_wesad(root, "S2", wrist_eda_len=4 * (T_S - 30))
    rec = aw.audit_subject(root, "S2")
    assert "wrist.EDA_duration_mismatch" in rec["flags"]
    assert rec["signals"]["wrist"]["EDA"]["duration_minus_label_s"] == -30.0


def test_nan_and_subject_field_mismatch_flagged(tmp_path: Path):
    root = tmp_path / "W"
    make_fake_wesad(root, "S2", inject_nan=True, subject_field="S99")
    rec = aw.audit_subject(root, "S2")
    assert "chest.Temp_nonfinite" in rec["flags"] and rec["signals"]["chest"]["Temp"]["finite_n_nan"] == 1
    assert "pkl_subject_field_mismatch" in rec["flags"]


def test_malformed_pickle_is_reported_not_raised(fake_wesad: Path):
    with open(fake_wesad / "S2" / "S2.pkl", "wb") as fh:
        pickle.dump({"unexpected": 1}, fh)
    rec = aw.audit_subject(fake_wesad, "S2")
    assert rec["readable"] and "pkl_unexpected_top_level_structure" in rec["flags"]


# --- release-level + determinism ------------------------------------------------


def test_audit_release_is_deterministic_and_reports_expected_vs_found(fake_wesad: Path):
    a1 = aw.audit_release(fake_wesad)
    a2 = aw.audit_release(fake_wesad)
    assert json.dumps(a1, sort_keys=True) == json.dumps(a2, sort_keys=True)
    assert a1["participants_found"] == ["S2"]
    assert a1["expected_but_not_found"] == [s for s in aw.EXPECTED_SUBJECTS if s != "S2"]
    assert a1["found_but_not_expected"] == []
    rows = aw.participant_table(a1)
    assert rows[0]["baseline_duration_s"] == 40.0 and rows[0]["stress_runs"] == 1 and rows[0]["classification"] == "apparently_usable"
    assert len(aw.blocks_table(a1)) == 9 and len(aw.signal_table(a1)) == 10


def test_cli_refuses_without_baseline_and_writes_outputs_with_skip(fake_wesad: Path, tmp_path: Path):
    out = tmp_path / "audit.json"
    tables = tmp_path / "tables"
    with pytest.raises(FileNotFoundError):  # integrity.verify: no baseline for the fixture parent
        aw._main(["--wesad-dir", str(fake_wesad), "--out-json", str(out), "--tables-dir", str(tables)])
    rc = aw._main(["--wesad-dir", str(fake_wesad), "--out-json", str(out), "--tables-dir", str(tables), "--skip-baseline-check"])
    assert rc == 0 and out.is_file()
    assert {p.name for p in tables.iterdir()} == {"wesad_participant_audit.csv", "wesad_signal_audit.csv", "wesad_label_audit.csv", "wesad_blocks_audit.csv"}


# --- real release (integration; requires data + committed baseline) --------------


def test_real_release_audit_matches_committed_manifest():
    if not aw.WESAD_DIR.is_dir() or not aw.discover_participants(aw.WESAD_DIR):
        pytest.skip("real WESAD release not present")
    manifest = integrity.manifest_path(DATA_RAW / "wesad", MANIFESTS)
    assert manifest.is_file(), "WESAD data is present but no checksum baseline is committed; run integrity snapshot"
    committed = json.loads((MANIFESTS / "wesad_audit.json").read_text(encoding="utf-8"))
    assert committed["participants_found"] == aw.EXPECTED_SUBJECTS
    rec = aw.audit_subject(aw.WESAD_DIR, "S2")
    assert aw.classify(rec) == "apparently_usable"
    assert rec["pkl_vs_raw_respiban"]["crop_offset_s"] == committed["participants"]["S2"]["pkl_vs_raw_respiban"]["crop_offset_s"]
    assert rec["labels"]["per_code"] == committed["participants"]["S2"]["labels"]["per_code"]


# --- Phase-1 closeout hardening (independent review) --------------------------


def test_non_integer_labels_are_rejected_not_truncated(tmp_path: Path):
    root = tmp_path / "W"
    make_fake_wesad(root, "S2", label_override=lambda l: l.astype(np.float64) + np.where(np.arange(l.size) == 5, 0.5, 0.0))
    rec = aw.audit_subject(root, "S2")
    assert rec["labels"]["integer_valued"] is False and rec["labels"]["per_code"] == {}
    assert "label_vector_not_integer_valued" in rec["flags"]
    assert aw.classify(rec).startswith("unusable")


def test_integral_float_labels_are_accepted(tmp_path: Path):
    root = tmp_path / "W"
    make_fake_wesad(root, "S2", label_override=lambda l: l.astype(np.float64))
    rec = aw.audit_subject(root, "S2")
    assert rec["labels"]["integer_valued"] is True and rec["labels"]["codes_present"] == [0, 1, 2, 3, 4]


def test_ecg_match_uses_absolute_tolerance_and_reports_max_error(fake_wesad: Path):
    d = aw.load_subject_pickle(fake_wesad / "S2" / "S2.pkl", trusted_root=fake_wesad)
    ecg = d["signal"]["chest"]["ECG"]
    ok = aw.raw_chest_crop_offset(fake_wesad / "S2" / "S2_respiban.txt", ecg)
    assert ok["full_length_match"] and ok["max_abs_error_mV"] == 0.0
    # a relative-scale perturbation (1e-6 relative) would pass rtol=1e-5 but must fail rtol=0/atol=1e-9
    perturbed = ecg.copy()
    perturbed[3000, 0] += perturbed[3000, 0] * 1e-6 + 1e-8
    bad = aw.raw_chest_crop_offset(fake_wesad / "S2" / "S2_respiban.txt", perturbed)
    assert bad["matched"] and bad["full_length_match"] is False and bad["max_abs_error_mV"] > 1e-9


def test_wrist_acc_integrality_checked_before_integer_compare(fake_wesad: Path):
    d = aw.load_subject_pickle(fake_wesad / "S2" / "S2.pkl", trusted_root=fake_wesad)
    acc = d["signal"]["wrist"]["ACC"].copy()
    acc[10, 1] += 0.4  # would silently round/truncate to a "match" if cast to int first
    res = aw.raw_wrist_crop_offset(fake_wesad / "S2" / "S2_E4_Data.zip", acc)
    assert res["matched"] is False and res["pkl_values_integral"] is False


def test_failed_crop_match_marks_alignment_unverified(tmp_path: Path):
    root = tmp_path / "W"
    folder = make_fake_wesad(root, "S2")
    # corrupt the raw RespiBAN ECG column so the pkl cannot be located in it
    lines = (folder / "S2_respiban.txt").read_text(encoding="utf-8").splitlines()
    out = []
    for ln in lines:
        if ln.startswith("#"):
            out.append(ln)
        else:
            parts = ln.split("\t")
            parts[2] = "0"
            out.append("\t".join(parts))
    (folder / "S2_respiban.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    rec = aw.audit_subject(root, "S2")
    assert "pkl_chest_not_found_in_raw_respiban" in rec["flags"]
    assert "schedule_alignment_unverified" in rec["flags"]
    assert isinstance(rec["schedule_vs_labels"], str) and rec["schedule_vs_labels"].startswith("UNVERIFIED")
    assert "device_clock_check" not in rec
    assert aw.classify(rec).startswith("questionable")


def test_unexpected_channel_count_flagged(fake_wesad: Path):
    pkl_path = fake_wesad / "S2" / "S2.pkl"
    d = aw.load_subject_pickle(pkl_path, trusted_root=fake_wesad)
    d["signal"]["wrist"]["ACC"] = d["signal"]["wrist"]["ACC"][:, :2]
    with open(pkl_path, "wb") as fh:
        pickle.dump(d, fh)
    rec = aw.audit_subject(fake_wesad, "S2")
    assert "wrist.ACC_channels_2_ne_3" in rec["flags"]
    assert aw.classify(rec).startswith("unusable")
