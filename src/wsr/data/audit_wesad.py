"""WESAD structural audit (Phase 1). No modelling, no features.

Reads the official WESAD release under data/raw/wesad/WESAD/ and writes a
machine-readable audit of participants, files, signals, sampling rates, label
codes, contiguous label runs, and alignment between the label vector, the
chest/wrist streams, the raw device files and the protocol schedule in
SX_quest.csv.

Documented facts used here come from the release file WESAD/wesad_readme.pdf
(sections cited inline). Everything else is measured from the files.

On pickle loading
-----------------
SX.pkl files are Python pickles (readme II.3). Unpickling executes code paths
chosen by the file, so it is only acceptable for files whose provenance is
established. This module loads pickles ONLY from the raw WESAD tree whose
SHA-256 baseline is committed in data/manifests/raw_checksums_wesad.json and
whose archive provenance is in data/manifests/wesad_source.json. It is not a
generic unpickler and never re-saves the objects.

Usage
-----
    python -m wsr.data.audit_wesad            # full audit, writes manifests + tables
    python -m wsr.data.audit_wesad --subjects S2 S3
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import pickle
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from wsr.data.audit_common import code_summary, contiguous_runs, finiteness, missing_codes, unknown_codes
from wsr.utils import integrity
from wsr.utils.paths import DATA_RAW, MANIFESTS, RESULTS_TABLES

# --------------------------------------------------------------------------
# Documented facts (wesad_readme.pdf). Do not edit without citing the source.
# --------------------------------------------------------------------------

#: readme II.3: label IDs sampled at 700 Hz.
LABEL_CODES: dict[int, str] = {
    0: "transient/undefined",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
    5: "ignore (readme: should be ignored)",
    6: "ignore (readme: should be ignored)",
    7: "ignore (readme: should be ignored)",
}
LABEL_RATE_HZ = 700.0  # readme II.3
CONDITION_CODES = [1, 2, 3, 4]  # the five protocol conditions map to 1,2,3,4 (meditation twice -> 4)

#: readme II.1: all RespiBAN channels at 700 Hz. readme II.2: E4 rates.
NOMINAL_RATES_HZ: dict[str, dict[str, float]] = {
    "chest": {"ACC": 700.0, "ECG": 700.0, "EMG": 700.0, "EDA": 700.0, "Temp": 700.0, "Resp": 700.0},
    "wrist": {"ACC": 32.0, "BVP": 64.0, "EDA": 4.0, "TEMP": 4.0},
}
#: readme II.1/II.2: accelerometers are 3-channel; every other modality is 1-channel.
EXPECTED_CHANNELS: dict[str, dict[str, int]] = {
    "chest": {"ACC": 3, "ECG": 1, "EMG": 1, "EDA": 1, "Temp": 1, "Resp": 1},
    "wrist": {"ACC": 3, "BVP": 1, "EDA": 1, "TEMP": 1},
}

#: readme II.3 names the chest modalities "ACC, ECG, EDA, EMG, RESP, TEMP"; the
#: pickle keys observed are 'Temp' and 'Resp'. Recorded as a doc/file discrepancy.
CHEST_KEY_ALIASES = {"TEMP": "Temp", "RESP": "Resp"}

#: readme I.2: 17 subjects, S1 and S12 discarded (sensor malfunction).
EXPECTED_SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]
DOCUMENTED_ABSENT = ["S1", "S12"]

#: readme I.1: files per subject folder.
EXPECTED_FILES = ["{s}.pkl", "{s}_E4_Data.zip", "{s}_quest.csv", "{s}_readme.txt", "{s}_respiban.txt"]

#: readme II.2: E4 raw files (derived HR/IBI/tags to be ignored).
E4_RAW_FILES = ["ACC.csv", "BVP.csv", "EDA.csv", "TEMP.csv"]

#: readme III.1: quest.csv condition names; bRead/fRead/sRead are to be ignored.
QUEST_CONDITION_TO_CODE = {"Base": 1, "TSST": 2, "Fun": 3, "Medi 1": 4, "Medi 2": 4}
QUEST_IGNORED = {"bRead", "fRead", "sRead"}

WESAD_DIR = DATA_RAW / "wesad" / "WESAD"


# --------------------------------------------------------------------------
# Small parsers
# --------------------------------------------------------------------------


def subject_number(subject_id: str) -> int:
    m = re.fullmatch(r"S(\d+)", subject_id)
    if not m:
        raise ValueError(f"not a WESAD subject id: {subject_id!r}")
    return int(m.group(1))


def discover_participants(wesad_dir: Path) -> list[str]:
    """Subject folders named S<int>, sorted numerically."""
    ids = [p.name for p in Path(wesad_dir).iterdir() if p.is_dir() and re.fullmatch(r"S\d+", p.name)]
    return sorted(ids, key=subject_number)


def expected_files_status(wesad_dir: Path, subject_id: str) -> dict[str, bool]:
    folder = Path(wesad_dir) / subject_id
    return {f.format(s=subject_id): (folder / f.format(s=subject_id)).is_file() for f in EXPECTED_FILES}


def parse_min_sec(token: str) -> float:
    """Parse the quest.csv 'minutes.seconds' time format into seconds.

    readme III.1: "Time is given in the format [minutes.seconds]". So '7.08'
    is 7 min 08 s = 428 s, NOT 7.08 decimal minutes. Seconds part is taken
    from the digits after the dot, left-justified to two digits ('50.3' ->
    50 min 30 s, as seen in S2_quest.csv line END for TSST).
    """
    token = token.strip()
    if "." not in token:
        return float(token) * 60.0
    minutes, frac = token.split(".", 1)
    frac = (frac + "0")[:2] if len(frac) < 2 else frac[:2]
    seconds = int(frac)
    if seconds >= 60:
        raise ValueError(f"seconds field >= 60 in minutes.seconds token {token!r}")
    return int(minutes) * 60.0 + seconds


def parse_quest_schedule(path: Path) -> list[dict[str, Any]]:
    """Read condition order + start/end (seconds from RespiBAN start) from SX_quest.csv.

    readme III.1: line 2 = ORDER, lines 3/4 = START/END. Reading elements are dropped.
    """
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh, delimiter=";"))
    header = {r[0].strip("# ").upper(): r[1:] for r in rows[:4] if r and r[0].startswith("#")}
    order = [x.strip() for x in header["ORDER"] if x.strip()]
    starts = [x for x in header["START"] if x.strip()]
    ends = [x for x in header["END"] if x.strip()]
    if not (len(order) == len(starts) == len(ends)):
        raise ValueError(f"{path}: ORDER/START/END length mismatch {len(order)}/{len(starts)}/{len(ends)}")
    out = []
    for pos, (name, s, e) in enumerate(zip(order, starts, ends)):
        if name in QUEST_IGNORED:
            continue
        if name not in QUEST_CONDITION_TO_CODE:
            raise ValueError(f"{path}: unexpected condition name {name!r}")
        out.append(
            {
                "position": pos,
                "condition": name,
                "expected_code": QUEST_CONDITION_TO_CODE[name],
                "start_s": parse_min_sec(s),
                "end_s": parse_min_sec(e),
                "duration_s": parse_min_sec(e) - parse_min_sec(s),
            }
        )
    return out


def parse_respiban_header(path: Path) -> dict[str, Any]:
    """Header JSON of the OpenSignals text file plus a line count of data rows."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
        second = fh.readline()
        third = fh.readline()
        n_data = 0
        for _ in fh:
            n_data += 1
    meta: dict[str, Any] = {"format_line": first.strip("# \n"), "n_data_rows": n_data}
    try:
        hdr = json.loads(second.strip("# \n"))
        dev = next(iter(hdr.values()))
        meta["sensors"] = dev.get("sensor")
        meta["columns"] = dev.get("column")
        meta["sampling_rate_declared"] = dev.get("sampling rate")
        meta["date"] = dev.get("date")
        meta["time"] = dev.get("time")
    except (json.JSONDecodeError, StopIteration, AttributeError) as exc:
        meta["header_parse_error"] = str(exc)
    meta["end_of_header_line_ok"] = third.strip() == "# EndOfHeader"
    return meta


def raw_chest_crop_offset(respiban_txt: Path, pkl_chest_ecg: np.ndarray, probe_len: int = 2000) -> dict[str, Any]:
    """Locate the pkl chest ECG inside the raw RespiBAN text file.

    readme II.1: raw CH1 = ECG in 16-bit counts; ECG(mV) = (signal/2^16 - 0.5)*3.
    If the converted raw series contains the pkl ECG exactly, the pkl is a crop
    of the raw recording starting at `raw_start_row`; `crop_offset_s` is that
    row / 700. This ties the pkl (and therefore the label vector) to the raw
    RespiBAN timeline that quest.csv times are counted from (readme III.1).
    """
    raw = np.loadtxt(respiban_txt, comments="#", usecols=(2,), dtype=np.int64)
    ecg_raw = (raw / 65536.0 - 0.5) * 3.0
    pkl = np.asarray(pkl_chest_ecg).ravel()
    probe = pkl[:probe_len]
    atol = 1e-9  # absolute tolerance only; a 16-bit count step is ~4.6e-5 mV, so this is far below one count
    cands = np.flatnonzero(np.isclose(ecg_raw, probe[0], rtol=0.0, atol=atol))
    hits = [int(c) for c in cands if c + probe.size <= ecg_raw.size and np.allclose(ecg_raw[c : c + probe.size], probe, rtol=0.0, atol=atol)]
    out: dict[str, Any] = {"raw_rows": int(raw.size), "n_probe_matches": len(hits), "atol": atol}
    if len(hits) != 1:
        out["matched"] = False
        return out
    h = hits[0]
    fits = h + pkl.size <= ecg_raw.size
    max_abs_err = float(np.max(np.abs(ecg_raw[h : h + pkl.size] - pkl))) if fits else None
    full = fits and max_abs_err <= atol
    out.update(
        {
            "matched": True,
            "full_length_match": full,
            "max_abs_error_mV": max_abs_err,
            "raw_start_row": h,
            "crop_offset_s": round(h / LABEL_RATE_HZ, 4),
            "raw_rows_after_pkl_end": int(raw.size - (h + pkl.size)),
            "raw_s_after_pkl_end": round((raw.size - (h + pkl.size)) / LABEL_RATE_HZ, 4),
        }
    )
    return out


def raw_wrist_crop_offset(e4_zip: Path, pkl_wrist_acc: np.ndarray, probe_len: int = 2000) -> dict[str, Any]:
    """Locate the pkl wrist ACC inside the raw E4 ACC.csv (read in memory).

    readme II.2: ACC.csv line 1 = start unix timestamp, line 2 = rate, then 3
    integer columns in 1/64 g. An exact match gives the E4-timeline crop point
    of the pkl and hence the wall-clock (UTC) time of pkl t=0 on the E4 clock.
    """
    with zipfile.ZipFile(e4_zip) as z:
        lines = z.read("ACC.csv").decode("utf-8").strip().splitlines()
    start_ts = float(lines[0].split(",")[0])
    rate = float(lines[1].split(",")[0])
    raw_f = np.array([[float(v) for v in ln.split(",")] for ln in lines[2:]], dtype=np.float64)
    pk_f = np.asarray(pkl_wrist_acc, dtype=np.float64)
    out: dict[str, Any] = {"raw_rows": int(raw_f.shape[0]), "declared_rate_hz": rate, "start_unix_ts": start_ts}
    # readme II.2: ACC is in integer 1/64 g counts. Only compare as integers once both sides are exactly integral.
    raw_integral = bool(np.all(np.isfinite(raw_f)) and np.array_equal(raw_f, np.round(raw_f)))
    pkl_integral = bool(np.all(np.isfinite(pk_f)) and np.array_equal(pk_f, np.round(pk_f)))
    out["raw_values_integral"] = raw_integral
    out["pkl_values_integral"] = pkl_integral
    if not (raw_integral and pkl_integral) or pk_f.ndim != 2 or pk_f.shape[1] != 3 or raw_f.ndim != 2 or raw_f.shape[1] != 3:
        out["matched"] = False
        out["reason"] = "non-integral values or unexpected channel count; integer comparison not attempted"
        return out
    raw = raw_f.astype(np.int64)
    pk = pk_f.astype(np.int64)
    probe = pk[:probe_len]
    cands = np.flatnonzero((raw[:, 0] == probe[0, 0]) & (raw[:, 1] == probe[0, 1]) & (raw[:, 2] == probe[0, 2]))
    hits = [int(c) for c in cands if c + probe.shape[0] <= raw.shape[0] and np.array_equal(raw[c : c + probe.shape[0]], probe)]
    out["n_probe_matches"] = len(hits)
    if len(hits) != 1:
        out["matched"] = False
        return out
    h = hits[0]
    full = h + pk.shape[0] <= raw.shape[0] and bool(np.array_equal(raw[h : h + pk.shape[0]], pk))
    t0 = start_ts + h / rate
    out.update(
        {
            "matched": True,
            "full_length_match": full,
            "raw_start_row": h,
            "crop_offset_s": round(h / rate, 4),
            "pkl_t0_unix_ts": round(t0, 4),
            "pkl_t0_utc": dt.datetime.fromtimestamp(t0, dt.timezone.utc).isoformat(),
        }
    )
    return out


def device_clock_difference(respiban_meta: dict[str, Any], chest_crop_s: float, e4_pkl_t0_unix: float) -> dict[str, Any]:
    """Wall-clock time of pkl t=0 on each device's own clock, and their difference.

    RespiBAN header 'date'/'time' are device-local wall clock with 1-s
    resolution; E4 timestamps are unix UTC. In Germany in May-Aug 2017 local
    time was CEST = UTC+2, so a difference of ~2.000 h means the two devices
    agree at the synchronisation point. INFERENCE, not a documented fact.
    """
    try:
        y, mo, d = (int(x) for x in respiban_meta["date"].split("-"))
        hh, mm, ss = respiban_meta["time"].split(":")
        resp_start = dt.datetime(y, mo, d, int(hh), int(mm), int(float(ss)))
    except (KeyError, ValueError, AttributeError) as exc:
        return {"error": f"cannot parse RespiBAN header date/time: {exc}"}
    resp_t0 = resp_start + dt.timedelta(seconds=chest_crop_s)
    e4_t0 = dt.datetime.fromtimestamp(e4_pkl_t0_unix, dt.timezone.utc).replace(tzinfo=None)
    diff_s = (resp_t0 - e4_t0).total_seconds()
    return {
        "respiban_pkl_t0_local": resp_t0.isoformat(timespec="milliseconds"),
        "e4_pkl_t0_utc": e4_t0.isoformat(timespec="milliseconds"),
        "local_minus_utc_s": round(diff_s, 3),
        "residual_after_2h_s": round(diff_s - 7200.0, 3),
    }


def parse_e4_zip(path: Path) -> dict[str, Any]:
    """Per-file start timestamp, declared rate and row count from SX_E4_Data.zip (read in memory)."""
    out: dict[str, Any] = {}
    with zipfile.ZipFile(path) as z:
        out["members"] = sorted(z.namelist())
        for name in E4_RAW_FILES:
            if name not in z.namelist():
                out[name] = {"present": False}
                continue
            with z.open(name) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8")
                ts_line = text.readline().strip()
                rate_line = text.readline().strip()
                n_rows = sum(1 for line in text if line.strip())
            ts = [float(x) for x in ts_line.split(",") if x]
            rates = [float(x) for x in rate_line.split(",") if x]
            out[name] = {
                "present": True,
                "start_unix_ts": ts[0] if ts else None,
                "declared_rate_hz": rates[0] if rates else None,
                "n_rows": n_rows,
                "duration_s": (n_rows / rates[0]) if rates and rates[0] else None,
            }
    return out


# --------------------------------------------------------------------------
# Pickle audit
# --------------------------------------------------------------------------


def load_subject_pickle(path: Path, trusted_root: Path = WESAD_DIR) -> dict[str, Any]:
    """Load SX.pkl from inside `trusted_root` only. See module docstring for the security rationale.

    `trusted_root` defaults to the raw WESAD tree; the CLI verifies that tree
    against the committed checksum baseline before any pickle is opened. Tests
    pass a synthetic fixture directory they built themselves.
    """
    path = Path(path).resolve()
    if Path(trusted_root).resolve() not in path.parents:
        raise ValueError(f"refusing to unpickle a file outside the trusted raw tree {trusted_root}: {path}")
    with open(path, "rb") as fh:
        return pickle.load(fh, encoding="latin1")  # Python-2 era pickle; latin1 preserves byte strings


def _signal_entry(name: str, arr: Any, nominal_rate: float | None, ref_duration_s: float) -> dict[str, Any]:
    arr = np.asarray(arr)
    n = int(arr.shape[0]) if arr.ndim >= 1 else 0
    entry: dict[str, Any] = {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "n_samples": n,
        "nominal_rate_hz": nominal_rate,
        "empty": n == 0,
    }
    entry.update({f"finite_{k}": v for k, v in finiteness(arr).items()})
    if n and np.issubdtype(arr.dtype, np.number):
        entry["min"] = float(np.nanmin(arr))
        entry["max"] = float(np.nanmax(arr))
    if nominal_rate:
        dur = n / nominal_rate
        entry["inferred_duration_s"] = round(dur, 4)
        entry["duration_minus_label_s"] = round(dur - ref_duration_s, 4)
    return entry


def audit_labels(labels: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels).ravel()
    integer_dtype = bool(np.issubdtype(labels.dtype, np.integer))
    integer_valued = integer_dtype or (
        labels.size > 0 and np.issubdtype(labels.dtype, np.number) and bool(np.all(np.isfinite(labels))) and bool(np.array_equal(labels, np.round(labels)))
    )
    if not integer_valued:
        return {"dtype": str(labels.dtype), "n_samples": int(labels.size), "integer_valued": False, "codes_present": [], "undocumented_codes": [], "missing_condition_codes": CONDITION_CODES, "per_code": {}, "runs": []}
    if not integer_dtype:
        labels = np.round(labels).astype(np.int64)
    summ = code_summary(labels, LABEL_RATE_HZ)
    runs = contiguous_runs(labels)
    return {
        "dtype": str(labels.dtype),
        "integer_valued": True,
        "n_samples": int(labels.size),
        "duration_s": round(labels.size / LABEL_RATE_HZ, 4),
        "codes_present": sorted(int(c) for c in np.unique(labels)),
        "undocumented_codes": unknown_codes(labels, LABEL_CODES),
        "missing_condition_codes": missing_codes(labels, CONDITION_CODES),
        "per_code": {
            str(code): {
                "meaning": LABEL_CODES.get(code, "UNDOCUMENTED"),
                "n_samples": s["n_samples"],
                "duration_s": round(s["duration_s"], 4),
                "n_runs": s["n_runs"],
                "run_durations_s": [round(x, 4) for x in s["run_durations_s"]],
            }
            for code, s in summ.items()
        },
        "runs": [
            {**r.to_dict(LABEL_RATE_HZ), "meaning": LABEL_CODES.get(r.code, "UNDOCUMENTED")}
            for r in runs
        ],
    }


def compare_runs_to_schedule(runs: list[dict[str, Any]], schedule: list[dict[str, Any]], crop_offset_s: float = 0.0) -> list[dict[str, Any]]:
    """For each scheduled condition, find the label run of the expected code that overlaps it.

    quest.csv times are counted from the raw RespiBAN start (readme III.1); the
    pkl label vector starts `crop_offset_s` later (see raw_chest_crop_offset),
    so the schedule is shifted into pkl time before comparison. Offsets are
    run_start - schedule_start and run_end - schedule_end in pkl time.
    """
    out = []
    for cond in schedule:
        cs, ce = cond["start_s"] - crop_offset_s, cond["end_s"] - crop_offset_s
        cands = [r for r in runs if r["code"] == cond["expected_code"] and r["end_s"] > cs and r["start_s"] < ce]
        if not cands:
            out.append({**cond, "schedule_start_pkl_s": round(cs, 3), "schedule_end_pkl_s": round(ce, 3), "matched": False})
            continue
        best = max(cands, key=lambda r: min(r["end_s"], ce) - max(r["start_s"], cs))
        out.append(
            {
                **cond,
                "schedule_start_pkl_s": round(cs, 3),
                "schedule_end_pkl_s": round(ce, 3),
                "matched": True,
                "n_overlapping_runs": len(cands),
                "run_start_s": best["start_s"],
                "run_end_s": best["end_s"],
                "run_duration_s": best["duration_s"],
                "start_offset_s": round(best["start_s"] - cs, 3),
                "end_offset_s": round(best["end_s"] - ce, 3),
            }
        )
    return out


def audit_subject(wesad_dir: Path, subject_id: str) -> dict[str, Any]:
    folder = Path(wesad_dir) / subject_id
    rec: dict[str, Any] = {"participant_id": subject_id, "files": expected_files_status(wesad_dir, subject_id), "flags": []}
    flags: list[str] = rec["flags"]

    # --- side files -------------------------------------------------------
    quest = folder / f"{subject_id}_quest.csv"
    if quest.is_file():
        try:
            rec["schedule"] = parse_quest_schedule(quest)
        except Exception as exc:  # noqa: BLE001 - audit must report, not crash
            rec["schedule_error"] = str(exc)
            flags.append("quest_parse_error")
    readme = folder / f"{subject_id}_readme.txt"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8", errors="replace")
        notes = text.split("### Additional notes ###", 1)[-1].strip() if "### Additional notes ###" in text else ""
        rec["readme_notes"] = notes
    resp = folder / f"{subject_id}_respiban.txt"
    if resp.is_file():
        rec["respiban_txt"] = parse_respiban_header(resp)
    e4 = folder / f"{subject_id}_E4_Data.zip"
    if e4.is_file():
        try:
            rec["e4_zip"] = parse_e4_zip(e4)
        except Exception as exc:  # noqa: BLE001
            rec["e4_zip"] = {"error": str(exc)}
            flags.append("e4_zip_error")

    # --- pickle -----------------------------------------------------------
    pkl = folder / f"{subject_id}.pkl"
    if not pkl.is_file():
        rec["readable"] = False
        flags.append("pkl_missing")
        return rec
    try:
        d = load_subject_pickle(pkl, trusted_root=wesad_dir)
    except Exception as exc:  # noqa: BLE001
        rec["readable"] = False
        rec["pkl_error"] = repr(exc)
        flags.append("pkl_unreadable")
        return rec
    rec["readable"] = True
    rec["pkl_top_level_keys"] = sorted(d.keys()) if isinstance(d, dict) else str(type(d))
    if not isinstance(d, dict) or not {"signal", "label", "subject"} <= set(d.keys()):
        flags.append("pkl_unexpected_top_level_structure")
        return rec
    rec["pkl_subject_field"] = str(d["subject"])
    if str(d["subject"]) != subject_id:
        flags.append("pkl_subject_field_mismatch")

    rec["labels"] = audit_labels(d["label"])
    if not rec["labels"]["integer_valued"]:
        flags.append("label_vector_not_integer_valued")
        return rec
    label_dur = rec["labels"]["duration_s"]
    if rec["labels"]["undocumented_codes"]:
        flags.append("undocumented_label_codes")
    if rec["labels"]["missing_condition_codes"]:
        flags.append("missing_condition_codes")

    rec["devices"] = sorted(d["signal"].keys())
    rec["signals"] = {}
    for dev in ("chest", "wrist"):
        if dev not in d["signal"]:
            flags.append(f"{dev}_missing")
            continue
        rec["signals"][dev] = {}
        present = set(d["signal"][dev].keys())
        expected = set(NOMINAL_RATES_HZ[dev].keys())
        for name in sorted(present | expected):
            if name not in present:
                rec["signals"][dev][name] = {"present": False}
                flags.append(f"{dev}.{name}_missing")
                continue
            entry = _signal_entry(name, d["signal"][dev][name], NOMINAL_RATES_HZ[dev].get(name), label_dur)
            entry["present"] = True
            exp_ch = EXPECTED_CHANNELS[dev].get(name)
            got_ch = entry["shape"][1] if len(entry["shape"]) == 2 else (1 if len(entry["shape"]) == 1 else None)
            entry["expected_channels"] = exp_ch
            if exp_ch is not None and got_ch != exp_ch:
                flags.append(f"{dev}.{name}_channels_{got_ch}_ne_{exp_ch}")
            if not entry.get("finite_numeric", True):
                flags.append(f"{dev}.{name}_not_numeric")
            if name not in expected:
                entry["undocumented_signal"] = True
                flags.append(f"{dev}.{name}_undocumented")
            if entry["empty"]:
                flags.append(f"{dev}.{name}_empty")
            if entry.get("finite_n_nan") or entry.get("finite_n_inf"):
                flags.append(f"{dev}.{name}_nonfinite")
            if entry.get("duration_minus_label_s") is not None and abs(entry["duration_minus_label_s"]) > 1.0:
                flags.append(f"{dev}.{name}_duration_mismatch")
            rec["signals"][dev][name] = entry
    # chest arrays must be sample-aligned with the label vector (readme II: same start, 700 Hz)
    if "chest" in rec["signals"]:
        n_lab = rec["labels"]["n_samples"]
        for name, e in rec["signals"]["chest"].items():
            if e.get("present") and e["n_samples"] != n_lab:
                flags.append(f"chest.{name}_length_ne_label")

    # --- cross-file alignment -------------------------------------------
    crop: float | None = None  # None = crop not established; schedule alignment is then UNVERIFIED
    if resp.is_file() and "chest" in rec["signals"] and rec["signals"]["chest"].get("ECG", {}).get("present"):
        try:
            rec["pkl_vs_raw_respiban"] = raw_chest_crop_offset(resp, d["signal"]["chest"]["ECG"])
        except Exception as exc:  # noqa: BLE001
            rec["pkl_vs_raw_respiban"] = {"error": repr(exc), "matched": False}
        if rec["pkl_vs_raw_respiban"].get("matched") and rec["pkl_vs_raw_respiban"].get("full_length_match"):
            crop = rec["pkl_vs_raw_respiban"]["crop_offset_s"]
        else:
            flags.append("pkl_chest_not_found_in_raw_respiban")
    if e4.is_file() and "wrist" in rec["signals"] and rec["signals"]["wrist"].get("ACC", {}).get("present"):
        try:
            rec["pkl_vs_raw_e4"] = raw_wrist_crop_offset(e4, d["signal"]["wrist"]["ACC"])
        except Exception as exc:  # noqa: BLE001
            rec["pkl_vs_raw_e4"] = {"error": repr(exc), "matched": False}
        if not (rec["pkl_vs_raw_e4"].get("matched") and rec["pkl_vs_raw_e4"].get("full_length_match")):
            flags.append("pkl_wrist_not_found_in_raw_e4")
        elif crop is not None and "respiban_txt" in rec:
            rec["device_clock_check"] = device_clock_difference(rec["respiban_txt"], crop, rec["pkl_vs_raw_e4"]["pkl_t0_unix_ts"])
            res = rec["device_clock_check"].get("residual_after_2h_s")
            if res is not None and abs(res) > 30.0:
                flags.append("device_clocks_disagree_gt_30s")
    if "schedule" in rec and crop is None:
        rec["schedule_vs_labels"] = "UNVERIFIED: pkl-to-raw crop offset could not be established, so quest.csv times cannot be placed on the pkl timeline"
        rec["schedule_offset_summary"] = {"status": "UNVERIFIED"}
        flags.append("schedule_alignment_unverified")
    elif "schedule" in rec:
        rec["schedule_vs_labels"] = compare_runs_to_schedule(rec["labels"]["runs"], rec["schedule"], crop)
        starts = [m["start_offset_s"] for m in rec["schedule_vs_labels"] if m["matched"]]
        ends = [m["end_offset_s"] for m in rec["schedule_vs_labels"] if m["matched"]]
        rec["schedule_offset_summary"] = {
            "crop_offset_applied_s": crop,
            "start_offsets_s": starts,
            "end_offsets_s": ends,
            "start_offset_spread_s": round(max(starts) - min(starts), 3) if starts else None,
            "end_offset_spread_s": round(max(ends) - min(ends), 3) if ends else None,
        }
        for m in rec["schedule_vs_labels"]:
            if not m["matched"]:
                flags.append(f"schedule_{m['condition']}_no_matching_run")
        # Offsets are expected to be a constant trim per participant; an inconsistent trim across conditions is suspicious.
        if starts and (max(starts) - min(starts) > 2.0 or max(ends) - min(ends) > 2.0):
            flags.append("schedule_offsets_inconsistent_across_conditions")
    if "respiban_txt" in rec and "n_data_rows" in rec["respiban_txt"]:
        rec["respiban_rows_minus_label_samples"] = rec["respiban_txt"]["n_data_rows"] - rec["labels"]["n_samples"]
    if "e4_zip" in rec and "wrist" in rec["signals"]:
        rec["e4_raw_minus_pkl_wrist_s"] = {}
        for name in E4_RAW_FILES:
            key = name[:-4]
            raw = rec["e4_zip"].get(name, {})
            pk = rec["signals"]["wrist"].get(key, {})
            if raw.get("duration_s") is not None and pk.get("inferred_duration_s") is not None:
                rec["e4_raw_minus_pkl_wrist_s"][key] = round(raw["duration_s"] - pk["inferred_duration_s"], 3)

    del d
    return rec


# --------------------------------------------------------------------------
# Whole-release audit + outputs
# --------------------------------------------------------------------------


def classify(rec: dict[str, Any]) -> str:
    """apparently_usable / questionable / unusable:<reason>. Never final; for review."""
    if not rec.get("readable"):
        return "unusable:pkl_unreadable_or_missing"
    hard = [f for f in rec["flags"] if f.endswith("_missing") or f.endswith("_no_matching_run") or "unexpected" in f or "not_integer_valued" in f or "_channels_" in f or "not_numeric" in f]
    if hard:
        return "unusable:" + ";".join(hard)
    if rec["flags"]:
        return "questionable:" + ";".join(rec["flags"])
    return "apparently_usable"


def audit_release(wesad_dir: Path = WESAD_DIR, subjects: list[str] | None = None) -> dict[str, Any]:
    wesad_dir = Path(wesad_dir)
    found = discover_participants(wesad_dir)
    todo = subjects or found
    out: dict[str, Any] = {
        "release_dir": str(wesad_dir),
        "readme_pdf_present": (wesad_dir / "wesad_readme.pdf").is_file(),
        "participants_found": found,
        "expected_participants_per_readme": EXPECTED_SUBJECTS,
        "documented_absent": DOCUMENTED_ABSENT,
        "found_but_not_expected": sorted(set(found) - set(EXPECTED_SUBJECTS), key=subject_number),
        "expected_but_not_found": sorted(set(EXPECTED_SUBJECTS) - set(found), key=subject_number),
        "label_codes_documented": {str(k): v for k, v in LABEL_CODES.items()},
        "label_rate_hz": LABEL_RATE_HZ,
        "nominal_rates_hz": NOMINAL_RATES_HZ,
        "participants": {},
    }
    for sid in todo:
        rec = audit_subject(wesad_dir, sid)
        rec["classification"] = classify(rec)
        out["participants"][sid] = rec
    return out


def participant_table(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for sid, r in audit["participants"].items():
        lab = r.get("labels", {})
        pc = lab.get("per_code", {})

        def dur(code: int) -> float | None:
            return pc.get(str(code), {}).get("duration_s")

        def nruns(code: int) -> int | None:
            return pc.get(str(code), {}).get("n_runs")

        sig = r.get("signals", {})
        missing = [f for f in r["flags"] if f.endswith("_missing")]
        rows.append(
            {
                "participant_id": sid,
                "readable": r.get("readable"),
                "recording_duration_s": lab.get("duration_s"),
                "baseline_duration_s": dur(1),
                "stress_duration_s": dur(2),
                "amusement_duration_s": dur(3),
                "meditation_duration_s": dur(4),
                "transient_duration_s": dur(0),
                "ignore_codes_duration_s": sum(dur(c) or 0 for c in (5, 6, 7)),
                "baseline_runs": nruns(1),
                "stress_runs": nruns(2),
                "codes_present": ",".join(map(str, lab.get("codes_present", []))),
                "undocumented_codes": ",".join(map(str, lab.get("undocumented_codes", []))),
                "wrist_available": "wrist" in sig,
                "chest_available": "chest" in sig,
                "missing_signals": ";".join(missing),
                "shape_alignment_flags": ";".join(f for f in r["flags"] if "length" in f or "duration" in f or "schedule" in f),
                "nan_inf_flags": ";".join(f for f in r["flags"] if "nonfinite" in f),
                "classification": r.get("classification"),
                "readme_notes": r.get("readme_notes", ""),
            }
        )
    return rows


def signal_table(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for sid, r in audit["participants"].items():
        for dev, sigs in r.get("signals", {}).items():
            for name, e in sigs.items():
                rows.append(
                    {
                        "participant_id": sid,
                        "device": dev,
                        "signal": name,
                        "present": e.get("present"),
                        "nominal_rate_hz": e.get("nominal_rate_hz"),
                        "n_samples": e.get("n_samples"),
                        "n_channels": (e.get("shape") or [None, None])[1] if e.get("shape") and len(e["shape"]) > 1 else 1,
                        "dtype": e.get("dtype"),
                        "inferred_duration_s": e.get("inferred_duration_s"),
                        "duration_minus_label_s": e.get("duration_minus_label_s"),
                        "n_nan": e.get("finite_n_nan"),
                        "n_inf": e.get("finite_n_inf"),
                        "min": e.get("min"),
                        "max": e.get("max"),
                    }
                )
    return rows


def label_table(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for sid, r in audit["participants"].items():
        for code, s in r.get("labels", {}).get("per_code", {}).items():
            rows.append({"participant_id": sid, "code": int(code), "meaning": s["meaning"], "n_samples": s["n_samples"], "duration_s": s["duration_s"], "n_runs": s["n_runs"]})
    return rows


def blocks_table(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for sid, r in audit["participants"].items():
        for i, run in enumerate(r.get("labels", {}).get("runs", [])):
            rows.append({"participant_id": sid, "run_index": i, "code": run["code"], "meaning": run["meaning"], "start_idx": run["start_idx"], "end_idx": run["end_idx"], "start_s": round(run["start_s"], 4), "end_s": round(run["end_s"], 4), "duration_s": round(run["duration_s"], 4)})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wesad-dir", type=Path, default=WESAD_DIR)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--out-json", type=Path, default=MANIFESTS / "wesad_audit.json")
    ap.add_argument("--tables-dir", type=Path, default=RESULTS_TABLES)
    ap.add_argument("--skip-baseline-check", action="store_true", help="do not require a verified raw checksum baseline (tests only)")
    args = ap.parse_args(argv)

    if not args.skip_baseline_check:
        raw_root = args.wesad_dir.parent
        diff = integrity.verify(raw_root)  # raises if no baseline
        if not integrity.is_unchanged(diff):
            print(f"REFUSING to audit: raw tree differs from committed baseline: {diff}", file=sys.stderr)
            return 2

    audit = audit_release(args.wesad_dir, args.subjects)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=1, sort_keys=True)
        fh.write("\n")
    write_csv(args.tables_dir / "wesad_participant_audit.csv", participant_table(audit))
    write_csv(args.tables_dir / "wesad_signal_audit.csv", signal_table(audit))
    write_csv(args.tables_dir / "wesad_label_audit.csv", label_table(audit))
    write_csv(args.tables_dir / "wesad_blocks_audit.csv", blocks_table(audit))
    print(f"audited {len(audit['participants'])} participants -> {args.out_json}")
    for sid, r in audit["participants"].items():
        print(f"  {sid}: {r['classification']}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
