"""Synthetic WESAD-like release builder shared by the audit, loader, windowing and feature tests.

Reproduces the documented release layout (wesad_readme.pdf I.1, II.1-II.3,
III.1) at a tiny scale: a 120-s recording with baseline / stress / amusement /
meditation runs, a raw RespiBAN text file of which the pkl chest ECG is an
exact crop, a raw E4 zip of which the pkl wrist ACC is an exact crop, and a
quest.csv whose intervals are the label runs widened by 10 s on each side in
raw-RespiBAN time (the relation observed on the real release).
"""

from __future__ import annotations

import io
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np

T_S = 120  # pkl duration in seconds
CROP_S = 3  # seconds of raw RespiBAN before the pkl starts
RESP_TAIL_S = 2
E4_CROP_S = 5.0
E4_TAIL_S = 4.0
BASE = (20, 60)  # pkl-time seconds
STRESS = (70, 100)
FUN = (104, 110)
MEDI = (112, 118)


def _min_sec(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}.{s:02d}"


def make_fake_wesad(root: Path, subject: str = "S2", *, label_override=None, chest_ecg_len=None, wrist_eda_len=None, inject_nan=False, subject_field=None) -> Path:
    """Build <root>/<subject>/ with pkl, respiban.txt, E4 zip, quest.csv, readme.txt."""
    rng = np.random.default_rng(0)
    n = 700 * T_S
    folder = root / subject
    folder.mkdir(parents=True, exist_ok=True)

    # labels
    label = np.zeros(n, dtype=np.int32)
    label[BASE[0] * 700 : BASE[1] * 700] = 1
    label[STRESS[0] * 700 : STRESS[1] * 700] = 2
    label[FUN[0] * 700 : FUN[1] * 700] = 3
    label[MEDI[0] * 700 : MEDI[1] * 700] = 4
    if label_override is not None:
        label = label_override(label)

    # raw RespiBAN counts -> the pkl ECG is the exact converted crop
    n_raw = 700 * (CROP_S + T_S + RESP_TAIL_S)
    raw_counts = rng.integers(20000, 45000, size=(n_raw, 8), dtype=np.int64)
    ecg_full = (raw_counts[:, 0] / 65536.0 - 0.5) * 3.0
    ecg = ecg_full[CROP_S * 700 : CROP_S * 700 + n]
    if chest_ecg_len is not None:
        ecg = ecg[:chest_ecg_len]
    header = {"00:11:22": {"sensor": ["ECG", "EDA", "EMG", "TEMP", "XYZ", "XYZ", "XYZ", "RESPIRATION"], "device name": "RespiBan", "column": ["nSeq", "DI", "CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7", "CH8"], "sampling rate": 700, "date": "2017-5-22", "time": "9:39:1.0"}}
    with open(folder / f"{subject}_respiban.txt", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# OpenSignals Text File Format\n")
        fh.write("# " + json.dumps(header) + "\n")
        fh.write("# EndOfHeader\n")
        for i, row in enumerate(raw_counts):
            fh.write(f"{i}\t0\t" + "\t".join(str(int(v)) for v in row) + "\t\n")

    # raw E4 ACC -> the pkl wrist ACC is the exact crop
    acc_rate = 32.0
    # RespiBAN header says local 2017-05-22 09:39:01 (CEST = UTC+2); pkl t0 is CROP_S later.
    # Make the E4 clock agree: pkl t0 on E4 = start_ts + E4_CROP_S must equal 07:39:01 + CROP_S UTC.
    import datetime as _dt

    e4_start_ts = _dt.datetime(2017, 5, 22, 7, 39, 1, tzinfo=_dt.timezone.utc).timestamp() + CROP_S - E4_CROP_S
    n_acc_raw = int(acc_rate * (E4_CROP_S + T_S + E4_TAIL_S))
    acc_raw = rng.integers(-64, 64, size=(n_acc_raw, 3), dtype=np.int64)
    acc_pkl = acc_raw[int(E4_CROP_S * acc_rate) : int(E4_CROP_S * acc_rate) + int(acc_rate * T_S)].astype(np.float64)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("ACC.csv", f"{e4_start_ts:.6f}, {e4_start_ts:.6f}, {e4_start_ts:.6f}\n{acc_rate:.6f}, {acc_rate:.6f}, {acc_rate:.6f}\n" + "\n".join(f"{a},{b},{c}" for a, b, c in acc_raw) + "\n")
        for name, rate in (("BVP.csv", 64.0), ("EDA.csv", 4.0), ("TEMP.csv", 4.0)):
            rows = int(rate * (E4_CROP_S + T_S + E4_TAIL_S))
            z.writestr(name, f"{e4_start_ts:.6f}\n{rate:.6f}\n" + "\n".join("0.5" for _ in range(rows)) + "\n")
        z.writestr("info.txt", "fake")
    (folder / f"{subject}_E4_Data.zip").write_bytes(buf.getvalue())

    wrist_eda = rng.random((4 * T_S, 1))
    if wrist_eda_len is not None:
        wrist_eda = wrist_eda[:wrist_eda_len]
    chest_temp = rng.random((n, 1)).astype(np.float32) + 30
    if inject_nan:
        chest_temp[10, 0] = np.nan
    d = {
        "subject": subject_field or subject,
        "label": label,
        "signal": {
            "chest": {
                "ACC": rng.random((n, 3)),
                "ECG": ecg.reshape(-1, 1),
                "EMG": rng.random((n, 1)),
                "EDA": rng.random((n, 1)),
                "Temp": chest_temp,
                "Resp": rng.random((n, 1)),
            },
            "wrist": {"ACC": acc_pkl, "BVP": rng.random((64 * T_S, 1)), "EDA": wrist_eda, "TEMP": rng.random((4 * T_S, 1)) + 33},
        },
    }
    with open(folder / f"{subject}.pkl", "wb") as fh:
        pickle.dump(d, fh)

    # quest.csv in raw-RespiBAN time: label run widened by 10 s each side, then + crop
    qb = (BASE[0] - 10 + CROP_S, BASE[1] + 10 + CROP_S)
    qs = (STRESS[0] - 10 + CROP_S, STRESS[1] + 10 + CROP_S)
    (folder / f"{subject}_quest.csv").write_text(
        f"# Subj;{subject};;\n# ORDER;Base;TSST;sRead;;\n# START;{_min_sec(qb[0])};{_min_sec(qs[0])};{_min_sec(qs[1] + 1)};;\n# END;{_min_sec(qb[1])};{_min_sec(qs[1])};{_min_sec(qs[1] + 5)};;\n;;;\n",
        encoding="utf-8",
    )
    (folder / f"{subject}_readme.txt").write_text("### Personal information ###\nAge: 27\n\n### Additional notes ###\nfixture note\n", encoding="utf-8")
    return folder


