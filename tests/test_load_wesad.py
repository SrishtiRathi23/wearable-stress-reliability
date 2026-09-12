"""Verified-loader tests (D-022) on a synthetic release with its own checksum baseline."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from tests.wesad_fixture import T_S, make_fake_wesad
from wsr.data.load_wesad import LoaderValidationError, RawIntegrityError, VerifiedRelease, load_participant, validate_participant_dict
from wsr.utils import integrity


@pytest.fixture
def raw_with_baseline(tmp_path: Path):
    raw = tmp_path / "raw" / "wesad"
    (raw / "WESAD").mkdir(parents=True)
    make_fake_wesad(raw / "WESAD", "S2")
    manifests = tmp_path / "manifests"
    integrity.snapshot(raw, manifests)
    return raw, manifests


def test_open_refuses_without_baseline(tmp_path: Path):
    raw = tmp_path / "raw" / "wesad"
    (raw / "WESAD").mkdir(parents=True)
    make_fake_wesad(raw / "WESAD", "S2")
    with pytest.raises(RawIntegrityError, match="no committed checksum baseline"):
        VerifiedRelease.open(raw, tmp_path / "manifests")


def test_open_refuses_when_tree_modified(raw_with_baseline):
    raw, manifests = raw_with_baseline
    (raw / "WESAD" / "S2" / "S2_readme.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(RawIntegrityError, match="differs from the committed baseline"):
        VerifiedRelease.open(raw, manifests)


def test_load_rehashes_file_and_refuses_modified_pickle(raw_with_baseline):
    """A: even with an already-opened release handle, a changed pickle is never unpickled."""
    raw, manifests = raw_with_baseline
    rel = VerifiedRelease.open(raw, manifests)
    pkl = raw / "WESAD" / "S2" / "S2.pkl"
    d = pickle.load(open(pkl, "rb"), encoding="latin1")
    d["label"] = d["label"].copy()
    d["label"][:] = 2
    with open(pkl, "wb") as fh:
        pickle.dump(d, fh)
    with pytest.raises(RawIntegrityError, match="hash"):
        load_participant("S2", rel)


def test_load_refuses_file_outside_baseline(raw_with_baseline):
    raw, manifests = raw_with_baseline
    rel = VerifiedRelease.open(raw, manifests)
    with pytest.raises(RawIntegrityError, match="not part of the committed baseline"):
        load_participant("S3", rel)  # no such subject in the baseline
    with pytest.raises(RawIntegrityError, match="outside the verified raw root"):
        rel.verified_path("../escape.pkl")


def test_successful_load_validates_and_exposes_streams(raw_with_baseline):
    """B: device/channel/rate validation + read-only streams."""
    raw, manifests = raw_with_baseline
    rel = VerifiedRelease.open(raw, manifests)
    p = load_participant("S2", rel)
    assert p.participant_id == "S2" and p.acc.shape == (32 * T_S, 3) and p.bvp.shape == (64 * T_S,)
    assert p.eda.shape == (4 * T_S,) and p.temp.shape == (4 * T_S,) and p.label.shape == (700 * T_S,)
    assert p.rates_hz == {"ACC": 32.0, "BVP": 64.0, "EDA": 4.0, "TEMP": 4.0} and p.label_rate_hz == 700.0
    assert p.duration_s == T_S and p.validation["duration_spread_s"] == 0.0
    assert p.source_file == "WESAD/S2/S2.pkl" and len(p.source_sha256) == 64
    assert all(c["ok"] for c in p.validation["checks"])
    with pytest.raises(ValueError):
        p.acc[0, 0] = 1.0  # read-only


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d["signal"]["wrist"].pop("EDA"), "wrist.EDA present"),
        (lambda d: d["signal"]["wrist"].__setitem__("ACC", d["signal"]["wrist"]["ACC"][:, :2]), "3 channel"),
        (lambda d: d["signal"]["wrist"].__setitem__("TEMP", d["signal"]["wrist"]["TEMP"].astype(str)), "numeric dtype"),
        (lambda d: d["signal"]["wrist"]["BVP"].__setitem__((5, 0), np.nan), "all finite"),
        (lambda d: d["signal"]["wrist"].__setitem__("EDA", d["signal"]["wrist"]["EDA"][: 4 * (T_S - 5)]), "durations agree"),
        (lambda d: d.__setitem__("label", d["label"][: 700 * (T_S - 5)]), "durations agree"),
        (lambda d: d.__setitem__("label", d["label"].astype(float) + 0.5), "integer-valued"),
        (lambda d: d.__setitem__("label", np.where(d["label"] == 0, 9, d["label"])), "codes documented"),
        (lambda d: d.__setitem__("subject", "S9"), "equals folder id"),
        (lambda d: d["signal"].pop("wrist"), "signal.wrist exists"),
    ],
)
def test_validation_rejects_structural_defects(raw_with_baseline, mutate, match):
    raw, manifests = raw_with_baseline
    d = pickle.load(open(raw / "WESAD" / "S2" / "S2.pkl", "rb"), encoding="latin1")
    mutate(d)
    with pytest.raises(LoaderValidationError, match=match):
        validate_participant_dict(d, "S2")
