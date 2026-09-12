"""Phase-2 closeout: sealed loader handle, real-valued inputs, config contract, ACC flags, allowlist."""

from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.wesad_fixture import T_S, make_fake_wesad
from wsr.data.load_wesad import LoaderValidationError, RawIntegrityError, VerifiedRelease, load_participant, validate_participant_dict
from wsr.features import activity
from wsr.features.build_features import FEATURE_DEFS, build, params_from_config, validate_config_contract
from wsr.features.schema import FeatureContractError, model_feature_columns_from_schema, schema_document, select_model_features
from wsr.preprocessing.quality import QualityParams, acc_flags, modality_flags
from wsr.preprocessing.windowing import ConfigContractError, binary_reference_from_config
from wsr.utils import integrity
from wsr.utils.config import load_config


@pytest.fixture
def raw_with_baseline(tmp_path: Path):
    raw = tmp_path / "raw" / "wesad"
    (raw / "WESAD").mkdir(parents=True)
    make_fake_wesad(raw / "WESAD", "S2")
    manifests = tmp_path / "manifests"
    integrity.snapshot(raw, manifests)
    return raw, manifests


# --- 2. sealed VerifiedRelease ---------------------------------------------------------


def test_unverified_caller_cannot_construct_release(raw_with_baseline):
    raw, manifests = raw_with_baseline
    pkl_rel = "WESAD/S2/S2.pkl"
    fake_hash = integrity.sha256_file(raw / pkl_rel)  # even the CORRECT hash must not grant a handle
    with pytest.raises(RawIntegrityError, match="cannot be constructed directly"):
        VerifiedRelease(raw, manifests, raw / "WESAD", {pkl_rel: fake_hash})
    with pytest.raises(RawIntegrityError):
        VerifiedRelease(raw, manifests, raw / "WESAD", {pkl_rel: fake_hash}, _seal=object())
    # bypassing __init__ via __new__ leaves no usable state either
    obj = VerifiedRelease.__new__(VerifiedRelease)
    with pytest.raises(AttributeError):
        load_participant("S2", obj)


def test_release_state_is_not_caller_mutable(raw_with_baseline):
    raw, manifests = raw_with_baseline
    rel = VerifiedRelease.open(raw, manifests)
    assert not hasattr(rel, "expected_hashes")
    with pytest.raises(AttributeError):
        rel._expected = {"WESAD/S2/S2.pkl": "0" * 64}  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        rel.raw_root = raw.parent  # type: ignore[misc]
    assert rel.n_files == 5 and len(rel.expected_hash("WESAD/S2/S2.pkl")) == 64
    with pytest.raises(RawIntegrityError):
        rel.expected_hash("WESAD/S9/S9.pkl")
    # tampering AFTER open is still caught by the per-file re-hash
    d = pickle.load(open(raw / "WESAD" / "S2" / "S2.pkl", "rb"), encoding="latin1")
    d["label"] = np.full_like(d["label"], 2)
    with open(raw / "WESAD" / "S2" / "S2.pkl", "wb") as fh:
        pickle.dump(d, fh)
    with pytest.raises(RawIntegrityError, match="hash"):
        load_participant("S2", rel)


# --- 3. real-valued inputs + label shape ------------------------------------------------


def _fixture_dict(raw: Path) -> dict:
    return pickle.load(open(raw / "WESAD" / "S2" / "S2.pkl", "rb"), encoding="latin1")


def test_complex_valued_signals_rejected(raw_with_baseline):
    raw, _ = raw_with_baseline
    for mod in ("ACC", "BVP", "EDA", "TEMP"):
        d = _fixture_dict(raw)
        d["signal"]["wrist"][mod] = d["signal"]["wrist"][mod].astype(np.complex128)
        with pytest.raises(LoaderValidationError, match="real-valued"):
            validate_participant_dict(d, "S2")
    d = _fixture_dict(raw)
    d["label"] = d["label"].astype(np.complex128)
    with pytest.raises(LoaderValidationError, match="real-valued"):
        validate_participant_dict(d, "S2")
    d = _fixture_dict(raw)
    d["signal"]["wrist"]["EDA"] = d["signal"]["wrist"]["EDA"].astype(bool)
    with pytest.raises(LoaderValidationError, match="real-valued"):
        validate_participant_dict(d, "S2")


def test_label_shape_not_silently_flattened(raw_with_baseline):
    raw, _ = raw_with_baseline
    d = _fixture_dict(raw)
    d["label"] = d["label"].reshape(-1, 1)  # (n,1) is accepted
    validate_participant_dict(d, "S2")
    d["label"] = d["label"].reshape(700, T_S)  # (700, 120) has the right size but wrong form
    with pytest.raises(LoaderValidationError, match="expected a vector"):
        validate_participant_dict(d, "S2")


# --- 4. config contract -----------------------------------------------------------------


@pytest.mark.parametrize(
    "path, value, match",
    [
        (("device",), "chest", "device"),
        (("windowing", "length_s"), 30, "length_s"),
        (("windowing", "step_s"), 30, "step_s"),
        (("windowing", "anchor"), "first_label", "anchor"),
        (("windowing", "grid_uses_labels"), True, "grid_uses_labels"),
        (("windowing", "eligibility_rule"), "majority_vote", "eligibility_rule"),
        (("windowing", "mixed_window_policy"), "relabel_majority", "mixed_window_policy"),
        (("features", "families"), ["activity", "eda"], "families"),
        (("features", "hrv", "enabled"), True, "hrv"),
        (("features", "acc", "counts_per_g"), 32, "counts_per_g"),
        (("labels", "positive_code"), 3, "positive_code"),
        (("labels", "negative_code"), 0, "negative_code"),
        (("labels", "ineligible_codes"), [3, 4, 5, 6, 7], "ineligible_codes"),
        (("labels", "preserve_raw_label"), False, "preserve_raw_label"),
    ],
)
def test_contradictory_frozen_config_rejected(path, value, match):
    cfg = copy.deepcopy(load_config("wesad"))
    cur = cfg
    for k in path[:-1]:
        cur = cur[k]
    cur[path[-1]] = value
    with pytest.raises((ConfigContractError, ValueError), match=match):
        validate_config_contract(cfg)


def test_config_drives_mapping_and_counts_per_g():
    cfg = load_config("wesad")
    params = params_from_config(cfg)
    assert params["binary_reference"] == {1: 0, 2: 1} and params["counts_per_g"] == 64.0
    assert binary_reference_from_config(cfg["labels"]) == {1: 0, 2: 1}
    with pytest.raises(ValueError, match="counts_per_g"):
        activity.counts_per_g_from_config({"counts_per_g": 100})


# --- 5. ACC quality flags --------------------------------------------------------------


def test_acc_temporal_constant_per_axis():
    p = QualityParams()
    rep = np.tile([0.0, 0.0, 64.0], (1920, 1))  # temporally constant, pooled range 64
    assert modality_flags("acc", rep, 32.0, 60.0, p)["q_acc_constant"] is True
    one_axis_moves = rep.copy()
    one_axis_moves[5, 0] = 1.0
    assert modality_flags("acc", one_axis_moves, 32.0, 60.0, p)["q_acc_constant"] is False
    assert modality_flags("eda", np.full(240, 0.7), 4.0, 60.0, p)["q_eda_constant"] is True


def test_acc_near_rail_flag_renamed_and_informational():
    p = QualityParams()
    calm = np.full((1920, 3), 20.0)
    assert acc_flags(calm, p) == {"q_acc_near_rail": False}
    calm[100, 1] = -127.0
    assert acc_flags(calm, p) == {"q_acc_near_rail": True}
    assert "q_acc_clipped" not in acc_flags(calm, p)
    doc = schema_document(FEATURE_DEFS, {})
    names = {c["name"] for c in doc["columns"]}
    assert "q_acc_near_rail" in names and "q_acc_clipped" not in names
    assert doc["schema_version"] == "1.1.0"


# --- 11. schema-driven allowlist -------------------------------------------------------


def test_schema_allowlist_is_exactly_the_59_model_features_and_blocks_provisional():
    doc = schema_document(FEATURE_DEFS, {})
    allow = model_feature_columns_from_schema(doc)
    assert len(allow) == 59 and len(set(allow)) == 59
    roles = {c["name"]: c["role"] for c in doc["columns"]}
    assert all(roles[c] == "feature" for c in allow)
    for prov in ("hr_mean", "hr_std", "bvp_beat_count", "bvp_beat_coverage", "bvp_valid_ibi_count"):
        assert prov not in allow and roles[prov] == "feature_provisional"
    for forbidden in ("participant_id", "window_id", "window_index", "start_seconds", "analysis_label", "binary_eligible", "raw_label_codes_present", "source_sha256", "q_structural_ok", "q_acc_near_rail"):
        assert forbidden not in allow
    tampered = copy.deepcopy(doc)
    tampered["model_feature_columns"].append("hr_mean")
    with pytest.raises(FeatureContractError, match="disagrees"):
        model_feature_columns_from_schema(tampered)


def test_select_model_features_fails_on_disagreement(raw_with_baseline):
    raw, manifests = raw_with_baseline
    rel = VerifiedRelease.open(raw, manifests)
    cfg = load_config("wesad")
    table, _ = build(rel, ["S2"], params_from_config(cfg))
    doc = schema_document(FEATURE_DEFS, {})
    X = select_model_features(table, doc)
    assert list(X.columns) == doc["model_feature_columns"] and X.shape == (len(table), 59)
    with pytest.raises(FeatureContractError, match="missing"):
        select_model_features(table.drop(columns=["eda_mean"]), doc)
    dup = pd.concat([table, table[["temp_mean"]]], axis=1)
    with pytest.raises(FeatureContractError, match="duplicate"):
        select_model_features(dup, doc)
    with pytest.raises(FeatureContractError, match="differ from the committed schema"):
        select_model_features(table.rename(columns={"q_acc_near_rail": "q_acc_clipped"}), doc)
    with pytest.raises(FeatureContractError, match="differ"):
        select_model_features(table.assign(extra=1.0), doc)


def test_committed_schema_file_agrees_with_code():
    from wsr.utils.paths import MANIFESTS

    p = MANIFESTS / "wesad_feature_schema.json"
    if not p.is_file():
        pytest.skip("schema not built")
    committed = json.loads(p.read_text(encoding="utf-8"))
    assert model_feature_columns_from_schema(committed) == schema_document(FEATURE_DEFS, {})["model_feature_columns"]
    assert committed["schema_version"] == "1.1.0"
