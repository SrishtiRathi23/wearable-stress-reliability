"""Phase-3 framework tests on synthetic schema-conformant data (no real WESAD needed)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.phase3_fixture import make_synthetic_table
from wsr.evaluation import metrics as M
from wsr.evaluation.splits import (
    ParticipantLeakageError,
    SplitValidationError,
    inner_split_from_dict,
    inner_split_to_dict,
    inner_stratified_group_folds,
    lopo_outer_splits,
    validate_inner_split,
    validate_outer_splits,
)
from wsr.experiments import baseline as B
from wsr.features.schema import FeatureContractError
from wsr.models import grids as G
from wsr.models.majority import MajorityBaseline
from wsr.models.preprocessing import TrainingMedianImputer


@pytest.fixture(scope="module")
def synth():
    return make_synthetic_table(n_participants=6, n_windows=30, seed=0)


@pytest.fixture(scope="module")
def run_ml(synth):
    t, s = synth
    return B.run_phase3(t, s, ["majority", "logistic"], 42, 4, "t" * 64, "s" * 64)


# --- 25. split safety -----------------------------------------------------------


def test_outer_lopo_each_participant_exactly_once_and_disjoint():
    pids = [f"S{i}" for i in (2, 3, 4, 5, 6, 7)]
    folds = lopo_outer_splits(pids)
    assert [f.test_participant for f in folds] == pids  # A
    for f in folds:
        assert f.test_participant not in f.train_participants and len(f.train_participants) == 5  # B
    with pytest.raises(SplitValidationError):
        validate_outer_splits(folds[:-1], pids)
    with pytest.raises(SplitValidationError):
        lopo_outer_splits(["S2", "S2", "S3"])


def test_inner_folds_only_outer_training_disjoint_and_cover_once(synth):
    t, _ = synth
    outer = lopo_outer_splits(sorted(t.participant_id.unique(), key=lambda s: int(s[1:])))
    for fold in outer:
        rows = t[t.participant_id.isin(fold.train_participants) & t.binary_eligible]
        split = inner_stratified_group_folds(rows, fold.test_participant, 42, 4)
        assert split.outer_test_participant not in set(split.outer_train_participants)
        seen = []
        for f in split.folds:
            assert set(f.train_participants) <= set(fold.train_participants) and set(f.validation_participants) <= set(fold.train_participants)  # C
            assert not set(f.train_participants) & set(f.validation_participants)  # D
            assert fold.test_participant not in f.train_participants + f.validation_participants
            seen += f.validation_participants
        assert sorted(seen) == sorted(fold.train_participants)  # E
        # round-trip through the manifest form re-validates
        assert inner_split_from_dict(inner_split_to_dict(split)) == split


def test_inner_split_rejects_leaky_or_malformed_input(synth):
    t, _ = synth
    rows = t[t.binary_eligible]  # includes S2 which we claim is the outer test participant
    with pytest.raises(ParticipantLeakageError):
        inner_stratified_group_folds(rows, "S2", 42, 4)
    good = inner_stratified_group_folds(rows[rows.participant_id != "S2"], "S2", 42, 4)
    d = inner_split_to_dict(good)
    d["folds"][0]["validation_participants"] = d["folds"][1]["validation_participants"]  # a participant validated twice (and now also in fold-0 train)
    with pytest.raises((SplitValidationError, ParticipantLeakageError)):
        inner_split_from_dict(d)
    d = inner_split_to_dict(good)
    d["folds"] = d["folds"][:3]  # a participant never validated
    with pytest.raises(SplitValidationError):
        inner_split_from_dict(d)
    d = inner_split_to_dict(good)
    d["folds"][0]["train_participants"].append("S2")  # outer test leaks into inner train
    with pytest.raises(ParticipantLeakageError):
        inner_split_from_dict(d)


def test_inner_seed_depends_on_outer_participant_and_is_deterministic(synth):
    t, _ = synth
    rows = t[t.binary_eligible]
    a = inner_stratified_group_folds(rows[rows.participant_id != "S2"], "S2", 42, 4)
    b = inner_stratified_group_folds(rows[rows.participant_id != "S2"], "S2", 42, 4)
    c = inner_stratified_group_folds(rows[rows.participant_id != "S3"], "S3", 42, 4)
    assert a == b and a.seed != c.seed and a.seed_stream == "inner_split:S2"


def test_same_inner_folds_reused_across_families(synth):
    t, s = synth
    res = B.run_phase3(t[t.participant_id.isin(["S2", "S3", "S4", "S5", "S6"])], s, ["logistic", "random_forest"], 42, 4, "t" * 64, "s" * 64)
    d = res["candidate_detail"]
    for pid, g in d.groupby("outer_test_participant"):
        folds_by_family = {fam: sorted(set(zip(gg.inner_fold, gg.inner_train_participants, gg.inner_validation_participants))) for fam, gg in g.groupby("family")}
        assert folds_by_family["logistic"] == folds_by_family["random_forest"]  # F
    assert len(res["inner_splits"]) == 5


def test_participant_level_aggregation_differs_from_fold_mean():
    # folds of sizes 2 and 1: participant-level mean weights each participant equally
    scores = {"S2": 1.0, "S3": 0.5, "S4": 0.0}  # fold A = {S2, S3}, fold B = {S4}
    fold_means = [np.mean([1.0, 0.5]), 0.0]
    assert M.participant_level_mean(scores) == pytest.approx(0.5)
    assert np.mean(fold_means) == pytest.approx(0.375)
    assert M.participant_level_mean(scores) != np.mean(fold_means)  # G


def test_single_class_validation_participant_is_an_error():
    with pytest.raises(M.SingleClassParticipantError):
        M.participant_balanced_accuracy(np.array([0, 0, 0]), np.array([0, 1, 0]))
    with pytest.raises(M.SingleClassParticipantError):
        M.participant_metrics(np.array([1, 1]), np.array([0.2, 0.9]))


# --- 26. preprocessing / label leakage --------------------------------------------


def test_imputer_and_scaler_fit_training_rows_only():
    X_tr = pd.DataFrame({"a": [1.0, 3.0, np.nan], "b": [np.nan, np.nan, np.nan]})
    X_te = pd.DataFrame({"a": [100.0, np.nan], "b": [50.0, np.nan]})
    imp = TrainingMedianImputer().fit(X_tr)
    assert imp.all_missing_features_ == ["b"] and imp.medians_.tolist() == [2.0, 0.0]
    out = imp.transform(X_te)
    assert out.tolist() == [[100.0, 50.0], [2.0, 0.0]]  # test values never change the fitted medians
    cand = G.LOGISTIC_GRID[0]
    pipe = G.build_pipeline(cand, 1, np.array([0, 1, 0]))
    Xf = pd.DataFrame(np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]), columns=["a", "b"])
    pipe.fit(Xf, np.array([0, 1, 0]))
    assert pipe.named_steps["scale"].mean_.tolist() == [2.0, 3.0]


def test_changing_outer_test_labels_changes_nothing_but_metrics(synth):
    t, s = synth
    res1 = B.run_phase3(t, s, ["majority", "logistic"], 42, 4, "t" * 64, "s" * 64)
    t2 = t.copy()
    mask = (t2.participant_id == "S4") & t2.binary_eligible
    t2.loc[mask, "analysis_label"] = (1 - t2.loc[mask, "analysis_label"].astype(int)).astype("Int8")  # flip S4's reference labels
    res2 = B.run_phase3(t2, s, ["majority", "logistic"], 42, 4, "t" * 64, "s" * 64)
    sel1 = res1["selections"][res1["selections"].outer_test_participant == "S4"]
    sel2 = res2["selections"][res2["selections"].outer_test_participant == "S4"]
    assert sel1.candidate_id.tolist() == sel2.candidate_id.tolist() and sel1.candidate_params.tolist() == sel2.candidate_params.tolist()
    p1 = res1["predictions"][res1["predictions"].participant_id == "S4"].sort_values(["model_family", "window_id"])
    p2 = res2["predictions"][res2["predictions"].participant_id == "S4"].sort_values(["model_family", "window_id"])
    assert np.array_equal(p1.prob_positive.to_numpy(), p2.prob_positive.to_numpy())
    m1 = res1["per_participant_metrics"].query("participant_id == 'S4' and model_family == 'logistic'").balanced_accuracy.iloc[0]
    m2 = res2["per_participant_metrics"].query("participant_id == 'S4' and model_family == 'logistic'").balanced_accuracy.iloc[0]
    assert m1 != m2  # only the evaluation changed
    # S4's flipped labels never entered any other fold's fitting either: other participants' predictions identical
    o1 = res1["predictions"][res1["predictions"].participant_id != "S4"].sort_values(["model_family", "window_id"])
    o2 = res2["predictions"][res2["predictions"].participant_id != "S4"].sort_values(["model_family", "window_id"])
    assert not np.array_equal(o1.prob_positive.to_numpy(), o2.prob_positive.to_numpy())  # S4 IS training data for others, so this must differ


# --- 27. model input safety ---------------------------------------------------------


def test_exactly_schema_features_enter_models(synth, monkeypatch):
    t, s = synth
    seen: list[list[str]] = []
    orig = G.build_pipeline

    def spy(cand, seed, y):
        pipe = orig(cand, seed, y)
        orig_fit = pipe.fit

        def fit(X, y):
            seen.append(list(X.columns))
            return orig_fit(X, y)

        pipe.fit = fit
        return pipe

    monkeypatch.setattr(B.G, "build_pipeline", spy)
    B.run_phase3(t[t.participant_id.isin(["S2", "S3", "S4", "S5", "S6"])], s, ["logistic"], 42, 4, "t" * 64, "s" * 64)
    assert seen and all(cols == s["model_feature_columns"] for cols in seen) and len(seen[0]) == 59
    forbidden = {"participant_id", "window_id", "window_index", "start_seconds", "analysis_label", "binary_eligible", "condition_name", "q_structural_ok", "source_sha256", "hr_mean", "bvp_beat_count", "bvp_beat_coverage"}
    assert not forbidden & set(seen[0])


def test_missing_schema_feature_fails_closed(synth):
    t, s = synth
    with pytest.raises(FeatureContractError):
        B.run_phase3(t.drop(columns=["eda_mean"]), s, ["majority"], 42, 4, "t" * 64, "s" * 64)


# --- 28. grids -------------------------------------------------------------------------


def test_grid_sizes_order_and_tie_breaking():
    G.assert_grids_frozen()
    assert len(G.LOGISTIC_GRID) == 6 and len(G.RANDOM_FOREST_GRID) == 8 and len(G.XGBOOST_GRID) == 8
    assert [c.params for c in G.LOGISTIC_GRID[:3]] == [{"C": 0.1, "class_weight": None}, {"C": 0.1, "class_weight": "balanced"}, {"C": 1.0, "class_weight": None}]
    assert G.RANDOM_FOREST_GRID[0].params == {"max_depth": 3, "min_samples_leaf": 5, "class_weight": None}
    assert G.XGBOOST_GRID[0].params == {"max_depth": 2, "learning_rate": 0.03, "class_weight_mode": "none"}
    assert G.select_best([0.7, 0.7 + 5e-13, 0.69]) == 0  # tie within 1e-12 -> earlier
    assert G.select_best([0.7, 0.7 + 1e-9]) == 1
    assert G.select_best([0.5, 0.5, 0.5]) == 0
    assert G.training_scale_pos_weight(np.array([0, 0, 0, 1])) == 3.0
    eff = G.effective_params(G.XGBOOST_GRID[1], np.array([0, 0, 1]))
    assert eff["scale_pos_weight"] == 2.0 and eff["class_weight_mode"] == "training_ratio"
    assert G.effective_params(G.XGBOOST_GRID[0], np.array([0, 0, 1]))["scale_pos_weight"] == 1.0


def test_grids_cannot_be_expanded_by_mutation():
    with pytest.raises(AttributeError):
        G.LOGISTIC_GRID.append(G.LOGISTIC_GRID[0])  # tuple
    with pytest.raises(AttributeError):  # dataclasses.FrozenInstanceError
        G.LOGISTIC_GRID[0].params = {}  # frozen dataclass


# --- 29. prediction completeness ----------------------------------------------------------


def test_predictions_cover_every_window_and_metrics_use_eligible_only(run_ml, synth):
    t, _ = synth
    p = run_ml["predictions"]
    for fam in ("majority", "logistic"):
        sub = p[p.model_family == fam]
        assert len(sub) == len(t) and set(sub.window_id) == set(t.window_id)
        assert sub.prob_positive.notna().all()  # ineligible windows have probabilities too
        assert (~sub.binary_eligible).sum() == (~t.binary_eligible).sum()
    assert not p.duplicated(["participant_id", "window_id", "model_family"]).any()
    assert set(p.model_family) == {"majority", "logistic"}
    per = run_ml["per_participant_metrics"]
    assert per.n_windows.eq(20).all()  # 12 baseline + 8 stress eligible per synthetic participant; ineligible excluded
    assert per.groupby("model_family").size().tolist() == [6, 6]
    assert set(p.columns) >= {"outer_fold", "candidate_id", "candidate_params", "model_seed", "feature_schema_version", "feature_table_sha256", "raw_label_codes_present", "homogeneous_raw_label"}


def test_majority_baseline_rules():
    mb = MajorityBaseline().fit(np.array([0, 0, 0, 1]))
    assert mb.training_prevalence_ == 0.25 and mb.hard_label_ == 0 and mb.predict_proba_positive(3).tolist() == [0.25] * 3
    assert MajorityBaseline().fit(np.array([0, 1])).hard_label_ == 0  # exactly 0.5 -> 0
    assert MajorityBaseline().fit(np.array([1, 1, 0])).hard_label_ == 1


def test_all_families_run_and_fit_counts(synth):
    t, s = synth
    small = t[t.participant_id.isin(["S2", "S3", "S4", "S5", "S6"])]
    res = B.run_phase3(small, s, list(G.FAMILIES), 42, 4, "t" * 64, "s" * 64)
    assert res["fit_counts"] == {"majority": 5, "logistic": 6 * 4 * 5 + 5, "random_forest": 8 * 4 * 5 + 5, "xgboost": 8 * 4 * 5 + 5}
    assert len(res["predictions"]) == 4 * len(small)
    sel = res["selections"]
    assert set(sel.family) == set(G.FAMILIES) and sel.query("family == 'xgboost'").candidate_params.str.contains("scale_pos_weight").all()
    assert res["model_summary"].model_family.tolist()[0] in G.FAMILIES and len(res["model_summary"]) == 4


# --- 30. reproducibility ----------------------------------------------------------------


def test_two_runs_identical(synth):
    t, s = synth
    a = B.run_phase3(t, s, ["majority", "logistic"], 42, 4, "t" * 64, "s" * 64)
    b = B.run_phase3(t, s, ["majority", "logistic"], 42, 4, "t" * 64, "s" * 64)
    pd.testing.assert_frame_equal(a["predictions"], b["predictions"])
    pd.testing.assert_frame_equal(a["selections"], b["selections"])
    assert a["inner_splits"] == b["inner_splits"]


def test_write_outputs_and_manifests(run_ml, synth, tmp_path: Path):
    _, s = synth
    run_ml["schema_version"] = s["schema_version"]
    cfg = {"evaluation": {"outer_split": "leave-one-participant-out"}, "phase3": {}}
    man = B.write_outputs(run_ml, cfg, ["majority", "logistic"], 42, tmp_path / "res", tmp_path / "man")
    assert (tmp_path / "res" / "oof_predictions.parquet").is_file() and man["prediction_artifact"]["n_rows"] == len(run_ml["predictions"])
    splits = json.loads((tmp_path / "man" / "phase3_splits.json").read_text())
    assert len(splits["outer_folds"]) == 6 and len(splits["inner_splits"]) == 6
    run = json.loads((tmp_path / "man" / "phase3_run.json").read_text())
    assert run["candidate_grids"]["logistic"][0]["params"] == {"C": 0.1, "class_weight": None} and run["classification_threshold"] == 0.5
    assert run["prediction_artifact"]["sha256"] == __import__("wsr.utils.integrity", fromlist=["sha256_file"]).sha256_file(tmp_path / "res" / "oof_predictions.parquet")
