"""Phase 3: participant-independent comparative baselines (D-028, D-029).

For every outer LOPO fold and every predeclared family:
  1. inner 4-fold StratifiedGroupKFold on the 14 outer-training participants
     (eligible rows only), seed from the `inner_split` stream + outer id,
     SAME folds for every candidate of every family;
  2. every predeclared candidate is fitted on inner-training rows and scored
     by the equal-weight mean of PARTICIPANT-level balanced accuracy over the
     14 inner-validation participants (threshold 0.5);
  3. the best candidate (earlier wins ties within 1e-12) is refitted on all
     eligible rows of the 14 outer-training participants;
  4. probabilities are produced for ALL complete windows of the held-out
     participant (eligible or not); metrics use eligible windows only.

Nothing about the held-out participant - rows, labels, prevalence - touches
imputation, scaling, candidate selection or fitting.

Usage:
    python -m wsr.experiments.baseline [--families majority logistic random_forest xgboost]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wsr.evaluation import metrics as M
from wsr.evaluation.splits import OuterFold, inner_split_to_dict, inner_stratified_group_folds, lopo_outer_splits
from wsr.features.schema import select_model_features
from wsr.models import grids as G
from wsr.models.majority import MajorityBaseline
from wsr.utils import integrity
from wsr.utils.config import load_config
from wsr.utils.paths import DATA_PROCESSED, MANIFESTS, RESULTS, ROOT
from wsr.utils.seeds import child_seed

RESULTS_PHASE3 = RESULTS / "phase3"
PARTICIPANT_COL, LABEL_COL, ELIG_COL, WINDOW_COL = "participant_id", "analysis_label", "binary_eligible", "window_id"
PROVENANCE_KEEP = ["dataset", "participant_id", "window_id", "window_index", "binary_eligible", "analysis_label", "raw_label_codes_present", "is_label_homogeneous", "homogeneous_raw_label", "condition_name"]


class Phase3InputError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def load_inputs(table_path: Path = DATA_PROCESSED / "wesad_windows_60s.parquet", schema_path: Path = MANIFESTS / "wesad_feature_schema.json", table_manifest_path: Path = MANIFESTS / "wesad_windows_60s_manifest.json") -> tuple[pd.DataFrame, dict, str, str]:
    """Load the canonical table + schema; refuse if the table hash differs from the Phase-2 manifest."""
    table_sha = integrity.sha256_file(table_path)
    manifest = json.loads(Path(table_manifest_path).read_text(encoding="utf-8"))
    if table_sha != manifest["table_sha256"]:
        raise Phase3InputError(f"feature table {table_path} sha256 {table_sha[:12]} != Phase-2 manifest {manifest['table_sha256'][:12]}; refusing")
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    schema_sha = integrity.sha256_file(schema_path)
    if schema["schema_version"] != manifest["schema_version"]:
        raise Phase3InputError("schema version differs from the Phase-2 manifest")
    table = pd.read_parquet(table_path)
    return table, schema, table_sha, schema_sha


def model_seed(global_seed: int, family: str, outer_pid: str) -> tuple[int, str]:
    stream = f"model_fit:{family}:{outer_pid}"
    return child_seed(global_seed, stream), stream


# --------------------------------------------------------------------------
# Per-fold procedures
# --------------------------------------------------------------------------


def _eligible_rows(table: pd.DataFrame, pids: tuple[str, ...] | list[str]) -> pd.DataFrame:
    return table[table[PARTICIPANT_COL].isin(list(pids)) & table[ELIG_COL]]


def evaluate_candidates(family: str, X: pd.DataFrame, table: pd.DataFrame, inner_split, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score every predeclared candidate on the frozen inner folds.

    Returns (candidate_summary_rows, per_validation_participant_rows).
    """
    summaries, detail = [], []
    for cand in G.GRIDS[family]:
        per_participant: dict[str, float] = {}
        for f in inner_split.folds:
            tr = _eligible_rows(table, f.train_participants)
            va = _eligible_rows(table, f.validation_participants)
            y_tr = tr[LABEL_COL].to_numpy().astype(int)
            pipe = G.build_pipeline(cand, seed, y_tr)
            pipe.fit(X.loc[tr.index], y_tr)
            prob = pipe.predict_proba(X.loc[va.index])[:, 1]
            pred = M.hard_predictions(prob)
            eff = G.effective_params(cand, y_tr)
            for pid in f.validation_participants:
                mask = (va[PARTICIPANT_COL] == pid).to_numpy()
                ba = M.participant_balanced_accuracy(va.loc[mask, LABEL_COL].to_numpy(), pred[mask], who=f"inner validation {pid}")
                if pid in per_participant:
                    raise RuntimeError(f"{pid} validated twice in inner folds")
                per_participant[pid] = ba
                detail.append(
                    {
                        "outer_test_participant": inner_split.outer_test_participant,
                        "family": family,
                        "candidate_id": cand.candidate_id,
                        "candidate_order": cand.order,
                        "inner_fold": f.fold_index,
                        "inner_train_participants": ",".join(f.train_participants),
                        "inner_validation_participants": ",".join(f.validation_participants),
                        "validation_participant": pid,
                        "participant_balanced_accuracy": ba,
                        "effective_params": json.dumps(eff, sort_keys=True),
                        "all_missing_features_in_inner_training": json.dumps(pipe.named_steps["impute"].all_missing_features_),
                        "model_seed": seed,
                    }
                )
        if len(per_participant) != len(inner_split.outer_train_participants):
            raise RuntimeError("inner validation did not cover every outer-training participant exactly once")
        summaries.append({"outer_test_participant": inner_split.outer_test_participant, "family": family, "candidate_id": cand.candidate_id, "candidate_order": cand.order, "params": json.dumps(cand.params, sort_keys=True), "participant_mean_balanced_accuracy": M.participant_level_mean(per_participant), "n_validation_participants": len(per_participant)})
    return summaries, detail


def refit_and_predict(family: str, cand: G.Candidate, X: pd.DataFrame, table: pd.DataFrame, fold: OuterFold, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Refit the selected candidate on all eligible outer-training rows; predict ALL windows of the outer participant."""
    tr = _eligible_rows(table, fold.train_participants)
    y_tr = tr[LABEL_COL].to_numpy().astype(int)
    pipe = G.build_pipeline(cand, seed, y_tr)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pipe.fit(X.loc[tr.index], y_tr)
    test_idx = table.index[table[PARTICIPANT_COL] == fold.test_participant]
    prob = pipe.predict_proba(X.loc[test_idx])[:, 1]
    info = {
        "effective_params": G.effective_params(cand, y_tr),
        "all_missing_features_in_refit_training": list(pipe.named_steps["impute"].all_missing_features_),
        "imputer_medians": {n: float(v) for n, v in zip(pipe.named_steps["impute"].feature_names_, pipe.named_steps["impute"].medians_)},
        "n_training_rows": int(len(tr)),
        "training_prevalence": float(y_tr.mean()),
        "fit_warnings": sorted({f"{x.category.__name__}: {str(x.message)[:120]}" for x in w}),
    }
    if "scale" in pipe.named_steps:
        sc = pipe.named_steps["scale"]
        info["scaler_mean_first5"] = [float(v) for v in sc.mean_[:5]]
    return prob, info


def majority_predict(table: pd.DataFrame, fold: OuterFold) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    tr = _eligible_rows(table, fold.train_participants)
    mb = MajorityBaseline().fit(tr[LABEL_COL].to_numpy().astype(int))
    n = int((table[PARTICIPANT_COL] == fold.test_participant).sum())
    return mb.predict_proba_positive(n), mb.predict(n), {"training_prevalence": mb.training_prevalence_, "hard_label": mb.hard_label_, "n_training_rows": int(len(tr))}


# --------------------------------------------------------------------------
# Whole experiment
# --------------------------------------------------------------------------


def run_phase3(table: pd.DataFrame, schema: dict, families: list[str], global_seed: int, n_inner: int, table_sha: str, schema_sha: str) -> dict[str, Any]:
    G.assert_grids_frozen()
    for fam in families:
        if fam not in G.FAMILIES:
            raise ValueError(f"unknown family {fam}")
    X = select_model_features(table, schema)  # exact schema allowlist; fails closed
    participants = sorted(table[PARTICIPANT_COL].unique(), key=lambda s: int(s[1:]))
    outer = lopo_outer_splits(participants)

    pred_rows: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    cand_summaries: list[dict[str, Any]] = []
    cand_detail: list[dict[str, Any]] = []
    inner_splits: list[dict[str, Any]] = []
    fit_counts: dict[str, int] = {f: 0 for f in families}
    all_missing_events: list[dict[str, Any]] = []
    fit_warnings: list[dict[str, Any]] = []

    for fold in outer:
        test_mask = table[PARTICIPANT_COL] == fold.test_participant
        prov = table.loc[test_mask, PROVENANCE_KEEP].reset_index(drop=True)
        train_elig = _eligible_rows(table, fold.train_participants)
        inner_split = inner_stratified_group_folds(train_elig, fold.test_participant, global_seed, n_inner) if any(f in G.LEARNED_FAMILIES for f in families) else None
        if inner_split is not None:
            inner_splits.append(inner_split_to_dict(inner_split))
        for fam in families:
            if fam == "majority":
                prob, pred, info = majority_predict(table, fold)
                sel = {"outer_fold": fold.fold_index, "outer_test_participant": fold.test_participant, "family": fam, "candidate_id": "majority", "candidate_params": json.dumps(info, sort_keys=True), "model_seed": None, "inner_score": None}
                fit_counts[fam] += 1
            else:
                seed, stream = model_seed(global_seed, fam, fold.test_participant)
                summ, det = evaluate_candidates(fam, X, table, inner_split, seed)
                fit_counts[fam] += len(G.GRIDS[fam]) * inner_split.n_splits
                cand_summaries += summ
                cand_detail += det
                best_i = G.select_best([s["participant_mean_balanced_accuracy"] for s in summ])
                cand = G.GRIDS[fam][best_i]
                prob, info = refit_and_predict(fam, cand, X, table, fold, seed)
                fit_counts[fam] += 1
                pred = M.hard_predictions(prob)
                if info["all_missing_features_in_refit_training"]:
                    all_missing_events.append({"outer_test_participant": fold.test_participant, "family": fam, "stage": "refit", "features": info["all_missing_features_in_refit_training"]})
                if info["fit_warnings"]:
                    fit_warnings.append({"outer_test_participant": fold.test_participant, "family": fam, "warnings": info["fit_warnings"]})
                sel = {"outer_fold": fold.fold_index, "outer_test_participant": fold.test_participant, "family": fam, "candidate_id": cand.candidate_id, "candidate_order": cand.order, "candidate_params": json.dumps(info["effective_params"], sort_keys=True), "model_seed": seed, "model_seed_stream": stream, "inner_score": summ[best_i]["participant_mean_balanced_accuracy"], "n_training_rows": info["n_training_rows"], "training_prevalence": info["training_prevalence"], "all_missing_features_refit": json.dumps(info["all_missing_features_in_refit_training"])}
            selections.append(sel)
            df = prov.copy()
            df.insert(4, "outer_fold", fold.fold_index)
            df.insert(5, "model_family", fam)
            df["prob_positive"] = prob.astype(np.float64)
            df["pred_threshold_0_5"] = pred.astype(np.int8)
            df["candidate_id"] = sel["candidate_id"]
            df["candidate_params"] = sel["candidate_params"]
            df["model_seed"] = pd.array([sel["model_seed"]] * len(df), dtype="Int64")
            df["feature_schema_version"] = schema["schema_version"]
            df["feature_table_sha256"] = table_sha
            pred_rows.append(df)
    predictions = pd.concat(pred_rows, ignore_index=True)
    validate_predictions(predictions, table, families)
    per_participant, summary = compute_metrics(predictions, families)
    return {
        "predictions": predictions,
        "selections": pd.DataFrame(selections),
        "candidate_scores": pd.DataFrame(cand_summaries),
        "candidate_detail": pd.DataFrame(cand_detail),
        "per_participant_metrics": per_participant,
        "model_summary": summary,
        "outer_splits": [{"fold_index": f.fold_index, "test_participant": f.test_participant, "train_participants": list(f.train_participants)} for f in outer],
        "inner_splits": inner_splits,
        "fit_counts": fit_counts,
        "all_missing_events": all_missing_events,
        "fit_warnings": fit_warnings,
        "participants": participants,
        "schema_sha256": schema_sha,
        "table_sha256": table_sha,
    }


def validate_predictions(pred: pd.DataFrame, table: pd.DataFrame, families: list[str]) -> None:
    n_windows = len(table)
    for fam in families:
        sub = pred[pred["model_family"] == fam]
        if len(sub) != n_windows:
            raise RuntimeError(f"{fam}: {len(sub)} prediction rows != {n_windows} complete windows")
        if set(sub[WINDOW_COL]) != set(table[WINDOW_COL]):
            raise RuntimeError(f"{fam}: prediction window ids differ from the table")
    if pred.duplicated([PARTICIPANT_COL, WINDOW_COL, "model_family"]).any():
        raise RuntimeError("duplicate (participant, window, model) rows")
    if not np.isfinite(pred["prob_positive"]).all():
        raise RuntimeError("non-finite probabilities")
    if set(pred["model_family"]) != set(families):
        raise RuntimeError("missing model family in predictions")


def compute_metrics(pred: pd.DataFrame, families: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for fam in families:
        for pid, g in pred[(pred["model_family"] == fam) & pred[ELIG_COL]].groupby(PARTICIPANT_COL, sort=False):
            # ALL hard-label metrics use the stored hard prediction (for learned models this is prob >= 0.5;
            # for the majority baseline it is its declared tie rule, prevalence exactly 0.5 -> 0).
            m = M.participant_metrics(g[LABEL_COL].to_numpy().astype(int), g["prob_positive"].to_numpy(), who=f"outer {pid}", y_pred=g["pred_threshold_0_5"].to_numpy())
            rows.append({"model_family": fam, "participant_id": pid, **m})
    per = pd.DataFrame(rows)
    summ_rows = []
    for fam in families:
        s = per[per["model_family"] == fam]
        row: dict[str, Any] = {"model_family": fam, "n_participants": int(len(s))}
        for met in ("balanced_accuracy", "macro_f1", "stress_recall", "baseline_recall", "auroc", "average_precision"):
            st = M.summarise_across_participants(dict(zip(s["participant_id"], s[met])))
            row[f"{met}_mean"] = st["mean"]
            row[f"{met}_median"] = st["median"]
            row[f"{met}_sd_across_participants"] = st["std"]
            row[f"{met}_min"] = st["min"]
            row[f"{met}_max"] = st["max"]
        summ_rows.append(row)
    summary = pd.DataFrame(summ_rows).sort_values("balanced_accuracy_mean", ascending=False, kind="stable").reset_index(drop=True)
    return per, summary


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------


def _git_state() -> dict[str, Any]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": sha, "dirty_tree": dirty}
    except Exception as exc:  # noqa: BLE001
        return {"commit": None, "dirty_tree": None, "error": repr(exc)}


def write_outputs(res: dict[str, Any], cfg: dict[str, Any], families: list[str], global_seed: int, out_dir: Path, manifests_dir: Path) -> dict[str, Any]:
    import sklearn
    import xgboost

    git_state = _git_state()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "oof_predictions.parquet"
    res["predictions"].to_parquet(pred_path, engine="pyarrow", index=False)
    res["per_participant_metrics"].to_csv(out_dir / "per_participant_metrics.csv", index=False, lineterminator="\n")
    res["model_summary"].to_csv(out_dir / "model_summary.csv", index=False, lineterminator="\n")
    res["selections"].to_csv(out_dir / "selected_hyperparameters.csv", index=False, lineterminator="\n")
    res["candidate_scores"].to_csv(out_dir / "inner_candidate_scores.csv", index=False, lineterminator="\n")
    res["candidate_detail"].to_csv(out_dir / "inner_candidate_participant_scores.csv", index=False, lineterminator="\n")

    grids = {fam: [{"order": c.order, "candidate_id": c.candidate_id, "params": c.params} for c in G.GRIDS[fam]] for fam in G.LEARNED_FAMILIES}
    common = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git": git_state,
        "feature_table_sha256": res["table_sha256"],
        "feature_schema_sha256": res["schema_sha256"],
        "feature_schema_version": res.get("schema_version"),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__},
    }
    splits_manifest = {**common, "global_seed": global_seed, "outer_split": "leave-one-participant-out", "inner_split": "StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=child_seed(seed, 'inner_split:<outer_pid>'))", "participants": res["participants"], "outer_folds": res["outer_splits"], "inner_splits": res["inner_splits"]}
    (manifests_dir / "phase3_splits.json").write_text(json.dumps(splits_manifest, indent=1) + "\n", encoding="utf-8")
    run_manifest = {
        **common,
        "families": families,
        "global_seed": global_seed,
        "seed_streams": {"inner_split": "inner_split:<outer_pid>", "model_fit": "model_fit:<family>:<outer_pid>"},
        "candidate_grids": grids,
        "fixed_params": {"logistic": {**G.LOGISTIC_FIXED, "penalty": G.LOGISTIC_PENALTY_NOTE}, "random_forest": G.RANDOM_FOREST_FIXED, "xgboost": G.XGBOOST_FIXED},
        "tie_breaking": "highest participant-mean inner balanced accuracy; ties within 1e-12 -> earlier candidate in predeclared order",
        "preprocessing": {"imputation": "median from the current training partition (TrainingMedianImputer); all-missing feature -> constant 0.0 and logged", "scaling": "StandardScaler fitted on the training partition, logistic only", "fit_scope": "inner-training rows for candidate evaluation; all 14 outer-training participants for refit"},
        "classification_threshold": 0.5,
        "majority_rule": "prob = training prevalence; hard = 1 if prevalence > 0.5 else 0 (0.5 -> 0)",
        "inner_score": "equal-weight mean of participant-level balanced accuracy over the 14 inner-validation participants",
        "metric_definitions": {"primary": "balanced_accuracy per outer participant on binary_eligible windows at threshold 0.5", "secondary": ["macro_f1", "baseline_recall", "stress_recall", "tn/fp/fn/tp", "auroc", "average_precision (not calibration evidence)"], "aggregation": "equal weight per participant; sd is across participants, never a window-level SE"},
        "fit_counts": res["fit_counts"],
        "n_fits_total": int(sum(res["fit_counts"].values())),
        "selected": res["selections"].to_dict(orient="records"),
        "all_missing_feature_events": res["all_missing_events"],
        "fit_warnings": res["fit_warnings"],
        "prediction_artifact": {"path": str(pred_path.relative_to(ROOT)) if pred_path.is_relative_to(ROOT) else str(pred_path), "sha256": integrity.sha256_file(pred_path), "n_rows": int(len(res["predictions"])), "rows_per_family": {f: int((res["predictions"]["model_family"] == f).sum()) for f in families}},
        "config_snapshot": {"evaluation": cfg.get("evaluation"), "phase3": cfg.get("phase3"), "seed": global_seed},
    }
    (manifests_dir / "phase3_run.json").write_text(json.dumps(run_manifest, indent=1, default=float) + "\n", encoding="utf-8")
    pred_manifest = {**common, "artifact": run_manifest["prediction_artifact"], "columns": list(res["predictions"].columns), "frozen": False, "note": "Set frozen=true after independent Phase-3 approval; Study A must verify sha256 before use and must never regenerate this file after seeing selective-label results."}
    (manifests_dir / "phase3_predictions_manifest.json").write_text(json.dumps(pred_manifest, indent=1) + "\n", encoding="utf-8")
    return run_manifest


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=list(G.FAMILIES))
    ap.add_argument("--out-dir", type=Path, default=RESULTS_PHASE3)
    ap.add_argument("--manifests-dir", type=Path, default=MANIFESTS)
    args = ap.parse_args(argv)
    cfg = load_config("wesad")
    if cfg["evaluation"]["outer_split"] != "leave-one-participant-out" or cfg["evaluation"]["inner_folds"] != 4 or cfg["phase3"]["classification_threshold"] != 0.5:
        print("config contradicts the frozen Phase-3 contract (D-028/D-029)", file=sys.stderr)
        return 2
    table, schema, table_sha, schema_sha = load_inputs()
    res = run_phase3(table, schema, args.families, int(cfg["seed"]), int(cfg["evaluation"]["inner_folds"]), table_sha, schema_sha)
    res["schema_version"] = schema["schema_version"]
    man = write_outputs(res, cfg, args.families, int(cfg["seed"]), args.out_dir, args.manifests_dir)
    print(f"fits={man['n_fits_total']} predictions={man['prediction_artifact']['n_rows']} sha={man['prediction_artifact']['sha256'][:16]} -> {args.out_dir}")
    print(res["model_summary"][["model_family", "balanced_accuracy_mean", "balanced_accuracy_median", "balanced_accuracy_min", "balanced_accuracy_max"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
