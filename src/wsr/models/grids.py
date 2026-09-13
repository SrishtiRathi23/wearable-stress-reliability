"""Frozen Phase-3 model families, candidate grids and pipeline construction (D-029).

The grids are CODE CONSTANTS, not configuration: the runner cannot expand
them from config or defaults. Candidate order is the predeclared Cartesian
order (simpler/more regularised first) and is the tie-breaking order.

Fixed (non-grid) settings are exactly those in the Phase-3 specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from wsr.models.preprocessing import TrainingMedianImputer

FAMILIES = ("majority", "logistic", "random_forest", "xgboost")
LEARNED_FAMILIES = ("logistic", "random_forest", "xgboost")
TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Candidate:
    family: str
    candidate_id: str  # e.g. "logistic:C=0.1,class_weight=None"
    params: dict[str, Any]
    order: int


def _cands(family: str, combos: list[dict[str, Any]]) -> tuple[Candidate, ...]:
    out = []
    for i, p in enumerate(combos):
        cid = family + ":" + ",".join(f"{k}={v}" for k, v in p.items())
        out.append(Candidate(family, cid, dict(p), i))
    return tuple(out)


# Logistic Regression: C outer loop (lower first), class_weight inner (None before balanced) -> 6
LOGISTIC_GRID = _cands("logistic", [{"C": C, "class_weight": cw} for C in (0.1, 1.0, 10.0) for cw in (None, "balanced")])
# Random Forest: Cartesian order max_depth (3,6) x min_samples_leaf (5,2) x class_weight (None, balanced) -> 8
RANDOM_FOREST_GRID = _cands("random_forest", [{"max_depth": d, "min_samples_leaf": l, "class_weight": cw} for d in (3, 6) for l in (5, 2) for cw in (None, "balanced")])
# XGBoost: max_depth (2,3) x learning_rate (0.03,0.10) x class_weight_mode (none, training_ratio) -> 8
XGBOOST_GRID = _cands("xgboost", [{"max_depth": d, "learning_rate": lr, "class_weight_mode": m} for d in (2, 3) for lr in (0.03, 0.10) for m in ("none", "training_ratio")])

GRIDS: dict[str, tuple[Candidate, ...]] = {"logistic": LOGISTIC_GRID, "random_forest": RANDOM_FOREST_GRID, "xgboost": XGBOOST_GRID}
EXPECTED_GRID_SIZES = {"logistic": 6, "random_forest": 8, "xgboost": 8}

# penalty is L2: sklearn's default for LogisticRegression (the explicit `penalty` kwarg is
# deprecated since 1.8 and is therefore not passed; liblinear + default penalty == L2).
LOGISTIC_FIXED = {"solver": "liblinear", "max_iter": 5000}
LOGISTIC_PENALTY_NOTE = "l2 (sklearn default; explicit penalty kwarg deprecated in sklearn 1.8, so not passed)"
RANDOM_FOREST_FIXED = {"n_estimators": 300, "max_features": "sqrt", "criterion": "gini", "bootstrap": True, "n_jobs": 1}
XGBOOST_FIXED = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "tree_method": "hist",
    "n_jobs": 1,
}


def assert_grids_frozen() -> None:
    for fam, n in EXPECTED_GRID_SIZES.items():
        if len(GRIDS[fam]) != n:
            raise RuntimeError(f"{fam} grid has {len(GRIDS[fam])} candidates; frozen size is {n}")
        if [c.order for c in GRIDS[fam]] != list(range(n)):
            raise RuntimeError(f"{fam} grid order corrupted")


def training_scale_pos_weight(y_train: np.ndarray) -> float:
    """negatives / positives computed from the CURRENT training partition only."""
    y = np.asarray(y_train).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        raise ValueError("training partition has no positive windows; scale_pos_weight undefined")
    return n_neg / n_pos


def effective_params(cand: Candidate, y_train: np.ndarray) -> dict[str, Any]:
    """Grid params plus any training-derived setting (XGBoost scale_pos_weight)."""
    p = dict(cand.params)
    if cand.family == "xgboost":
        mode = p.pop("class_weight_mode")
        p["scale_pos_weight"] = training_scale_pos_weight(y_train) if mode == "training_ratio" else 1.0
        p["class_weight_mode"] = mode
    return p


def build_pipeline(cand: Candidate, seed: int, y_train: np.ndarray) -> Pipeline:
    """Imputation (training-only) -> [scaling for LR] -> estimator, all fitted together on the training partition."""
    if cand.family == "logistic":
        est = LogisticRegression(C=cand.params["C"], class_weight=cand.params["class_weight"], random_state=seed, **LOGISTIC_FIXED)
        return Pipeline([("impute", TrainingMedianImputer()), ("scale", StandardScaler()), ("model", est)])
    if cand.family == "random_forest":
        est = RandomForestClassifier(max_depth=cand.params["max_depth"], min_samples_leaf=cand.params["min_samples_leaf"], class_weight=cand.params["class_weight"], random_state=seed, **RANDOM_FOREST_FIXED)
        return Pipeline([("impute", TrainingMedianImputer()), ("model", est)])
    if cand.family == "xgboost":
        eff = effective_params(cand, y_train)
        est = XGBClassifier(max_depth=eff["max_depth"], learning_rate=eff["learning_rate"], scale_pos_weight=eff["scale_pos_weight"], random_state=seed, **XGBOOST_FIXED)
        return Pipeline([("impute", TrainingMedianImputer()), ("model", est)])
    raise ValueError(f"unknown learned family {cand.family!r}")


def select_best(scores_in_order: list[float]) -> int:
    """Index of the best candidate; ties within TIE_TOLERANCE go to the earlier candidate."""
    best_i, best = 0, scores_in_order[0]
    for i, s in enumerate(scores_in_order[1:], start=1):
        if s > best + TIE_TOLERANCE:
            best_i, best = i, s
    return best_i
