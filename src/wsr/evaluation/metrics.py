"""Participant-level classification metrics (Phase 3, D-029).

All metrics are computed on one participant's binary-eligible windows at a
time; aggregation across participants is done with equal weight per
participant. Windows are never pooled across participants for a headline
number.

Threshold is fixed at 0.5 (`hard_predictions`). A validation participant
with only one reference class makes balanced accuracy undefined; that is an
error here, not a silently redefined metric.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score

THRESHOLD = 0.5


class SingleClassParticipantError(ValueError):
    """A participant's reference labels contain only one class; the metric is undefined."""


def hard_predictions(prob_positive: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    return (np.asarray(prob_positive, dtype=np.float64) >= threshold).astype(np.int8)


def _check_binary(y_true: np.ndarray, who: str) -> np.ndarray:
    y = np.asarray(y_true).astype(int)
    if y.size == 0:
        raise SingleClassParticipantError(f"{who}: no reference labels")
    classes = set(np.unique(y).tolist())
    if not classes <= {0, 1}:
        raise ValueError(f"{who}: labels must be 0/1, got {sorted(classes)}")
    if len(classes) < 2:
        raise SingleClassParticipantError(f"{who}: only class {classes.pop()} present; balanced accuracy undefined")
    return y


def participant_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, who: str = "participant") -> float:
    y = _check_binary(y_true, who)
    return float(balanced_accuracy_score(y, np.asarray(y_pred).astype(int)))


def participant_metrics(y_true: np.ndarray, prob_positive: np.ndarray, who: str = "participant", threshold: float = THRESHOLD) -> dict[str, float]:
    """Full per-participant metric set on eligible windows only."""
    y = _check_binary(y_true, who)
    p = np.asarray(prob_positive, dtype=np.float64)
    pred = hard_predictions(p, threshold)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n_windows": int(y.size),
        "n_baseline": int((y == 0).sum()),
        "n_stress": int((y == 1).sum()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "baseline_recall": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "stress_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
    }


def participant_level_mean(scores_by_participant: dict[str, float]) -> float:
    """Equal-weight mean over participants (never over folds or windows)."""
    if not scores_by_participant:
        raise ValueError("no participant scores")
    return float(np.mean(list(scores_by_participant.values())))


def summarise_across_participants(values: dict[str, float]) -> dict[str, float]:
    v = np.asarray(list(values.values()), dtype=np.float64)
    return {
        "n_participants": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "std": float(v.std(ddof=1)) if v.size > 1 else float("nan"),  # sample SD across participants; NOT a window-level SE
        "min": float(v.min()),
        "max": float(v.max()),
    }
