"""Participant-level split generation and invariants (D-006, D-028 / T-04).

Outer: leave-one-participant-out (LOPO). Inner: 4-fold StratifiedGroupKFold
(group = participant) computed on the outer-training rows only, with a seed
derived from the named `inner_split` stream and the outer participant id.

Every generated split is validated before use:
- outer train/test participant sets are disjoint and every participant is
  the outer test exactly once;
- inner folds contain only outer-training participants, inner train/
  validation are disjoint, and every outer-training participant is a
  validation participant exactly once across the folds.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from wsr.utils.seeds import child_seed


class ParticipantLeakageError(AssertionError):
    """Raised when the same participant would be seen in training and evaluation."""


class SplitValidationError(AssertionError):
    """Raised when a generated split violates a frozen invariant."""


def assert_participant_disjoint(train_ids: Iterable[Hashable], test_ids: Iterable[Hashable]) -> None:
    """Raise ParticipantLeakageError if any participant id is in both sets.

    Accepts any iterables of hashable ids (one entry per window/row is fine;
    duplicates within a set are expected and ignored).
    """
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise ParticipantLeakageError(f"{len(overlap)} participant(s) present in both train and test: {sorted(map(str, overlap))}")


# --------------------------------------------------------------------------
# Outer LOPO
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OuterFold:
    fold_index: int
    test_participant: str
    train_participants: tuple[str, ...]


def lopo_outer_splits(participants: list[str]) -> list[OuterFold]:
    """One fold per participant, in the given (deterministic) order."""
    if len(set(participants)) != len(participants):
        raise SplitValidationError("duplicate participant ids")
    folds = [OuterFold(i, p, tuple(q for q in participants if q != p)) for i, p in enumerate(participants)]
    validate_outer_splits(folds, participants)
    return folds


def validate_outer_splits(folds: list[OuterFold], participants: list[str]) -> None:
    tests = [f.test_participant for f in folds]
    if sorted(tests) != sorted(participants) or len(tests) != len(set(tests)):
        raise SplitValidationError("every participant must be the outer test participant exactly once")
    for f in folds:
        assert_participant_disjoint(f.train_participants, [f.test_participant])
        if set(f.train_participants) | {f.test_participant} != set(participants):
            raise SplitValidationError(f"outer fold {f.fold_index}: train + test != all participants")
        if len(f.train_participants) != len(participants) - 1:
            raise SplitValidationError(f"outer fold {f.fold_index}: expected {len(participants) - 1} training participants")


# --------------------------------------------------------------------------
# Inner StratifiedGroupKFold
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InnerFold:
    fold_index: int
    train_participants: tuple[str, ...]
    validation_participants: tuple[str, ...]


@dataclass(frozen=True)
class InnerSplit:
    outer_test_participant: str
    outer_train_participants: tuple[str, ...]
    n_splits: int
    seed: int
    seed_stream: str
    folds: tuple[InnerFold, ...] = field(default_factory=tuple)


def inner_seed(global_seed: int, outer_test_participant: str) -> tuple[int, str]:
    stream = f"inner_split:{outer_test_participant}"
    return child_seed(global_seed, stream), stream


def inner_stratified_group_folds(outer_train_rows: pd.DataFrame, outer_test_participant: str, global_seed: int, n_splits: int = 4, participant_col: str = "participant_id", label_col: str = "analysis_label") -> InnerSplit:
    """4-fold StratifiedGroupKFold on the OUTER-TRAINING eligible rows only.

    `outer_train_rows` must contain only outer-training participants and only
    binary-eligible rows (labels 0/1). The split is computed on rows (so
    stratification sees the label distribution) but recorded and validated at
    participant level; models later use the participant memberships.
    """
    pids = outer_train_rows[participant_col].to_numpy()
    if outer_test_participant in set(pids):
        raise ParticipantLeakageError(f"outer test participant {outer_test_participant} present in outer-training rows")
    y = outer_train_rows[label_col].to_numpy().astype(int)
    if set(np.unique(y)) - {0, 1}:
        raise SplitValidationError("inner split rows must carry binary analysis labels 0/1 only")
    seed, stream = inner_seed(global_seed, outer_test_participant)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for k, (tr, va) in enumerate(sgkf.split(np.zeros(len(y)), y, groups=pids)):
        tr_p = tuple(sorted(set(pids[tr])))
        va_p = tuple(sorted(set(pids[va])))
        folds.append(InnerFold(k, tr_p, va_p))
    split = InnerSplit(outer_test_participant, tuple(sorted(set(pids))), n_splits, seed, stream, tuple(folds))
    validate_inner_split(split)
    return split


def validate_inner_split(split: InnerSplit) -> None:
    outer_train = set(split.outer_train_participants)
    if split.outer_test_participant in outer_train:
        raise ParticipantLeakageError("outer test participant inside inner split")
    if len(split.folds) != split.n_splits:
        raise SplitValidationError(f"expected {split.n_splits} inner folds, got {len(split.folds)}")
    seen_val: list[str] = []
    for f in split.folds:
        assert_participant_disjoint(f.train_participants, f.validation_participants)
        if not set(f.train_participants) <= outer_train or not set(f.validation_participants) <= outer_train:
            raise ParticipantLeakageError(f"inner fold {f.fold_index} uses a participant outside the outer-training set")
        if set(f.train_participants) | set(f.validation_participants) != outer_train:
            raise SplitValidationError(f"inner fold {f.fold_index}: train + validation != outer-training set")
        if not f.validation_participants:
            raise SplitValidationError(f"inner fold {f.fold_index} has no validation participant")
        seen_val.extend(f.validation_participants)
    if sorted(seen_val) != sorted(outer_train):
        raise SplitValidationError("every outer-training participant must be an inner validation participant exactly once")


def inner_split_to_dict(split: InnerSplit) -> dict:
    return {
        "outer_test_participant": split.outer_test_participant,
        "outer_train_participants": list(split.outer_train_participants),
        "n_splits": split.n_splits,
        "seed": split.seed,
        "seed_stream": split.seed_stream,
        "folds": [{"fold_index": f.fold_index, "train_participants": list(f.train_participants), "validation_participants": list(f.validation_participants)} for f in split.folds],
    }


def inner_split_from_dict(d: dict) -> InnerSplit:
    split = InnerSplit(
        d["outer_test_participant"],
        tuple(d["outer_train_participants"]),
        int(d["n_splits"]),
        int(d["seed"]),
        d["seed_stream"],
        tuple(InnerFold(f["fold_index"], tuple(f["train_participants"]), tuple(f["validation_participants"])) for f in d["folds"]),
    )
    validate_inner_split(split)
    return split
