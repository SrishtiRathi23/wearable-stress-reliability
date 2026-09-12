"""Participant-level split invariants.

The only function here today is the check that protects the non-negotiable
rule: no participant appears in both a training set and its held-out set.
Split *generation* (LOPO / grouped k-fold) will be added alongside the split
manifest writer once participant lists are verified by the dataset audits.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable


class ParticipantLeakageError(AssertionError):
    """Raised when the same participant would be seen in training and evaluation."""


def assert_participant_disjoint(train_ids: Iterable[Hashable], test_ids: Iterable[Hashable]) -> None:
    """Raise ParticipantLeakageError if any participant id is in both sets.

    Accepts any iterables of hashable ids (one entry per window/row is fine;
    duplicates within a set are expected and ignored).
    """
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise ParticipantLeakageError(
            f"{len(overlap)} participant(s) present in both train and test: {sorted(map(str, overlap))}"
        )
