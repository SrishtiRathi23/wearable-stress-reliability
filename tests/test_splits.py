"""Participant-independent evaluation invariant (research invariant #2/#6)."""

import pytest

from wsr.evaluation.splits import ParticipantLeakageError, assert_participant_disjoint


def test_disjoint_participants_pass():
    assert_participant_disjoint(["S2", "S3", "S3"], ["S4"])


def test_overlapping_participants_raise():
    with pytest.raises(ParticipantLeakageError, match="S3"):
        assert_participant_disjoint(["S2", "S3"], ["S3", "S4"])


@pytest.mark.skip(reason="Split generator (LOPO / grouped k-fold) not implemented yet; add once participant lists are verified by the WESAD audit.")
def test_generated_outer_splits_are_participant_disjoint():
    raise NotImplementedError
