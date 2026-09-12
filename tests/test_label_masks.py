"""Observation-mask invariants (research invariant #5).

When mask builders exist, every non-'full_reference' policy must produce an
identical mask when the held-out labels are permuted, proving the mask does
not depend on test labels.

Required test design (independent review, 2026-09-12; see KI-06): the check
must rebuild the ENTIRE candidate-construction pipeline (episode boundaries,
detector scores, durations, ranking, mask) from raw held-out inputs under
permuted labels. Permuting labels only after candidate metadata already
exists would miss upstream leakage.
"""

import pytest


@pytest.mark.skip(reason="Observation-mask builders not implemented yet (Study A). Must rebuild candidates from scratch under permuted labels, see module docstring.")
def test_candidates_and_masks_are_invariant_to_permuted_test_labels():
    raise NotImplementedError


@pytest.mark.skip(reason="Detector for detector_triggered policy not implemented yet.")
def test_detector_is_trained_without_held_out_participant():
    raise NotImplementedError
