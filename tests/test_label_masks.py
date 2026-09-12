"""Observation-mask invariants (research invariant #5).

When mask builders exist, every non-'full_reference' policy must produce an
identical mask when the held-out labels are permuted, proving the mask does
not depend on test labels.
"""

import pytest


@pytest.mark.skip(reason="Observation-mask builders not implemented yet (Study A).")
def test_masks_are_invariant_to_permuted_test_labels():
    raise NotImplementedError


@pytest.mark.skip(reason="Detector for detector_triggered policy not implemented yet.")
def test_detector_is_trained_without_held_out_participant():
    raise NotImplementedError
