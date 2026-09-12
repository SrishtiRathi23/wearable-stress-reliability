"""End-to-end leakage guards for preprocessing/model/calibration fitting.

These will assert that scalers, imputers, calibrators and thresholds are fit
on development participants only (research invariants #2, #3, #4).
"""

import pytest


@pytest.mark.skip(reason="Feature pipeline not implemented yet; add once scaler/imputer fitting exists.")
def test_preprocessing_fit_excludes_held_out_participant():
    raise NotImplementedError


@pytest.mark.skip(reason="Calibration / abstention not implemented yet.")
def test_thresholds_and_calibration_use_development_labels_only():
    raise NotImplementedError
