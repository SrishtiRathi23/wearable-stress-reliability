"""Time-only windowing and reference annotation (D-021, invariant #19)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wsr.preprocessing.windowing import (
    PROVENANCE_COLUMNS,
    REFERENCE_COLUMNS,
    WindowSpec,
    annotate_reference_labels,
    build_window_frame,
    n_complete_windows,
    omitted_tail_s,
    sample_slice,
)

RATES = {"ACC": 32.0, "BVP": 64.0, "EDA": 4.0, "TEMP": 4.0}
LABEL_RATE = 700.0


def test_native_sample_counts_for_60s_window():
    """C: ACC 1920, BVP 3840, EDA 240, TEMP 240, label 42000 for one 60-s window."""
    f = build_window_frame("S2", 120.0, RATES, LABEL_RATE)
    r = f.iloc[0]
    assert (r.acc_end_sample - r.acc_start_sample, r.bvp_end_sample - r.bvp_start_sample) == (1920, 3840)
    assert (r.eda_end_sample - r.eda_start_sample, r.temp_end_sample - r.temp_start_sample) == (240, 240)
    assert r.end_label_sample - r.start_label_sample == 42000
    assert sample_slice(60.0, 120.0, 32.0) == (1920, 3840)


def test_fixed_t0_origin_and_non_overlap():
    """D: origins 0, 60, 120, ...; half-open, contiguous, never anchored elsewhere."""
    f = build_window_frame("S2", 300.0, RATES, LABEL_RATE)
    assert f.start_seconds.tolist() == [0.0, 60.0, 120.0, 180.0, 240.0]
    assert (f.end_seconds - f.start_seconds).eq(60.0).all()
    assert (f.start_label_sample.to_numpy()[1:] == f.end_label_sample.to_numpy()[:-1]).all()
    assert f.window_id.tolist()[:2] == ["wesad:S2:w00000", "wesad:S2:w00001"] and f.window_index.tolist() == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="anchored at synchronised t=0"):
        WindowSpec(origin_s=5.0)


def test_incomplete_tail_dropped_not_padded():
    """J: 6079 s -> 101 complete windows, 19 s tail reported, nothing padded."""
    assert n_complete_windows(6079.0, WindowSpec()) == 101
    assert omitted_tail_s(6079.0, WindowSpec()) == 19.0
    f = build_window_frame("S2", 6079.0, RATES, LABEL_RATE)
    assert len(f) == 101 and f.end_seconds.max() == 6060.0 and f.end_label_sample.max() == 6060 * 700
    assert n_complete_windows(59.9, WindowSpec()) == 0 and len(build_window_frame("S2", 59.9, RATES, LABEL_RATE)) == 0
    assert n_complete_windows(120.0, WindowSpec()) == 2 and omitted_tail_s(120.0, WindowSpec()) == 0.0


def _labels(spec: list[tuple[int, int, int]], total_s: int) -> np.ndarray:
    lab = np.zeros(700 * total_s, dtype=np.int32)
    for code, s, e in spec:
        lab[700 * s : 700 * e] = code
    return lab


def test_boundaries_identical_when_labels_permuted():
    """E + Section 19: the frame is built from duration/rates only; permuting labels changes only annotation."""
    duration = 601.0
    frame_a = build_window_frame("S2", duration, RATES, LABEL_RATE)
    lab = _labels([(1, 30, 250), (2, 300, 500)], 601)
    rng = np.random.default_rng(0)
    lab_perm = rng.permutation(lab)
    lab_const = np.full_like(lab, 3)
    frame_b = build_window_frame("S2", duration, RATES, LABEL_RATE)  # rebuilt from scratch
    prov = list(PROVENANCE_COLUMNS) + [c for c in frame_a.columns if c.endswith("_sample")]
    pd.testing.assert_frame_equal(frame_a[prov], frame_b[prov])
    ann_a = annotate_reference_labels(frame_a, lab)
    ann_p = annotate_reference_labels(frame_b, lab_perm)
    ann_c = annotate_reference_labels(frame_b, lab_const)
    for ann in (ann_p, ann_c):
        pd.testing.assert_frame_equal(ann_a[prov], ann[prov])
        assert len(ann) == len(ann_a) == 10
    assert not ann_a[list(REFERENCE_COLUMNS)].equals(ann_p[list(REFERENCE_COLUMNS)])
    assert ann_c.condition_name.eq("amusement").all() and not ann_c.binary_eligible.any()


def test_eligibility_rules():
    """F/G/H/I: homogeneous 1 -> baseline 0; homogeneous 2 -> stress 1; other codes and mixed windows ineligible, never majority-voted."""
    lab = _labels([(1, 0, 60), (2, 60, 120), (3, 120, 180), (4, 180, 240), (0, 240, 300), (1, 300, 359), (2, 359, 360), (2, 360, 361), (1, 361, 420)], 420)
    ann = annotate_reference_labels(build_window_frame("S2", 420.0, RATES, LABEL_RATE), lab)
    assert ann.condition_name.tolist() == ["baseline", "stress", "amusement", "meditation", "transient", "mixed", "mixed"]
    assert ann.binary_eligible.tolist() == [True, True, False, False, False, False, False]
    assert ann.analysis_label.tolist()[:2] == [0, 1] and ann.analysis_label.isna().tolist()[2:] == [True] * 5
    assert ann.is_label_homogeneous.tolist() == [True] * 5 + [False, False]
    assert ann.homogeneous_raw_label.tolist()[:5] == [1, 2, 3, 4, 0]
    assert ann.ineligibility_reason.tolist()[2:] == ["homogeneous_code_3_not_in_binary_reference", "homogeneous_code_4_not_in_binary_reference", "homogeneous_code_0_not_in_binary_reference", "mixed_label_codes", "mixed_label_codes"]
    # window 5 is 59 s baseline + 1 s stress: majority would say baseline; it must stay ineligible
    assert ann.raw_label_codes_present.tolist()[5] == "1,2" and ann.n_label_samples.tolist()[5] == 42000
    # window 6 is 1 s stress + 59 s baseline
    assert ann.raw_label_codes_present.tolist()[6] == "1,2"


def test_reference_codes_present_are_sorted_and_complete():
    lab = _labels([(7, 0, 10), (2, 10, 20), (0, 20, 60)], 60)
    ann = annotate_reference_labels(build_window_frame("S2", 60.0, RATES, LABEL_RATE), lab)
    assert ann.raw_label_codes_present.tolist() == ["0,2,7"]
