"""Majority / prevalence reference baseline (D-029).

For an outer fold: training_prevalence = mean(analysis_label) over the
eligible windows of the outer-training participants. Every held-out window
gets that constant as its positive-class probability; the hard prediction
is 1 if training_prevalence > 0.5 else 0 (exactly 0.5 -> 0 by predeclared
tie rule). No inner tuning.
"""

from __future__ import annotations

import numpy as np


class MajorityBaseline:
    def __init__(self) -> None:
        self.training_prevalence_: float | None = None
        self.hard_label_: int | None = None

    def fit(self, y_train: np.ndarray) -> "MajorityBaseline":
        y = np.asarray(y_train).astype(int)
        if y.size == 0:
            raise ValueError("empty training partition")
        self.training_prevalence_ = float(y.mean())
        self.hard_label_ = 1 if self.training_prevalence_ > 0.5 else 0
        return self

    def predict_proba_positive(self, n: int) -> np.ndarray:
        if self.training_prevalence_ is None:
            raise RuntimeError("fit first")
        return np.full(n, self.training_prevalence_, dtype=np.float64)

    def predict(self, n: int) -> np.ndarray:
        if self.hard_label_ is None:
            raise RuntimeError("fit first")
        return np.full(n, self.hard_label_, dtype=np.int8)
