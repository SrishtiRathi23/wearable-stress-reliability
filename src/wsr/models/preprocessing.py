"""Training-partition-only preprocessing (D-029).

`TrainingMedianImputer` wraps sklearn's SimpleImputer(strategy="median",
keep_empty_features=True): statistics come from the rows it is fitted on
and nothing else. A feature that is entirely missing in the fitted
partition is imputed with the frozen constant 0.0 and its name is recorded
in `all_missing_features_` so the run manifest can log the event. The
imputer never looks at validation/test rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

ALL_MISSING_FILL_VALUE = 0.0


class TrainingMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.imputer_: SimpleImputer | None = None
        self.feature_names_: list[str] | None = None
        self.all_missing_features_: list[str] = []
        self.medians_: np.ndarray | None = None

    def fit(self, X, y=None):  # noqa: N803 - sklearn API
        Xa = self._as_array(X)
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(Xa.shape[1])]
        all_missing = np.all(np.isnan(Xa), axis=0)
        self.all_missing_features_ = [n for n, m in zip(self.feature_names_, all_missing) if m]
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        self.imputer_.fit(Xa)
        stats = np.array(self.imputer_.statistics_, dtype=np.float64)
        # sklearn imputes empty features with 0 for numerical data; make the frozen rule explicit and verifiable
        stats[all_missing] = ALL_MISSING_FILL_VALUE
        self.imputer_.statistics_ = stats
        self.medians_ = stats
        return self

    def transform(self, X):  # noqa: N803
        if self.imputer_ is None:
            raise RuntimeError("TrainingMedianImputer must be fitted first")
        Xa = self._as_array(X)
        if Xa.shape[1] != len(self.feature_names_ or []):
            raise ValueError("feature count differs from the fitted partition")
        return self.imputer_.transform(Xa)

    @staticmethod
    def _as_array(X) -> np.ndarray:  # noqa: N803
        return np.asarray(X, dtype=np.float64)
