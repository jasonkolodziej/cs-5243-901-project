"""Fluency-scoring models.

Stage 4 of the proposal pipeline:

    NaiveSpeedBaseline  -- predict fluency from letters-per-second alone
                           (the naive baseline the hypothesis must beat).
    Ridge / RF / GBM    -- interpretable scorers over kinematic features,
                           appropriate for the small rated set (Section 7
                           warns against overfitting end-to-end deep models).
    FusedModel          -- optional kinematic + raw-video embedding fusion
                           (ablation arm; lives in models.fused, torch extra).

All models share fit/predict(X: DataFrame) -> np.ndarray and respect the
feature_set column selection from features.pipeline.FEATURE_SETS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..features.pipeline import FEATURE_SETS


class NaiveSpeedBaseline:
    """Linear map from letters-per-second to the 1-5 fluency rating."""

    def __init__(self) -> None:
        self._lr = LinearRegression()

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> NaiveSpeedBaseline:
        self._lr.fit(X[["letters_per_second"]], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._lr.predict(X[["letters_per_second"]])


def make_model(kind: str, feature_set: str = "kinematic", seed: int = 42):
    """Factory returning an unfitted model for the given config."""
    cols = FEATURE_SETS[feature_set]

    if kind == "naive_speed":
        return NaiveSpeedBaseline()

    class _Reg:
        def __init__(self, est):
            self.est = make_pipeline(StandardScaler(), est)

        def fit(self, X, y):
            self.est.fit(X[cols].to_numpy(dtype=float), y)
            return self

        def predict(self, X):
            return self.est.predict(X[cols].to_numpy(dtype=float))

    if kind == "ridge":
        return _Reg(Ridge(alpha=1.0))
    if kind == "random_forest":
        return _Reg(RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1))
    if kind == "gradient_boosting":
        return _Reg(GradientBoostingRegressor(random_state=seed))
    raise ValueError(f"unknown model kind: {kind}")
