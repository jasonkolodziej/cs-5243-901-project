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


class SklearnRegressor:
    """StandardScaler + sklearn regressor over a fixed feature-set column list.

    Module-level (not a closure) so fitted instances are picklable — the
    `train` command persists them with joblib.
    """

    def __init__(self, estimator, cols: list[str]) -> None:
        self.cols = cols
        self.est = make_pipeline(StandardScaler(), estimator)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> SklearnRegressor:
        self.est.fit(X[self.cols].to_numpy(dtype=float), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.est.predict(X[self.cols].to_numpy(dtype=float))


def make_model(kind: str, feature_set: str = "kinematic", seed: int = 42):
    """Factory returning an unfitted model for the given config."""
    cols = FEATURE_SETS[feature_set]

    if kind == "naive_speed":
        return NaiveSpeedBaseline()
    if kind == "ridge":
        return SklearnRegressor(Ridge(alpha=1.0), cols)
    if kind == "random_forest":
        return SklearnRegressor(
            RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1), cols
        )
    if kind == "gradient_boosting":
        return SklearnRegressor(GradientBoostingRegressor(random_state=seed), cols)
    raise ValueError(f"unknown model kind: {kind}")
