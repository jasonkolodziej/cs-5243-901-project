"""Fluency-scoring models.

Stage 4 of the proposal pipeline:

    MeanBaseline        -- predict the training-fold mean rating; the
                           "no information" floor every arm's MAE must beat.
    NaiveSpeedBaseline  -- predict fluency from letters-per-second alone
                           (the naive baseline the hypothesis must beat).
    RidgeCV             -- ridge whose shrinkage is chosen by efficient LOO
                           inside each training fold (nested, no test leakage).
                           The fixed alpha=1 ridge under-regularizes ~18
                           features on ~50 clips and overfits.
    OrdinalScorer       -- proportional-odds (ordered logit) model; respects
                           that 1-5 ratings are ordered categories, not
                           interval-scaled numbers. Predicts E[rating].
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
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
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


RIDGE_CV_ALPHAS = np.logspace(-2, 4, 25)


class MeanBaseline:
    """Predict the training mean — the reference MAE every arm must beat."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> MeanBaseline:
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_)


class OrdinalScorer:
    """Proportional-odds (ordered logit) model over standardized features.

    Predicts the expected rating sum_k k * P(rating = k), so it plugs into the
    same Spearman/MAE metrics as the regressors. Falls back to the training
    mean if the fold has <2 rating levels or the optimizer fails — a tiny
    training fold must not crash a whole ablation run.
    """

    def __init__(self, cols: list[str]) -> None:
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> OrdinalScorer:
        import warnings

        from statsmodels.miscmodels.ordinal_model import OrderedModel

        y = np.asarray(y, float)
        self.mean_ = float(np.mean(y))
        self.scaler_ = StandardScaler().fit(X[self.cols].to_numpy(dtype=float))
        # Consensus means can be fractional (e.g. 3.67); ordinal levels need
        # integers, so round for fitting. E[rating] stays continuous.
        levels = np.rint(y).astype(int)
        self.levels_ = np.unique(levels)
        self.res_ = None
        if len(self.levels_) < 2:
            return self
        Z = self.scaler_.transform(X[self.cols].to_numpy(dtype=float))
        endog = pd.Categorical(levels, categories=self.levels_, ordered=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.res_ = OrderedModel(endog, Z, distr="logit").fit(
                    method="bfgs", disp=False, maxiter=500
                )
        except (np.linalg.LinAlgError, ValueError):
            self.res_ = None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.res_ is None:
            return np.full(len(X), self.mean_)
        Z = self.scaler_.transform(X[self.cols].to_numpy(dtype=float))
        probs = np.asarray(self.res_.model.predict(self.res_.params, exog=Z))
        return probs @ self.levels_.astype(float)


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

    if kind == "mean":
        return MeanBaseline()
    if kind == "naive_speed":
        return NaiveSpeedBaseline()
    if kind == "ridge":
        return SklearnRegressor(Ridge(alpha=1.0), cols)
    if kind == "ridge_cv":
        return SklearnRegressor(RidgeCV(alphas=RIDGE_CV_ALPHAS), cols)
    if kind == "ordinal":
        return OrdinalScorer(cols)
    if kind == "random_forest":
        return SklearnRegressor(
            RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1), cols
        )
    if kind == "gradient_boosting":
        return SklearnRegressor(GradientBoostingRegressor(random_state=seed), cols)
    raise ValueError(f"unknown model kind: {kind}")
