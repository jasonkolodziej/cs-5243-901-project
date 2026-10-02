"""Evaluation metrics from proposal Section 6.

    - Spearman / Pearson correlation between predicted and human fluency
    - MAE on held-out rated clips
    - bootstrap confidence intervals for both
    - signer-disjoint cross-validation driver
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class Metrics:
    spearman: float
    pearson: float
    mae: float
    n: int

    def as_dict(self) -> dict[str, float]:
        return {
            "spearman": self.spearman,
            "pearson": self.pearson,
            "mae": self.mae,
            "n": float(self.n),
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 3:
        return Metrics(np.nan, np.nan, np.nan, len(y_true))
    return Metrics(
        spearman=float(stats.spearmanr(y_true, y_pred).statistic),
        pearson=float(stats.pearsonr(y_true, y_pred).statistic),
        mae=float(np.mean(np.abs(y_true - y_pred))),
        n=len(y_true),
    )


def bootstrap_ci(
    y_true: np.ndarray, y_pred: np.ndarray, iters: int = 1000, seed: int = 42
) -> dict[str, tuple[float, float]]:
    """95% bootstrap CIs for each metric."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    n = len(y_true)
    acc: dict[str, list[float]] = {"spearman": [], "pearson": [], "mae": []}
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        m = compute_metrics(y_true[idx], y_pred[idx])
        for k, v in acc.items():
            v.append(getattr(m, k))
    return {
        k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
        for k, v in acc.items()
    }


def cross_validate(
    model_factory,
    features: pd.DataFrame,
    y: pd.Series,
    *,
    group_col: str | None = "signer_id",
    n_splits: int = 5,
) -> tuple[pd.Series, pd.DataFrame]:
    """Out-of-fold predictions, signer-disjoint when group_col is present.

    Returns (oof predictions as a Series indexed by clip_id, per-fold metrics).
    """
    from sklearn.model_selection import GroupKFold, KFold

    X = features.set_index("clip_id") if "clip_id" in features.columns else features
    y = y.reindex(X.index).to_numpy(dtype=float)
    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]

    if group_col and group_col in X.columns and X[group_col].notna().all():
        splitter = GroupKFold(n_splits).split(X, y, X[group_col])
    else:
        splitter = KFold(n_splits, shuffle=True, random_state=42).split(X, y)

    oof = pd.Series(np.full(len(X), np.nan), index=X.index, name="y_pred")
    fold_rows = []
    for fold, (tr, te) in enumerate(splitter):
        model = model_factory()
        model.fit(X.iloc[tr], y[tr])
        pred = model.predict(X.iloc[te])
        oof.iloc[te] = pred
        m = compute_metrics(y[te], pred)
        fold_rows.append({"fold": fold, **m.as_dict()})
    return oof, pd.DataFrame(fold_rows)
