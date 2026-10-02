"""Ablation runner: compare model arms against each other.

Proposal Section 6 requires an ablation comparing:
    naive speed-only baseline  vs.  kinematic-features  vs.  kinematic + video
plus the stratified (skin tone / lighting) evaluation from Section 7.
"""

from __future__ import annotations

import pandas as pd

from ..models.baseline import make_model
from .metrics import bootstrap_ci, compute_metrics, cross_validate


def run_ablation(
    features: pd.DataFrame,
    y: pd.Series,
    arms: list[dict] | None = None,
    **cv_kw,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Run each arm under identical signer-disjoint CV; returns summary table.

    Each arm is a dict like {"name": "naive", "kind": "naive_speed",
    "feature_set": "naive"}.
    """
    arms = arms or [
        {"name": "naive_speed_only", "kind": "naive_speed", "feature_set": "naive"},
        {"name": "kinematic_ridge", "kind": "ridge", "feature_set": "kinematic"},
        {"name": "kinematic_gbm", "kind": "gradient_boosting", "feature_set": "kinematic"},
    ]
    rows, details = [], {}
    for arm in arms:
        factory = lambda a=arm: make_model(a["kind"], a["feature_set"])
        oof, folds = cross_validate(factory, features, y, **cv_kw)
        y_true = y.reindex(oof.index).to_numpy(dtype=float)
        m = compute_metrics(y_true, oof.to_numpy())
        ci = bootstrap_ci(y_true, oof.to_numpy())
        rows.append({
            "arm": arm["name"],
            **m.as_dict(),
            "spearman_ci": ci["spearman"],
            "mae_ci": ci["mae"],
        })
        details[arm["name"]] = {"oof": oof, "folds": folds}
    return pd.DataFrame(rows), details


def stratified_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    context: pd.DataFrame,
    by: list[str],
) -> pd.DataFrame:
    """Metrics computed separately per stratum (skin tone x lighting ...).

    y_true/y_pred are indexed by clip_id; context has clip_id + strata cols.
    """
    df = context.set_index("clip_id")[by].join(
        pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    ).dropna(subset=["y_true", "y_pred"])
    rows = []
    for key, g in df.groupby(by, dropna=False):
        m = compute_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), **m.as_dict()})
    return pd.DataFrame(rows)
