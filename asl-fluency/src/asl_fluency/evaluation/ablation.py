"""Ablation runner: compare model arms against each other.

Proposal Section 6 requires an ablation comparing:
    naive speed-only baseline  vs.  kinematic-features  vs.  kinematic + video
plus the stratified (skin tone / lighting) evaluation from Section 7.

Pre-registered analysis plan (fixed before instructor ratings arrive):
    primary test   PRIMARY_ARM vs. BASELINE_ARM, paired-bootstrap delta
                   Spearman on out-of-fold predictions (one-sided).
    references     mean_baseline (no-information MAE floor).
    secondary      every other arm — reported, never used to pick a winner
                   after looking at the held-out numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.baseline import make_model
from .metrics import bootstrap_ci, compute_metrics, cross_validate, paired_bootstrap_delta

PRIMARY_ARM = "core_ridge_cv"
BASELINE_ARM = "naive_speed_only"

DEFAULT_ARMS = [
    {"name": "mean_baseline", "kind": "mean", "feature_set": "naive"},
    {"name": BASELINE_ARM, "kind": "naive_speed", "feature_set": "naive"},
    {"name": PRIMARY_ARM, "kind": "ridge_cv", "feature_set": "core"},
    {"name": "core_ordinal", "kind": "ordinal", "feature_set": "core"},
    {"name": "kinematic_robust_ridge_cv", "kind": "ridge_cv",
     "feature_set": "kinematic_robust"},
    # Original arms, kept for comparability: fixed alpha=1 ridge and GBM over
    # ~18 features overfit at n~50 — reported, not used as the primary test.
    {"name": "kinematic_ridge", "kind": "ridge", "feature_set": "kinematic"},
    {"name": "kinematic_robust_ridge", "kind": "ridge",
     "feature_set": "kinematic_robust"},
    {"name": "kinematic_gbm", "kind": "gradient_boosting", "feature_set": "kinematic"},
]


def run_ablation(
    features: pd.DataFrame,
    y: pd.Series,
    arms: list[dict] | None = None,
    bootstrap_iters: int = 1000,
    **cv_kw,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Run each arm under identical signer-disjoint CV; returns summary table.

    Each arm is a dict like {"name": "naive", "kind": "naive_speed",
    "feature_set": "naive"}. When the baseline arm is present, every row also
    gets a paired-bootstrap delta Spearman vs. that baseline (the hypothesis
    test), its 95% CI, and a one-sided p-value.
    """
    arms = arms or DEFAULT_ARMS
    rows, details = [], {}
    for arm in arms:
        factory = lambda a=arm: make_model(a["kind"], a["feature_set"])
        oof, folds = cross_validate(factory, features, y, **cv_kw)
        y_true = y.reindex(oof.index).to_numpy(dtype=float)
        m = compute_metrics(y_true, oof.to_numpy()).as_dict()
        ci = bootstrap_ci(y_true, oof.to_numpy(), iters=bootstrap_iters)
        if arm["kind"] == "mean":
            # Fold means are anti-correlated with held-out ratings by
            # construction (dropping high-rated clips lowers the fold mean),
            # so this arm's correlation is an artifact: it is an MAE floor only.
            m["spearman"] = m["pearson"] = np.nan
            ci["spearman"] = (np.nan, np.nan)
        rows.append({
            "arm": arm["name"],
            **m,
            "spearman_ci": ci["spearman"],
            "mae_ci": ci["mae"],
        })
        details[arm["name"]] = {"oof": oof, "folds": folds, "kind": arm["kind"]}

    summary = pd.DataFrame(rows)
    if BASELINE_ARM in details:
        base = details[BASELINE_ARM]["oof"]
        tests = []
        for name, d in details.items():
            oof = d["oof"]
            y_true = y.reindex(oof.index).to_numpy(dtype=float)
            if name == BASELINE_ARM:
                tests.append({"delta_spearman": 0.0, "delta_ci": (0.0, 0.0),
                              "p_vs_naive": np.nan})
                continue
            if d["kind"] == "mean":
                tests.append({"delta_spearman": np.nan, "delta_ci": (np.nan, np.nan),
                              "p_vs_naive": np.nan})
                continue
            t = paired_bootstrap_delta(y_true, oof.to_numpy(),
                                       base.reindex(oof.index).to_numpy(),
                                       iters=bootstrap_iters)
            tests.append({"delta_spearman": t["delta"],
                          "delta_ci": (t["ci_lo"], t["ci_hi"]),
                          "p_vs_naive": t["p_one_sided"]})
        summary = pd.concat([summary, pd.DataFrame(tests)], axis=1)
    return summary, details


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


def marginal_stratified_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    context: pd.DataFrame,
    by: list[str],
) -> pd.DataFrame:
    """Per-column strata, stacked (column, stratum, metrics...).

    With ~50 rated clips the joint skin_tone x lighting grid leaves most cells
    below the 3 clips a correlation needs, so each factor is sliced on its own.
    Signed error (mean y_pred - y_true) is included: a systematic over/under-
    score for one stratum is the bias Section 7 warns about, and correlation
    alone cannot show it.
    """
    out = []
    for col in by:
        if col not in context.columns:
            continue
        t = stratified_metrics(y_true, y_pred, context, [col])
        df = context.set_index("clip_id")[[col]].join(
            pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        ).dropna(subset=["y_true", "y_pred"])
        bias = (df["y_pred"] - df["y_true"]).groupby(df[col], dropna=False).mean()
        t["mean_signed_error"] = t[col].map(bias)
        t.insert(0, "stratum", t.pop(col).astype(str))
        t.insert(0, "factor", col)
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
