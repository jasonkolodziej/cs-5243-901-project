"""Inter-rater agreement for the human fluency labels.

Proposal Section 6: report Cohen's / Fleiss' kappa or ICC on the collected
ratings as a ground-truth sanity check before trusting the labels.

    cohen_kappa   -- two raters, ordinal weights supported
    fleiss_kappa  -- 3+ raters, per-subject category counts
    icc           -- ICC(2, k) via a two-way random-effects ANOVA, the usual
                     choice for mean-of-raters fluency ground truth
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cohen_kappa(r1: pd.Series, r2: pd.Series, weights: str = "quadratic") -> float:
    """Weighted Cohen's kappa for two raters (quadratic = ordinal-aware)."""
    from sklearn.metrics import cohen_kappa_score

    pair = pd.concat([r1, r2], axis=1).dropna()
    if len(pair) < 2:
        return np.nan
    return float(cohen_kappa_score(pair.iloc[:, 0], pair.iloc[:, 1], weights=weights))


def fleiss_kappa(matrix: pd.DataFrame, categories: list[int] | None = None) -> float:
    """Fleiss' kappa on a wide ratings matrix (rows=clips, cols=raters)."""
    cats = categories or sorted(pd.unique(matrix.to_numpy().ravel()[~pd.isna(matrix.to_numpy().ravel())]))
    n_raters = matrix.notna().sum(axis=1).to_numpy(float)
    counts = np.zeros((len(matrix), len(cats)))
    for j, c in enumerate(cats):
        counts[:, j] = (matrix == c).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p_i = ((counts**2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_i = p_i[~np.isnan(p_i)]
    p_bar = p_i.mean()
    p_j = counts.sum(axis=0) / counts.sum()
    p_e = (p_j**2).sum()
    return float((p_bar - p_e) / (1 - p_e))


def icc_2k(matrix: pd.DataFrame) -> float:
    """ICC(2, k): two-way random effects, absolute agreement, mean of k raters.

    Computed from the two-way ANOVA mean squares of the wide matrix
    (rows = clips, cols = raters). Rows with missing ratings are dropped.
    """
    M = matrix.dropna().to_numpy(dtype=float)
    n, k = M.shape
    if n < 2 or k < 2:
        return np.nan
    grand = M.mean()
    ss_rows = k * ((M.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((M.mean(axis=0) - grand) ** 2).sum()
    ss_total = ((M - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + (k / n) * (ms_cols - ms_err)
    if denom == 0:
        return np.nan
    return float((ms_rows - ms_err) / denom)


def agreement_report(ratings: pd.DataFrame) -> dict[str, float]:
    """All agreement statistics for a long-format ratings table."""
    from ..data.fluency_labels import ratings_matrix

    wide = ratings_matrix(ratings)
    out = {
        "n_clips": float(len(wide)),
        "n_raters": float(wide.shape[1]),
        "fleiss_kappa": fleiss_kappa(wide),
        "icc_2k": icc_2k(wide),
    }
    if wide.shape[1] == 2:
        out["cohen_kappa_quad"] = cohen_kappa(wide.iloc[:, 0], wide.iloc[:, 1])
    return out
