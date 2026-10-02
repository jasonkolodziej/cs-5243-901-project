"""Human fluency ratings: schema, loading, and collection helpers.

The proposal (Section 5) notes no dataset has native fluency labels, so a
small human-rated subset must be collected: instructors or fluent signers
rate each clip 1-5. This module defines the ratings CSV format and the
wide/long conversions used by evaluation.agreement.

Ratings CSV (long format), one row per rater x clip:

    clip_id, rater_id, rating
    000123, instructor_a, 4
    000123, instructor_b, 5
    ...

Optional per-clip context columns (merged in from metadata, used for the
stratified fairness evaluation in proposal Section 7):

    skin_tone   e.g. Monk Skin Tone scale 1-10, annotated per clip/signer
    lighting    e.g. low | normal | bright, annotated or auto-estimated
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("clip_id", "rater_id", "rating")
CONTEXT_COLUMNS = ("skin_tone", "lighting")


def load_ratings(path: str | Path, rating_scale: tuple[int, int] = (1, 5)) -> pd.DataFrame:
    """Load a long-format ratings CSV and validate it."""
    df = pd.read_csv(path, dtype={"clip_id": str, "rater_id": str})
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Ratings file {path} missing columns: {sorted(missing)}")
    lo, hi = rating_scale
    bad = ~df["rating"].between(lo, hi)
    if bad.any():
        raise ValueError(f"{bad.sum()} ratings outside the {lo}-{hi} scale in {path}")
    dup = df.duplicated(subset=["clip_id", "rater_id"])
    if dup.any():
        raise ValueError(f"{dup.sum()} duplicate (clip_id, rater_id) rows in {path}")
    return df


def ratings_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """Pivot long ratings to wide: rows=clip_id, cols=rater_id (NaN = missing)."""
    return ratings.pivot(index="clip_id", columns="rater_id", values="rating")


def consensus_scores(ratings: pd.DataFrame, method: str = "mean") -> pd.Series:
    """Per-clip ground-truth score: mean (default) or median across raters."""
    if method not in ("mean", "median"):
        raise ValueError("method must be 'mean' or 'median'")
    return ratings.groupby("clip_id")["rating"].agg(method)


def rating_sheet_template(clip_ids: list[str], rater_ids: list[str]) -> pd.DataFrame:
    """Empty long-format sheet for rater intake (one row per clip x rater)."""
    rows = [
        {"clip_id": c, "rater_id": r, "rating": ""} for c in clip_ids for r in rater_ids
    ]
    return pd.DataFrame(rows, columns=["clip_id", "rater_id", "rating"])
