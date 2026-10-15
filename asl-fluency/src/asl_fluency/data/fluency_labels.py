"""Human fluency ratings: schema, loading, and collection helpers.

The proposal (Section 5) notes no dataset has native fluency labels, so a
small human-rated subset must be collected: instructors or fluent signers
rate each clip 1-5. This module defines the ratings CSV format, stratified
clip sampling for the intake sheet, and the human-only consensus guard.

Ratings CSV (long format), one row per rater x clip:

    clip_id, rater_id, rating
    000123, instructor_a, 4
    000123, instructor_b, 5
    ...

AI-judge rows (from `asl-fluency ai-rate`) use rater ids prefixed with
``vlm_`` (default ``vlm_judge``). They may live in the same file or a
separate one, but they are NEVER part of the human ground truth:
``consensus_scores`` excludes them by default. Use them only for
stratification, secondary analyses, and human-vs-AI agreement checks.

Optional per-clip context columns (merged in from metadata, used for the
stratified fairness evaluation in proposal Section 7):

    skin_tone   e.g. Monk Skin Tone scale 1-10, annotated per clip/signer
    lighting    e.g. low | normal | bright, annotated or auto-estimated
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("clip_id", "rater_id", "rating")
CONTEXT_COLUMNS = ("skin_tone", "lighting")

#: Rater ids starting with this prefix are AI judges, not humans.
AI_RATER_PREFIX = "vlm_"

#: Default rater id used by `asl-fluency ai-rate`.
DEFAULT_AI_RATER_ID = "vlm_judge"


def is_ai_rater(rater_id: str, prefix: str = AI_RATER_PREFIX) -> bool:
    """True for rater ids that denote AI judges (``vlm_*``)."""
    return str(rater_id).startswith(prefix)


def split_ai_raters(
    ratings: pd.DataFrame, prefix: str = AI_RATER_PREFIX
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a long ratings frame into (human_rows, ai_rows)."""
    mask = ratings["rater_id"].map(lambda r: is_ai_rater(r, prefix))
    return ratings[~mask].copy(), ratings[mask].copy()


def load_ratings(path: str | Path, rating_scale: tuple[int, int] = (1, 5)) -> pd.DataFrame:
    """Load a long-format ratings CSV and validate it.

    Rows with a blank rating (an intake sheet that is still being filled in,
    e.g. while the browser rating UI is running) are dropped, so partially
    completed sheets load cleanly.
    """
    df = pd.read_csv(path, dtype={"clip_id": str, "rater_id": str})
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Ratings file {path} missing columns: {sorted(missing)}")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"]).reset_index(drop=True)
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


def consensus_scores(
    ratings: pd.DataFrame,
    method: str = "mean",
    include_ai: bool = False,
    ai_prefix: str = AI_RATER_PREFIX,
) -> pd.Series:
    """Per-clip ground-truth score: mean (default) or median across raters.

    Human-only by default: AI-judge rows (``vlm_*``) are excluded so the
    ground truth stays independent of model-generated scores. Pass
    ``include_ai=True`` only for auxiliary analyses.
    """
    if method not in ("mean", "median"):
        raise ValueError("method must be 'mean' or 'median'")
    if not include_ai:
        ratings, _ = split_ai_raters(ratings, ai_prefix)
    return ratings.groupby("clip_id")["rating"].agg(method)


def rating_sheet_template(clip_ids: list[str], rater_ids: list[str]) -> pd.DataFrame:
    """Empty long-format sheet for rater intake (one row per clip x rater)."""
    rows = [
        {"clip_id": c, "rater_id": r, "rating": ""} for c in clip_ids for r in rater_ids
    ]
    return pd.DataFrame(rows, columns=["clip_id", "rater_id", "rating"])


def stratified_rating_sample(
    features: pd.DataFrame,
    n_clips: int,
    seed: int = 42,
    score_col: str | None = None,
    n_strata: int = 5,
    min_detection_rate: float = 0.0,
    exclude_clip_ids: set[str] | None = None,
    max_per_signer: int | None = None,
) -> pd.DataFrame:
    """Stratified clip sample for human rating, drawn from the feature table.

    Sampling from ``features.csv`` (rather than the raw clip index) guarantees
    every rated clip can actually be evaluated by the model. Clips are
    stratified on ``score_col`` quantiles — the AI-judge score when available,
    otherwise a kinematic proxy such as ``letters_per_second`` — so a small
    sample spans the full fluency range instead of clustering wherever a
    uniform draw lands (ChicagoFSWild skews fluent).

    Parameters
    ----------
    features:
        Feature table with at least ``clip_id`` and ``score_col``; a
        ``signer_id`` column enables signer caps.
    n_clips:
        Target number of clips to draw (fewer if the pool is too small).
    score_col:
        Column to stratify on. Required.
    n_strata:
        Number of quantile strata (5 = quintiles).
    min_detection_rate:
        Tracking-quality gate; clips below this detection rate are excluded.
    exclude_clip_ids:
        Clips already sampled in earlier batches (adaptive collection).
    max_per_signer:
        Optional cap on clips per signer for signer diversity.

    Returns a DataFrame with clip_id, signer_id (if present), stratum, and
    stratum_score (stratum mean of score_col), ordered by stratum.
    """
    if score_col is None:
        raise ValueError("score_col is required for stratified sampling")
    if score_col not in features.columns:
        raise ValueError(f"features table has no '{score_col}' column to stratify on")

    pool = features.copy()
    if "detection_rate" in pool.columns and min_detection_rate > 0:
        pool = pool[pool["detection_rate"] >= min_detection_rate]
    if exclude_clip_ids:
        pool = pool[~pool["clip_id"].isin(exclude_clip_ids)]
    pool = pool.dropna(subset=[score_col])
    if pool.empty:
        return pd.DataFrame(
            columns=["clip_id", "signer_id", "stratum", "stratum_score"]
        )

    n_strata = max(1, min(n_strata, len(pool)))
    # quantile bins; duplicates='drop' guards against degenerate distributions
    stratum = pd.qcut(
        pool[score_col].rank(method="first"), n_strata, labels=False
    )
    pool = pool.assign(stratum=stratum.astype(int))
    stratum_means = pool.groupby("stratum")[score_col].mean()

    rng_seed = seed
    picked: list[pd.DataFrame] = []
    per_stratum = n_clips // n_strata
    remainder = n_clips % n_strata
    signer_counts: dict[str, int] = {}
    for s in sorted(pool["stratum"].unique()):
        take = per_stratum + (1 if remainder > 0 else 0)
        remainder = max(0, remainder - 1)
        candidates = pool[pool["stratum"] == s]
        if max_per_signer is not None and "signer_id" in candidates.columns:
            keep = []
            for _, row in candidates.sample(
                frac=1.0, random_state=rng_seed
            ).iterrows():
                sid = str(row["signer_id"])
                if signer_counts.get(sid, 0) < max_per_signer:
                    keep.append(row.name)
                    signer_counts[sid] = signer_counts.get(sid, 0) + 1
                if len(keep) >= take:
                    break
            chosen = candidates.loc[keep]
        else:
            chosen = candidates.sample(
                min(take, len(candidates)), random_state=rng_seed
            )
        picked.append(chosen)
        rng_seed += 1

    out = pd.concat(picked) if picked else pool.head(0)
    out = out.assign(stratum_score=out["stratum"].map(stratum_means))
    keep_cols = ["clip_id"]
    if "signer_id" in out.columns:
        keep_cols.append("signer_id")
    keep_cols += ["stratum", "stratum_score"]
    return out.sort_values(["stratum", "clip_id"])[keep_cols].reset_index(drop=True)


def save_rating(
    path: str | Path,
    clip_id: str,
    rater_id: str,
    rating: float | None,
    rating_scale: tuple[int, int] = (1, 5),
) -> pd.DataFrame:
    """Upsert one rating cell into a long-format ratings CSV.

    Creates the file if missing; updates the existing (clip_id, rater_id)
    row otherwise. ``rating=None`` clears the cell back to blank. The write
    is atomic (temp file + replace) so a browser UI can save on every click
    without corrupting the sheet.
    """
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path, dtype={"clip_id": str, "rater_id": str})
    else:
        df = pd.DataFrame(columns=list(REQUIRED_COLUMNS))
    if rating is not None:
        lo, hi = rating_scale
        if not lo <= float(rating) <= hi:
            raise ValueError(f"rating {rating} outside the {lo}-{hi} scale")

    df["rating"] = df["rating"].astype(object)
    mask = (df["clip_id"] == str(clip_id)) & (df["rater_id"] == str(rater_id))
    if mask.any():
        df.loc[mask, "rating"] = "" if rating is None else rating
    else:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "clip_id": str(clip_id),
                            "rater_id": str(rater_id),
                            "rating": "" if rating is None else rating,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return df
