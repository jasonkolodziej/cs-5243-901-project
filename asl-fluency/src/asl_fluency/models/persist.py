"""Fit and persist the final fluency scorer.

`asl-fluency evaluate` fits models only inside cross-validation folds (to
produce honest out-of-fold metrics) and discards them. This module is the
persistence step: once the ablation has picked a winning arm, `train` fits
that arm on ALL rated data and saves it with joblib; `score` loads the
artifact and applies it to new clips.

The artifact is a plain dict so it stays inspectable:

    {
      "model":       fitted model (module-level classes only — picklable),
      "kind":        e.g. "ridge",
      "feature_set": e.g. "kinematic",
      "feature_cols": the exact columns the model consumes,
      "n_train_clips": how many rated clips it was fit on,
      "rating_scale": (1, 5),
    }

Small-sample caveat (proposal Section 7): with ~50-100 rated clips this is a
research scorer — present it as proof-of-concept validated by the CV metrics,
not a production instrument.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from ..features.pipeline import FEATURE_SETS
from .baseline import make_model

ARTIFACT_VERSION = 1


def fit_scorer(
    feats: pd.DataFrame,
    y: pd.Series,
    kind: str,
    feature_set: str,
    seed: int = 42,
):
    """Fit one model on all rated data. `y` is indexed by clip_id."""
    labeled = feats[feats["clip_id"].isin(y.index)].reset_index(drop=True)
    if labeled.empty:
        raise ValueError(
            "no overlap between features and rated clips — collect ratings first"
        )
    target = y.loc[labeled["clip_id"]].to_numpy(dtype=float)
    return make_model(kind, feature_set, seed=seed).fit(labeled, target), labeled


def save_scorer(
    model,
    path: str | Path,
    *,
    kind: str,
    feature_set: str,
    n_train_clips: int,
    rating_scale: tuple[int, int] = (1, 5),
) -> Path:
    """Persist a fitted scorer artifact (atomic write)."""
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "version": ARTIFACT_VERSION,
        "model": model,
        "kind": kind,
        "feature_set": feature_set,
        "feature_cols": list(FEATURE_SETS[feature_set]),
        "n_train_clips": int(n_train_clips),
        "rating_scale": tuple(rating_scale),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(artifact, tmp)
    os.replace(tmp, path)
    return path


def load_scorer(path: str | Path) -> dict:
    """Load a scorer artifact saved by save_scorer."""
    artifact = joblib.load(path)
    for key in ("model", "kind", "feature_set"):
        if key not in artifact:
            raise ValueError(f"{path} is not a fluency-scorer artifact (missing {key})")
    return artifact


def predict_scores(artifact: dict, feats: pd.DataFrame) -> pd.Series:
    """Score a feature table; returns a Series indexed by clip_id."""
    preds = artifact["model"].predict(feats)
    lo, hi = artifact.get("rating_scale", (1, 5))
    preds = preds.clip(lo, hi)
    return pd.Series(preds, index=feats["clip_id"].astype(str), name="predicted_fluency")
