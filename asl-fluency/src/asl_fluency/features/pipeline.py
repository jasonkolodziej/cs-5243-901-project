"""Assemble per-clip feature rows into the experiment feature table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..landmarks.extractor import LandmarkSequence
from ..segmentation.aligner import segment_letters
from .kinematics import kinematic_features

NAIVE_FEATURES = ["letters_per_second"]
KINEMATIC_FEATURES = [
    "letter_dur_mean_s", "letter_dur_std_s", "letter_dur_cv",
    "trans_vel_mean", "trans_vel_std",
    "trans_acc_mean", "trans_acc_std",
    "trans_jerk_mean", "trans_jerk_std", "trans_jerk_max",
    "log_dimless_jerk",
    "tip_vel_mean", "tip_jerk_std",
    "pause_count", "pause_total_s", "pause_max_s",
    "path_efficiency",
]
QUALITY_COVARIATES = ["detection_rate"]  # kept out of the model, used for slicing

FEATURE_SETS = {
    "naive": NAIVE_FEATURES,                                  # baseline arm
    "kinematic": NAIVE_FEATURES + KINEMATIC_FEATURES,         # proposal's main arm
    "full": NAIVE_FEATURES + KINEMATIC_FEATURES,              # + fused embeddings later
}


def compute_feature_row(
    seq: LandmarkSequence,
    transcript: str,
    *,
    seg_method: str = "uniform",
    refine_radius: int = 6,
    **kw,
) -> dict:
    """One clip -> one feature dict (clip_id + feature columns)."""
    spans = segment_letters(seq, transcript, method=seg_method, refine_radius=refine_radius)
    row = {"clip_id": seq.clip_id, "n_letters": len(spans)}
    row.update(kinematic_features(seq, spans, **kw))
    return row


def build_feature_table(
    landmark_dir: str | Path,
    transcripts: dict[str, str],
    **kw,
) -> pd.DataFrame:
    """Build the full feature table from a directory of .npz landmark files.

    Rows whose kinematic features came out NaN (tracking too sparse to
    smooth, even though the clip passed the segmentation gate) are dropped
    and reported — sklearn estimators cannot fit NaNs, and such rows carry
    no real signal.
    """
    import logging

    log = logging.getLogger(__name__)
    rows = []
    for path in sorted(Path(landmark_dir).glob("*.npz")):
        seq = LandmarkSequence.load(path)
        if seq.clip_id not in transcripts:
            continue
        try:
            rows.append(compute_feature_row(seq, transcripts[seq.clip_id], **kw))
        except ValueError as exc:
            log.warning("skip %s: %s", seq.clip_id, exc)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    feature_cols = NAIVE_FEATURES + KINEMATIC_FEATURES
    bad = table[feature_cols].isna().any(axis=1)
    if bad.any():
        log.warning(
            "drop %d clips with NaN features (tracking too sparse): %s",
            int(bad.sum()),
            ", ".join(table.loc[bad, "clip_id"].head(10)),
        )
        table = table[~bad].reset_index(drop=True)
    return table
