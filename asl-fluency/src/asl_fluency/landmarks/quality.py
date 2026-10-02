"""Tracking-quality diagnostics.

Proposal Section 7: tracking reliability degrades with occlusion, motion
blur, camera angle, and lighting — and any accuracy gap by skin tone /
lighting would bias fluency scores independent of signing skill. These
metrics quantify per-clip tracking quality so downstream evaluation can
stratify by condition instead of reporting only aggregate numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .extractor import LandmarkSequence


def quality_report(seqs: list[LandmarkSequence]) -> pd.DataFrame:
    """One row per clip with tracking-quality metrics."""
    rows = []
    for s in seqs:
        rows.append(
            {
                "clip_id": s.clip_id,
                "n_frames": s.n_frames,
                "fps": s.fps,
                "detection_rate": s.detection_rate,
                "max_dropout_run": _max_false_run(s.frame_ok),
                "mean_interframe_jump": _mean_interframe_jump(s),
            }
        )
    return pd.DataFrame(rows)


def _max_false_run(mask: np.ndarray) -> int:
    """Longest consecutive run of missed detections (dropped frames)."""
    best = run = 0
    for ok in mask:
        run = 0 if ok else run + 1
        best = max(best, run)
    return best


def _mean_interframe_jump(seq: LandmarkSequence) -> float:
    """Mean |Δwrist| between consecutive detected frames.

    Unusually large jumps flag tracker re-acquisition (jitter source) rather
    than real hand motion.
    """
    w = seq.wrist_xy()
    ok = seq.frame_ok
    jumps = []
    for i in range(1, len(w)):
        if ok[i] and ok[i - 1]:
            jumps.append(float(np.linalg.norm(w[i] - w[i - 1])))
    return float(np.mean(jumps)) if jumps else np.nan


def stratify(
    quality: pd.DataFrame, context: pd.DataFrame, by: list[str]
) -> dict[tuple, pd.DataFrame]:
    """Group the quality table by context columns (skin_tone, lighting, ...).

    `context` is a DataFrame with clip_id plus the columns named in `by`
    (from the ratings/metadata annotation). Returns {(slice values): df}.
    """
    merged = quality.merge(context[["clip_id", *by]], on="clip_id", how="inner")
    return {k: g for k, g in merged.groupby(by, dropna=False)}
