"""Hand-labeled letter boundaries (Roadmap Gap 1, step 1).

Ground truth for validating the segmentation methods and for measuring how
much boundary error propagates into features. Collected with the browser
UI's boundary-labeling mode (`asl-fluency label-boundaries`) on the same
clips that receive fluency ratings — one session serves both.

CSV schema (one row per letter):

    clip_id, letter, start_frame, end_frame
    elsie_stecker_0001, D, 3, 12
    elsie_stecker_0001, C, 12, 19

Frames are integer indices into the clip's frame sequence; spans are
[start, end) and tile the clip without gaps.
"""

from __future__ import annotations

import os
from itertools import pairwise
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("clip_id", "letter", "start_frame", "end_frame")


def load_boundary_labels(path: str | Path) -> pd.DataFrame:
    """Load boundary labels; returns an empty frame if the file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS))
    df = pd.read_csv(path, dtype={"clip_id": str, "letter": str})
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Boundary labels {path} missing columns: {sorted(missing)}")
    return df


def spans_for(labels: pd.DataFrame, clip_id: str) -> list[tuple[int, int, str]]:
    """Ordered [(start, end, letter), ...] spans for one clip ([] if none)."""
    rows = labels[labels["clip_id"] == str(clip_id)]
    return [
        (int(r.start_frame), int(r.end_frame), str(r.letter))
        for r in rows.itertuples()
    ]


def save_clip_spans(
    path: str | Path, clip_id: str, spans: list[tuple[int, int, str]]
) -> pd.DataFrame:
    """Replace all rows for one clip with the given spans (atomic write).

    Validity rules: spans must be ordered, non-overlapping, non-empty, and
    share the clip's first start / last end is whatever the labeler set.
    """
    for (s1, e1, _), (s2, e2, _) in pairwise(spans):
        if not (s1 < e1 <= s2):
            raise ValueError(
                f"spans for {clip_id} are not ordered/non-overlapping: {spans}"
            )
    df = load_boundary_labels(path)
    df = df[df["clip_id"] != str(clip_id)]
    new = pd.DataFrame(
        [
            {
                "clip_id": str(clip_id),
                "letter": letter,
                "start_frame": int(s),
                "end_frame": int(e),
            }
            for s, e, letter in spans
        ]
    )
    df = pd.concat([df, new], ignore_index=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return df
