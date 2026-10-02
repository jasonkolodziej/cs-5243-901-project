"""Letter-boundary segmentation within a fingerspelling clip.

Stage 2 of the proposal pipeline. Given a landmark sequence and its known
transcript (ChicagoFSWild provides the spelled word), produce per-letter
[start, end) frame spans.

Methods:
    uniform        -- split the detected span evenly across letters.
                      Simple, surprisingly strong baseline for fingerspelling,
                      where letters are roughly isochronous.
    energy_valley  -- start from uniform boundaries, then snap each boundary
                      to the nearest local minimum of wrist speed (letters
                      tend to be separated by brief motion minima).
    forced_alignment -- placeholder hook for a CTC-based recognizer that
                      emits per-letter alignments (not bundled; plug in e.g.
                      a ChicagoFSWild-pretrained model).
"""

from __future__ import annotations

import numpy as np

from ..landmarks.extractor import LandmarkSequence


def segment_letters(
    seq: LandmarkSequence,
    transcript: str,
    method: str = "uniform",
    refine_radius: int = 6,
) -> list[tuple[int, int, str]]:
    """Return [(start_frame, end_frame, letter), ...] covering the clip.

    Only frames with a detected hand are used; boundaries are in original
    frame indices.
    """
    letters = [c for c in transcript.upper() if c.isalpha()]
    if not letters:
        raise ValueError("Transcript has no letters to align")
    detected = np.flatnonzero(seq.frame_ok)
    if len(detected) < len(letters):
        raise ValueError(
            f"clip {seq.clip_id}: only {len(detected)} detected frames for "
            f"{len(letters)} letters -- tracking too poor to segment"
        )
    first, last = int(detected[0]), int(detected[-1]) + 1

    boundaries = np.linspace(first, last, len(letters) + 1).round().astype(int)
    if method == "energy_valley":
        boundaries = _snap_to_speed_valleys(seq, boundaries, refine_radius)
    elif method == "forced_alignment":
        raise NotImplementedError(
            "Plug a CTC recognizer's alignment output in here; see README."
        )
    elif method != "uniform":
        raise ValueError(f"unknown segmentation method: {method}")

    return [
        (int(boundaries[i]), int(boundaries[i + 1]), letters[i])
        for i in range(len(letters))
    ]


def _snap_to_speed_valleys(
    seq: LandmarkSequence, boundaries: np.ndarray, radius: int
) -> np.ndarray:
    """Move interior boundaries to the nearest wrist-speed local minimum."""
    speed = _wrist_speed(seq)
    out = boundaries.copy()
    for i in range(1, len(boundaries) - 1):
        b = boundaries[i]
        lo, hi = max(b - radius, boundaries[i - 1] + 1), min(
            b + radius + 1, boundaries[i + 1]
        )
        window = speed[lo:hi]
        if len(window) and not np.all(np.isnan(window)):
            out[i] = lo + int(np.nanargmin(window))
    return out


def _wrist_speed(seq: LandmarkSequence) -> np.ndarray:
    """Per-frame wrist speed (units: normalized-image-distance per second)."""
    w = seq.wrist_xy()
    v = np.full(len(w), np.nan)
    diffs = np.linalg.norm(np.diff(w, axis=0), axis=1) * seq.fps
    v[1:] = diffs
    return v
