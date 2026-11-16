"""Objective per-clip context for the Section 7 fairness evaluation.

ChicagoFSWild ships no skin-tone or lighting labels, and hand-annotating them
is subjective. This module derives both from pixels, located by the already-
extracted hand landmarks, so the stratified evaluation runs without extra
human annotation:

    lighting   mean CIELAB L* of the whole frame (scene brightness), bucketed
               into low / normal / bright by tertiles of the clips scored
               together (relative: in-the-wild video has no absolute scale).
    skin_tone  Individual Typology Angle, ITA = atan((L* - 50) / b*) in
               degrees, from palm pixels (centroid of wrist + 4 finger MCPs).
               Chardon et al. (1991) bands, collapsed to three groups so each
               stratum keeps enough clips at n~50: light (ITA > 41),
               medium (10 < ITA <= 41), dark (ITA <= 10).

Caveat for the paper: ITA is measured on video pixels, so it is confounded
with white balance, exposure and compression. Treat it as a *proxy* stratifier
(it is the same signal a tracker sees), and report it alongside the lighting
factor rather than as a demographic label.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .extractor import LandmarkSequence, open_frame_source

PALM_LANDMARKS = (0, 5, 9, 13, 17)  # wrist + index/middle/ring/pinky MCPs
MIDDLE_MCP = 9
ITA_BANDS = ((41.0, "light"), (10.0, "medium"), (-np.inf, "dark"))


def ita_degrees(lab_pixels: np.ndarray) -> float:
    """ITA from (N, 3) CIELAB pixels (L* in 0-100, b* signed); NaN if undefined.

    Uses the per-channel median, which is robust to the background pixels
    and specular highlights that leak into a palm patch.
    """
    if len(lab_pixels) == 0:
        return np.nan
    L, b = np.median(lab_pixels[:, 0]), np.median(lab_pixels[:, 2])
    if b <= 0:  # blue/neutral cast: ITA is defined for skin (b* > 0) only
        return np.nan
    return float(np.degrees(np.arctan((L - 50.0) / b)))


def ita_band(ita: float) -> str | float:
    if np.isnan(ita):
        return np.nan
    for lo, name in ITA_BANDS:
        if ita > lo:
            return name
    return ITA_BANDS[-1][1]


def _to_lab(frame_bgr: np.ndarray) -> np.ndarray:
    """uint8 BGR -> float CIELAB with L* in [0, 100] and signed a*, b*."""
    return cv2.cvtColor(frame_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def _palm_patch(lab: np.ndarray, lm: np.ndarray) -> np.ndarray:
    """(N, 3) Lab pixels in a disc at the palm centroid of one frame."""
    h, w = lab.shape[:2]
    pts = lm[list(PALM_LANDMARKS), :2] * np.array([w, h])
    cx, cy = pts.mean(axis=0)
    hand_len = np.linalg.norm((lm[MIDDLE_MCP, :2] - lm[0, :2]) * np.array([w, h]))
    r = max(2.0, 0.35 * hand_len)
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    return lab[mask]


def clip_context_row(
    media: str | Path, seq: LandmarkSequence, max_frames: int = 16
) -> dict:
    """Raw context measurements for one clip (no bucketing)."""
    cap, _ = open_frame_source(media)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    if hasattr(cap, "release"):
        cap.release()

    n = min(len(frames), seq.n_frames)
    ok_idx = [i for i in range(n) if seq.frame_ok[i]]
    all_idx = np.linspace(0, n - 1, min(n, max_frames)).round().astype(int) if n else []
    hand_idx = (
        [ok_idx[i] for i in np.linspace(0, len(ok_idx) - 1, min(len(ok_idx), max_frames))
         .round().astype(int)] if ok_idx else []
    )

    luma = [float(_to_lab(frames[i])[..., 0].mean()) for i in all_idx]
    patches = [_palm_patch(_to_lab(frames[i]), seq.landmarks[i]) for i in hand_idx]
    palm = np.concatenate(patches) if patches else np.empty((0, 3))
    return {
        "clip_id": seq.clip_id,
        "luma_L": float(np.mean(luma)) if luma else np.nan,
        "ita_deg": ita_degrees(palm),
        "n_palm_pixels": len(palm),
        "n_hand_frames": len(hand_idx),
    }


def bucket_context(raw: pd.DataFrame) -> pd.DataFrame:
    """Add categorical `skin_tone` and `lighting` columns to raw measurements."""
    out = raw.copy()
    out["skin_tone"] = out["ita_deg"].map(ita_band)
    valid = out["luma_L"].notna()
    out["lighting"] = pd.Series(np.nan, index=out.index, dtype=object)
    if valid.sum() >= 3:
        out.loc[valid, "lighting"] = pd.qcut(
            out.loc[valid, "luma_L"], 3, labels=["low", "normal", "bright"]
        ).astype(str)
    out["context_source"] = "auto_ita_luma"
    return out


def build_clip_context(
    clip_media: dict[str, Path], landmark_dir: str | Path, max_frames: int = 16
) -> pd.DataFrame:
    """Context table for every clip that has both media and landmarks."""
    rows = []
    for clip_id, media in clip_media.items():
        npz = Path(landmark_dir) / f"{clip_id}.npz"
        if not npz.exists() or not Path(media).exists():
            continue
        try:
            rows.append(clip_context_row(media, LandmarkSequence.load(npz), max_frames))
        except OSError:
            continue  # unreadable media for one clip must not abort the batch
    cols = ["clip_id", "luma_L", "ita_deg", "n_palm_pixels", "n_hand_frames"]
    return bucket_context(pd.DataFrame(rows, columns=cols))
