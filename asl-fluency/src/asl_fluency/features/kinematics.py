"""Kinematic feature engineering.

Stage 3 of the proposal pipeline. From a landmark sequence plus letter
spans, compute the fluency features named in the proposal:

    - letters per second (overall signing rate)
    - per-letter duration statistics (mean, std, coefficient of variation)
    - transition velocity / jerk statistics between consecutive letters
    - pause count, total pause time, longest pause
    - path efficiency (wrist path length vs. straight-line displacement)

All features are floats; a per-clip row is produced by
features.pipeline.compute_feature_row.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from scipy.signal import savgol_filter

from ..landmarks.extractor import FINGER_TIPS, LandmarkSequence


def smooth_positions(xy: np.ndarray, window: int, polyorder: int, fps: float) -> np.ndarray:
    """Savitzky-Golay smoothing of a (T, D) position trace (NaNs interpolated)."""
    out = xy.copy().astype(np.float64)
    for d in range(out.shape[1]):
        col = out[:, d]
        good = ~np.isnan(col)
        if good.sum() < polyorder + 2:
            continue
        col[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), col[good])
        w = min(window, good.sum() // 2 * 2 - 1)  # odd, <= series length
        if w > polyorder:
            out[:, d] = savgol_filter(col, w, polyorder)
    return out


def derivatives(xy: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Velocity, acceleration, jerk magnitudes from smoothed positions."""
    v = np.gradient(xy, 1.0 / fps, axis=0)
    a = np.gradient(v, 1.0 / fps, axis=0)
    j = np.gradient(a, 1.0 / fps, axis=0)
    return (
        np.linalg.norm(v, axis=1),
        np.linalg.norm(a, axis=1),
        np.linalg.norm(j, axis=1),
    )


def _segment_stats(values: np.ndarray, spans: list[tuple[int, int, str]]) -> dict[str, float]:
    """Stats of a per-frame signal restricted to inter-letter transition zones."""
    trans_vals = []
    for (_, e1, _), (s2, _, _) in pairwise(spans):
        zone = values[e1:s2] if s2 > e1 else values[max(e1 - 1, 0): e1 + 1]
        trans_vals.extend(zone[~np.isnan(zone)])
    trans_vals = np.asarray(trans_vals)
    if len(trans_vals) == 0:
        return {"mean": np.nan, "std": np.nan, "max": np.nan}
    return {
        "mean": float(np.mean(trans_vals)),
        "std": float(np.std(trans_vals)),
        "max": float(np.max(trans_vals)),
    }


def kinematic_features(
    seq: LandmarkSequence,
    spans: list[tuple[int, int, str]],
    *,
    smooth_window: int = 5,
    smooth_polyorder: int = 2,
    pause_speed_percentile: float = 15.0,
    min_pause_frames: int = 3,
) -> dict[str, float]:
    """The full kinematic feature vector for one clip."""
    feats: dict[str, float] = {}
    fps = seq.fps
    n_letters = len(spans)

    duration_s = (spans[-1][1] - spans[0][0]) / fps
    feats["letters_per_second"] = n_letters / duration_s if duration_s > 0 else np.nan

    letter_durs = np.array([(e - s) / fps for s, e, _ in spans])
    feats["letter_dur_mean_s"] = float(letter_durs.mean())
    feats["letter_dur_std_s"] = float(letter_durs.std())
    feats["letter_dur_cv"] = float(letter_durs.std() / letter_durs.mean()) if letter_durs.mean() > 0 else np.nan

    # Wrist kinematics (primary motion channel for transitions).
    wrist = smooth_positions(seq.wrist_xy(), smooth_window, smooth_polyorder, fps)
    v, a, j = derivatives(wrist, fps)
    for name, sig in (("vel", v), ("acc", a), ("jerk", j)):
        stats = _segment_stats(sig, spans)
        feats[f"trans_{name}_mean"] = stats["mean"]
        feats[f"trans_{name}_std"] = stats["std"]
    feats["trans_jerk_max"] = _segment_stats(j, spans)["max"]
    # Log dimensionless jerk (smoothness index; lower = smoother).
    if duration_s > 0 and np.nanmax(v) > 0:
        d_jerk = duration_s**5 / (np.nanmax(v) ** 2) * np.nanmean(j**2)
        feats["log_dimless_jerk"] = float(-np.log(abs(d_jerk) + 1e-12))
    else:
        feats["log_dimless_jerk"] = np.nan

    # Fingertip kinematics (shape-change channel between letters).
    tips = seq.landmarks[:, FINGER_TIPS, :2].reshape(len(seq.frame_ok), -1)
    tips = smooth_positions(tips, smooth_window, smooth_polyorder, fps)
    tv, _, tj = derivatives(tips, fps)
    feats["tip_vel_mean"] = float(np.nanmean(tv))
    feats["tip_jerk_std"] = float(np.nanstd(tj))

    # Pauses: runs of frames where wrist speed is below the clip's low percentile.
    thresh = np.nanpercentile(v[seq.frame_ok], pause_speed_percentile) if seq.frame_ok.any() else np.nan
    pauses = _pause_runs(v, thresh, min_pause_frames)
    feats["pause_count"] = float(len(pauses))
    feats["pause_total_s"] = float(sum(p[1] - p[0] for p in pauses) / fps)
    feats["pause_max_s"] = float(max((p[1] - p[0] for p in pauses), default=0) / fps)

    # Path efficiency: straight-line / actual wrist path length.
    w_ok = wrist[seq.frame_ok]
    if len(w_ok) > 1:
        path = np.linalg.norm(np.diff(w_ok, axis=0), axis=1).sum()
        disp = np.linalg.norm(w_ok[-1] - w_ok[0])
        feats["path_efficiency"] = float(disp / path) if path > 0 else np.nan
    else:
        feats["path_efficiency"] = np.nan

    # Tracking-quality covariates (kept in the row for stratified analysis).
    feats["detection_rate"] = seq.detection_rate
    return feats


def _pause_runs(v: np.ndarray, thresh: float, min_len: int) -> list[tuple[int, int]]:
    """Contiguous [start, end) frame runs with speed below threshold."""
    if np.isnan(thresh):
        return []
    slow = np.where(np.isnan(v), False, v < thresh)
    runs, start = [], None
    for i, s in enumerate(slow):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    if start is not None and len(slow) - start >= min_len:
        runs.append((start, len(slow)))
    return runs
