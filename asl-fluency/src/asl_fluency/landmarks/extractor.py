"""Per-frame hand-landmark extraction with MediaPipe Hands.

Stage 1 of the proposal pipeline. For each clip we store:

    landmarks : float32 array (n_frames, 21, 3)  -- x, y, z per landmark, NaN if no hand
    frame_ok  : bool array (n_frames,)           -- whether a hand was detected
    fps       : float                            -- native (or resampled) frame rate

Saved as one .npz per clip under the interim dir. MediaPipe coordinates are
image-normalised (x, y in [0, 1], z relative to wrist depth), which keeps
features resolution-independent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

N_LANDMARKS = 21
WRIST = 0
FINGER_TIPS = (4, 8, 12, 16, 20)  # thumb, index, middle, ring, pinky


@dataclass
class LandmarkSequence:
    clip_id: str
    landmarks: np.ndarray  # (T, 21, 3), NaN where no hand detected
    frame_ok: np.ndarray   # (T,) bool
    fps: float

    @property
    def n_frames(self) -> int:
        return len(self.frame_ok)

    @property
    def detection_rate(self) -> float:
        """Fraction of frames with a detected hand (tracking-quality indicator)."""
        return float(self.frame_ok.mean()) if self.n_frames else 0.0

    def wrist_xy(self) -> np.ndarray:
        """(T, 2) wrist positions, NaN on missed frames."""
        return self.landmarks[:, WRIST, :2]

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            clip_id=self.clip_id,
            landmarks=self.landmarks.astype(np.float32),
            frame_ok=self.frame_ok,
            fps=np.float64(self.fps),
        )

    @classmethod
    def load(cls, path: str | Path) -> LandmarkSequence:
        z = np.load(path, allow_pickle=False)
        return cls(
            clip_id=str(z["clip_id"]),
            landmarks=z["landmarks"],
            frame_ok=z["frame_ok"].astype(bool),
            fps=float(z["fps"]),
        )


def extract_landmarks(
    video_path: str | Path,
    clip_id: str,
    *,
    max_num_hands: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    hand: str = "auto",
    target_fps: float | None = None,
) -> LandmarkSequence:
    """Run MediaPipe Hands over a video file.

    `hand="auto"` picks, per frame, the detected hand most likely to be the
    fingerspelling hand: prefer MediaPipe's dominant-hand label consistency,
    else the hand nearest the previous frame's position (motion continuity).
    """
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native_fps / target_fps)) if target_fps else 1
    fps = native_fps / step

    frames, oks = [], []
    prev_wrist: np.ndarray | None = None

    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as hands:
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % step:
                idx += 1
                continue
            idx += 1
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            lm, prev_wrist = _select_hand(res, prev_wrist, hand)
            if lm is None:
                frames.append(np.full((N_LANDMARKS, 3), np.nan, dtype=np.float32))
                oks.append(False)
            else:
                frames.append(lm)
                oks.append(True)
    cap.release()

    return LandmarkSequence(
        clip_id=clip_id,
        landmarks=np.stack(frames) if frames else np.zeros((0, N_LANDMARKS, 3), np.float32),
        frame_ok=np.asarray(oks, dtype=bool),
        fps=fps,
    )


def _select_hand(res, prev_wrist: np.ndarray | None, hand: str):
    """Pick the signing hand's landmarks from a MediaPipe result."""
    if not res.multi_hand_landmarks:
        return None, prev_wrist
    cands = []
    for lm_list, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
        arr = np.array([[p.x, p.y, p.z] for p in lm_list.landmark], dtype=np.float32)
        cands.append((arr, handed.classification[0].label.lower()))
    if hand in ("left", "right"):
        # Note: MediaPipe labels are from the *camera's* perspective (selfie-flipped).
        wanted = "right" if hand == "left" else "left"
        for arr, label in cands:
            if label == wanted:
                return arr, arr[WRIST, :2]
        return None, prev_wrist
    # auto: motion continuity with previous frame
    if prev_wrist is not None and len(cands) > 1:
        arr = min(cands, key=lambda c: np.nan if np.any(np.isnan(c[0][WRIST]))
                  else np.linalg.norm(c[0][WRIST, :2] - prev_wrist))[0]
    else:
        arr = cands[0][0]
    return arr, arr[WRIST, :2]


def batch_extract(
    clip_index, out_dir: str | Path, *, overwrite: bool = False, **kwargs
) -> list[Path]:
    """Extract landmarks for every clip in a ClipIndex; returns saved paths."""
    from tqdm import tqdm

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for row in tqdm(list(clip_index.frame.itertuples()), desc="landmarks"):
        out = out_dir / f"{row.clip_id}.npz"
        if out.exists() and not overwrite:
            saved.append(out)
            continue
        try:
            seq = extract_landmarks(row.video_path, row.clip_id, **kwargs)
            seq.save(out)
        except Exception as exc:  # keep batch going; failures are logged
            log.warning("clip %s failed: %s", row.clip_id, exc)
            continue
        saved.append(out)
    return saved
