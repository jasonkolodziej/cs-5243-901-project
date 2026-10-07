"""Per-frame hand-landmark extraction with MediaPipe HandLandmarker (Tasks API).

Stage 1 of the proposal pipeline. For each clip we store:

    landmarks : float32 array (n_frames, 21, 3)  -- x, y, z per landmark, NaN if no hand
    frame_ok  : bool array (n_frames,)           -- whether a hand was detected
    fps       : float                            -- native (or resampled) frame rate

Saved as one .npz per clip under the interim dir. MediaPipe coordinates are
image-normalised (x, y in [0, 1], z relative to wrist depth), which keeps
features resolution-independent.

Requires the hand landmarker model (one-time download, ~8 MB):

    curl -L -o assets/hand_landmarker.task \\
      https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
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

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "assets" / "hand_landmarker.task"


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


def _make_landmarker(model_path: str | Path, max_num_hands: int,
                     min_detection_confidence: float, min_tracking_confidence: float):
    """Build a VIDEO-mode HandLandmarker (mediapipe >= 0.10.14 Tasks API)."""
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Hand landmarker model not found at {model_path}. Download it with:\n"
            "  curl -L -o assets/hand_landmarker.task https://storage.googleapis.com/"
            "mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_hands=max_num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return HandLandmarker.create_from_options(options)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def open_frame_source(
    source: str | Path, default_fps: float = 30.0
) -> tuple[cv2.VideoCapture | FrameDirectoryCapture, float]:
    """Open a clip source: a video file OR a directory of frame images.

    ChicagoFSWild distributes clips as frame directories (e.g.
    `aslized/elsie_stecker_0001/0001.jpg`), not video files, so both layouts
    are first-class. Returns (capture-like object, fps). Frame directories
    have no native frame rate, so `default_fps` is used.
    """
    source = Path(source)
    if source.is_dir():
        cap = FrameDirectoryCapture(source)
        if not cap.isOpened():
            raise OSError(f"No frame images ({sorted(IMAGE_SUFFIXES)}) in {source}")
        return cap, float(default_fps)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {source}")
    return cap, float(cap.get(cv2.CAP_PROP_FPS) or default_fps)


class FrameDirectoryCapture:
    """Minimal cv2.VideoCapture-compatible reader over a frame directory."""

    def __init__(self, directory: Path) -> None:
        self.files = sorted(
            p for p in Path(directory).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
        self._i = 0

    def isOpened(self) -> bool:
        return bool(self.files)

    def read(self):
        if self._i >= len(self.files):
            return False, None
        img = cv2.imread(str(self.files[self._i]))
        self._i += 1
        if img is None:
            return False, None
        return True, img

    def get(self, prop: int) -> float:  # only FPS is meaningful
        return 0.0

    def release(self) -> None:
        pass


def extract_landmarks(
    video_path: str | Path,
    clip_id: str,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    max_num_hands: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    hand: str = "auto",
    target_fps: float | None = None,
    default_fps: float = 30.0,
) -> LandmarkSequence:
    """Run HandLandmarker over a video file or a frame directory.

    `hand="auto"` tracks the most active hand over time (fingertip-motion EMA),
    which is the fingerspelling hand; see SigningHandSelector.
    """
    import mediapipe as mp

    cap, native_fps = open_frame_source(video_path, default_fps)
    step = max(1, round(native_fps / target_fps)) if target_fps else 1
    fps = native_fps / step

    frames, oks = [], []
    prev_wrist: np.ndarray | None = None
    selector = SigningHandSelector() if hand == "auto" else None

    landmarker = _make_landmarker(
        model_path, max_num_hands, min_detection_confidence, min_tracking_confidence
    )
    with landmarker:
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % step:
                idx += 1
                continue
            timestamp_ms = int(idx * 1000 / native_fps)
            idx += 1
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(image, timestamp_ms)
            lm, prev_wrist = _select_hand(res, prev_wrist, hand, selector)
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


class SigningHandSelector:
    """Pick the fingerspelling hand as the more *active* hand over time.

    MediaPipe does not give stable hand IDs, so candidates are matched across
    frames by wrist proximity (greedy, thresholded). Each track accumulates an
    EMA of fingertip speed; the fingerspelling hand is the higher-activity
    track. Before any motion evidence accumulates, falls back to temporal
    continuity with the previously selected hand.
    """

    def __init__(self, alpha: float = 0.2, match_thresh: float = 0.3,
                 stale_after: int = 30) -> None:
        self.alpha = alpha
        self.match_thresh = match_thresh
        self.stale_after = stale_after
        self.tracks: list[dict] = []      # {wrist, tips, activity, age}
        self.prev_selected_wrist: np.ndarray | None = None

    def select(self, res) -> np.ndarray | None:
        if not res.hand_landmarks:
            self._age_tracks()
            return None
        cands = [np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
                 for lm in res.hand_landmarks]
        assigned = self._match(cands)
        best, best_activity = None, -1.0
        for arr, track in assigned:
            if track["activity"] > best_activity:
                best, best_activity = arr, track["activity"]
        if best is None or (best_activity == 0 and self.prev_selected_wrist is not None):
            # no motion evidence yet: continuity with previous selection
            best = min(assigned, key=lambda at: np.linalg.norm(
                at[0][WRIST, :2] - self.prev_selected_wrist))[0]
        self.prev_selected_wrist = best[WRIST, :2]
        self._age_tracks()
        return best

    def _match(self, cands: list[np.ndarray]) -> list[tuple[np.ndarray, dict]]:
        assigned, used = [], set()
        for arr in cands:
            wrist = arr[WRIST, :2]
            track, best_d = None, self.match_thresh
            for i, t in enumerate(self.tracks):
                if i in used:
                    continue
                d = float(np.linalg.norm(wrist - t["wrist"]))
                if d < best_d:
                    track, best_d, best_i = t, d, i
            if track is None:
                track = {"wrist": wrist, "tips": arr[FINGER_TIPS, :2],
                         "activity": 0.0, "age": 0}
                self.tracks.append(track)
                used.add(len(self.tracks) - 1)
            else:
                used.add(best_i)
                tip_speed = float(np.linalg.norm(
                    arr[FINGER_TIPS, :2] - track["tips"], axis=1).mean())
                track["activity"] = (self.alpha * tip_speed
                                     + (1 - self.alpha) * track["activity"])
                track["wrist"], track["tips"], track["age"] = wrist, arr[FINGER_TIPS, :2], 0
            assigned.append((arr, track))
        return assigned

    def _age_tracks(self) -> None:
        for t in self.tracks:
            t["age"] += 1
        self.tracks = [t for t in self.tracks if t["age"] <= self.stale_after]


def _select_hand(res, prev_wrist: np.ndarray | None, hand: str,
                 selector: SigningHandSelector | None = None):
    """Pick the signing hand's landmarks from a HandLandmarker result."""
    if not res.hand_landmarks:
        return None, prev_wrist
    cands = []
    for lm_list, handed in zip(res.hand_landmarks, res.handedness):
        arr = np.array([[p.x, p.y, p.z] for p in lm_list], dtype=np.float32)
        cands.append((arr, handed[0].category_name.lower()))
    if hand in ("left", "right"):
        # Note: MediaPipe handedness is from the *camera's* perspective (selfie-flipped).
        wanted = "right" if hand == "left" else "left"
        for arr, label in cands:
            if label == wanted:
                return arr, arr[WRIST, :2]
        return None, prev_wrist
    # auto: most-active hand over time (fingerspelling hand), see selector
    if selector is not None:
        arr = selector.select(res)
        return (arr, arr[WRIST, :2]) if arr is not None else (None, prev_wrist)
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


# MediaPipe hand skeleton edges for visualization (pairs of landmark indices).
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_landmarks(bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Overlay the 21-point hand skeleton on a BGR frame (in place copy)."""
    out = bgr.copy()
    if landmarks is None or np.all(np.isnan(landmarks)):
        return out
    h, w = out.shape[:2]
    pts = [(int(p[0] * w), int(p[1] * h)) for p in landmarks]
    for a, b in HAND_EDGES:
        cv2.line(out, pts[a], pts[b], (0, 200, 0), 2)
    for x, y in pts:
        cv2.circle(out, (x, y), 3, (0, 0, 255), -1)
    return out
