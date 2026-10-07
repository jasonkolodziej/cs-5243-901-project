"""Live / streaming fluency monitoring.

Runs HandLandmarker on a frame stream — a webcam (`source=0`) or a video
file — and maintains:

    - the full landmark sequence (for the offline feature pipeline at the end)
    - rolling live stats: wrist speed (EMA), detection rate, elapsed signing
      time, running letters-per-second when a transcript is given

With a display available it shows the skeleton overlay plus a HUD; headless
it can still save an annotated video (used for testing and for the paper's
qualitative figures). Press Q to stop early when displayed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .extractor import (
    N_LANDMARKS,
    WRIST,
    LandmarkSequence,
    SigningHandSelector,
    _make_landmarker,
    draw_landmarks,
)


@dataclass
class LiveStats:
    wrist_speed: float = 0.0       # exponential moving average, norm-units/s
    detection_rate: float = 1.0
    elapsed_s: float = 0.0
    letters_per_second: float | None = None


@dataclass
class LiveFluencyMonitor:
    """Incremental landmark tracker over a live frame stream."""

    model_path: str | Path
    transcript: str | None = None
    max_num_hands: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    ema_alpha: float = 0.3
    _landmarker: object = field(init=False, default=None)
    _frames: list = field(init=False, default_factory=list)
    _oks: list = field(init=False, default_factory=list)
    _prev_wrist: np.ndarray | None = field(init=False, default=None)
    _fps: float = field(init=False, default=30.0)
    _stats: LiveStats = field(init=False, default_factory=LiveStats)

    def __post_init__(self) -> None:
        self._landmarker = _make_landmarker(
            self.model_path, self.max_num_hands,
            self.min_detection_confidence, self.min_tracking_confidence,
        )
        self._selector = SigningHandSelector()

    def process_frame(self, bgr: np.ndarray, timestamp_ms: int) -> tuple[np.ndarray, LiveStats]:
        """Track one frame; returns (annotated BGR, live stats)."""
        import mediapipe as mp

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect_for_video(image, timestamp_ms)

        lm = self._selector.select(res)

        if lm is None:
            self._frames.append(np.full((N_LANDMARKS, 3), np.nan, dtype=np.float32))
            self._oks.append(False)
        else:
            wrist = lm[WRIST, :2]
            if self._prev_wrist is not None:
                inst = float(np.linalg.norm(wrist - self._prev_wrist)) * self._fps
                self._stats.wrist_speed = (
                    self.ema_alpha * inst + (1 - self.ema_alpha) * self._stats.wrist_speed
                )
            self._prev_wrist = wrist
            self._frames.append(lm)
            self._oks.append(True)

        n = len(self._oks)
        self._stats.detection_rate = float(np.mean(self._oks))
        self._stats.elapsed_s = n / self._fps
        if self.transcript:
            letters = sum(c.isalpha() for c in self.transcript)
            self._stats.letters_per_second = letters / self._stats.elapsed_s if n else None

        return draw_landmarks(bgr, lm), self._stats

    def finalize(self, clip_id: str = "live") -> LandmarkSequence:
        """Close the stream and return the accumulated LandmarkSequence."""
        self._landmarker.close()
        return LandmarkSequence(
            clip_id=clip_id,
            landmarks=np.stack(self._frames) if self._frames
            else np.zeros((0, N_LANDMARKS, 3), np.float32),
            frame_ok=np.asarray(self._oks, dtype=bool),
            fps=self._fps,
        )


def run_live(
    source: int | str,
    monitor: LiveFluencyMonitor,
    *,
    display: bool = True,
    save_video: str | Path | None = None,
    default_fps: float = 30.0,
) -> LandmarkSequence:
    """Drive a monitor from a webcam index, video file, or frame directory."""
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise OSError(f"Cannot open source: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or default_fps
    else:
        from .extractor import open_frame_source

        cap, fps = open_frame_source(source, default_fps)
    monitor._fps = fps

    writer = None
    t0 = time.monotonic()
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            ts = int((time.monotonic() - t0) * 1000)
            annotated, stats = monitor.process_frame(bgr, ts)
            _draw_hud(annotated, stats)
            if save_video:
                if writer is None:  # lazy init: frame dirs report no width/height
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(str(save_video),
                                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(annotated)
            if display:
                cv2.imshow("ASL fluency (Q to stop)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
    return monitor.finalize()


def _draw_hud(frame: np.ndarray, stats: LiveStats) -> None:
    lines = [
        f"wrist speed: {stats.wrist_speed:5.2f} u/s",
        f"detection:   {stats.detection_rate * 100:5.1f}%",
        f"elapsed:     {stats.elapsed_s:5.1f}s",
    ]
    if stats.letters_per_second is not None:
        lines.append(f"letters/s:   {stats.letters_per_second:5.2f}")
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
