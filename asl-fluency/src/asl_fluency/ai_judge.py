"""AI-judge fluency pre-scoring (auxiliary labels, never ground truth).

A video-capable VLM scores each clip 1-5 with the rubric below. The scores
feed three things, all downstream of the *human* ground truth:

1. **Stratification** — `rating-sheet --stratify-by ai_score` draws human
   rating batches that span the full 1-5 range (ChicagoFSWild skews fluent,
   so uniform sampling under-covers disfluent clips).
2. **Active top-ups** — after a pilot batch, add clips where the kinematic
   model and the AI judge disagree most.
3. **A secondary, large-N evaluation** — kinematic model vs. AI scores on
   the full pool, explicitly labelled as proxy evaluation.

AI rows are written in the same long format as human ratings with a
``vlm_``-prefixed rater id, so `evaluation.agreement` computes human-vs-AI
agreement with no schema changes, and `consensus_scores` excludes them from
ground truth automatically.

Backends: the deterministic ``mock`` backend ships enabled (for testing the
plumbing), and ``grok`` (xAI) is implemented via the OpenAI-compatible chat
API over evenly-sampled frames — Grok has no video-input endpoint, so clips
are sent as N frames in temporal order rather than as an mp4. To add another
provider, implement a ``_score_<name>`` function and register it in
``_score_with_backend`` — the prompt, rendering, and CSV plumbing are done.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data.fluency_labels import DEFAULT_AI_RATER_ID, REQUIRED_COLUMNS, save_rating
from .landmarks.extractor import open_frame_source

# ---------------------------------------------------------------------------
# Rubric prompt. Keep the scale anchors identical to what human raters see in
# the browser UI so human-vs-AI agreement is meaningful.
# ---------------------------------------------------------------------------

FLUENCY_RUBRIC_PROMPT = """\
You are rating the FLUENCY of ASL fingerspelling in a short video clip — how \
smoothly and consistently the person spells, NOT what they spell and NOT \
whether the letters are correct.

Watch the whole clip, then rate overall fingerspelling fluency on this scale:

1 - Very disfluent: slow, hesitant, long pauses between letters, jerky or \
restarted transitions, visible searching for handshapes.
2 - Disfluent: frequent hesitations or uneven rhythm; transitions between \
letters are rough; pace is slow.
3 - Moderate: understandable rhythm with occasional hesitations; some rough \
or inconsistent transitions; moderate pace.
4 - Fluent: smooth, even rhythm; quick and consistent letter transitions; \
only minor irregularities.
5 - Very fluent: fast, effortless, highly consistent rhythm; seamless \
letter-to-letter transitions with no visible hesitation.

Consider: signing speed, smoothness of hand motion between letters, \
consistency of letter timing, and pause/hesitation patterns. Ignore video \
quality, clothing, background, and the signer's appearance.

Respond with ONLY a JSON object on one line, no other text:
{"rating": <integer 1-5>, "confidence": <low|medium|high>, \
"rationale": "<one sentence>"}\
"""


def build_prompt() -> str:
    """The full prompt sent to the VLM for one clip.

    Deliberately blind to the transcript: fluency is about motion quality,
    and showing the target word would invite the judge to grade spelling
    correctness instead.
    """
    return FLUENCY_RUBRIC_PROMPT


# ---------------------------------------------------------------------------
# Clip rendering for VLM upload
# ---------------------------------------------------------------------------

def render_clip_video(
    clip_path: str | Path, out_path: str | Path, fps: float = 15.0
) -> Path:
    """Render a clip (frame directory or video file) to an mp4 for upload.

    Frame directories (ChicagoFSWild's native layout) have no container, so
    most video APIs need them reassembled; 15 fps keeps files small while
    preserving transition dynamics.
    """
    import cv2

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap, src_fps = open_frame_source(clip_path, default_fps=fps)
    writer = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h)
            )
        writer.write(frame)
    if writer is None:
        raise OSError(f"No frames readable from {clip_path}")
    writer.release()
    return out_path


def render_contact_sheet(
    clip_path: str | Path,
    out_path: str | Path,
    max_frames: int = 16,
    cols: int = 4,
) -> Path:
    """Fallback for image-only APIs: an evenly-sampled grid of frames."""
    import cv2

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap, _ = open_frame_source(clip_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    if not frames:
        raise OSError(f"No frames readable from {clip_path}")
    idx = np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).astype(int)
    picked = [frames[i] for i in idx]
    h, w = picked[0].shape[:2]
    rows = int(np.ceil(len(picked) / cols))
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, fr in enumerate(picked):
        r, c = divmod(i, cols)
        sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = fr
    cv2.imwrite(str(out_path), sheet)
    return out_path


# ---------------------------------------------------------------------------
# Scoring backends
# ---------------------------------------------------------------------------

def score_clip(
    clip_path: str | Path,
    backend: str = "mock",
    work_dir: str | Path | None = None,
    clip_id: str | None = None,
    **opts: Any,
) -> dict:
    """Score one clip; returns {"rating": int, "confidence": str, "rationale": str}.

    ``**opts`` carries backend settings from the ``ai_judge`` config section
    (model, n_frames, temperature, jpeg_quality); CLI flags override config.
    """
    if backend == "mock":
        return _score_mock(clip_path)
    return _score_with_backend(clip_path, backend, work_dir, clip_id, **opts)


def _score_mock(clip_path: str | Path) -> dict:
    """Deterministic placeholder so the pipeline is testable without a VLM.

    Hashes the clip path to a stable pseudo-rating. Obviously carries no
    fluency signal — for plumbing tests only.
    """
    digest = hashlib.sha1(str(clip_path).encode()).hexdigest()
    rating = 1 + int(digest[:8], 16) % 5
    return {
        "rating": rating,
        "confidence": "low",
        "rationale": "mock backend (no VLM call)",
    }


def _score_with_backend(
    clip_path: str | Path,
    backend: str,
    work_dir: str | Path | None,
    clip_id: str | None,
    **opts: Any,
) -> dict:
    """Live VLM backends. Register new providers here."""
    if backend == "grok":
        return _score_grok(clip_path, work_dir, clip_id, **opts)
    raise NotImplementedError(
        f"AI backend '{backend}' is not implemented. Available: mock, grok. "
        "To add a provider, write a _score_<name> function and route it here "
        "(render_clip_video / render_contact_sheet + build_prompt are ready)."
    )


def _sample_frames(clip_path: str | Path, n_frames: int) -> list:
    """Evenly sample up to n_frames frames from a clip, in temporal order."""
    cap, _ = open_frame_source(clip_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    if not frames:
        raise OSError(f"No frames readable from {clip_path}")
    idx = np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).astype(int)
    return [frames[i] for i in idx]


def _score_grok(
    clip_path: str | Path,
    work_dir: str | Path | None,
    clip_id: str | None,
    *,
    model: str = "grok-4.7",
    n_frames: int = 16,
    temperature: float = 0,
    jpeg_quality: int = 80,
    **_: Any,
) -> dict:
    """xAI Grok backend: vision chat over N frames sent in temporal order.

    Grok's chat API has no video input, so instead of an mp4 we send evenly
    sampled JPEG frames as separate images (xAI allows many images per
    request). Frame order is stated in the prompt so the judge can reason
    about frame-to-frame motion — the actual fluency signal.

    Requires: XAI_API_KEY in the environment (config.load_dotenv() picks up
    .env automatically; see example.env) and the `openai` package.
    """
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "XAI_API_KEY is not set. Copy example.env to .env and fill it in "
            "(config.py loads .env automatically), or export it in your shell."
        )
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("grok backend needs the openai package: uv add openai") from e
    import cv2

    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

    frames = _sample_frames(clip_path, n_frames)
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"These {len(frames)} images are frames from ONE fingerspelling "
                "clip, in temporal order at even intervals. Judge the smoothness "
                "and consistency of motion ACROSS frames, then answer per the "
                "rubric.\n\n" + build_prompt()
            ),
        }
    ]
    for frame in frames:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            raise OSError(f"JPEG encoding failed for a frame of {clip_path}")
        b64 = base64.b64encode(buf).decode()
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    payload = json.loads(resp.choices[0].message.content)
    payload["rating"] = int(payload["rating"])
    assert 1 <= payload["rating"] <= 5, f"rating out of range: {payload}"

    if work_dir:  # keep raw responses for auditability
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        name = f"{clip_id or Path(str(clip_path)).stem}.json"
        (Path(work_dir) / name).write_text(json.dumps(payload, indent=2))
    return payload


def run_ai_rating(
    clip_paths: dict[str, str | Path],
    out_csv: str | Path,
    backend: str = "mock",
    rater_id: str = DEFAULT_AI_RATER_ID,
    work_dir: str | Path | None = None,
    limit: int | None = None,
    **score_opts: Any,
) -> pd.DataFrame:
    """Score a pool of clips and write/extend a long-format AI ratings CSV.

    Saves incrementally (one row per clip via save_rating) so a long run
    survives interruption and can simply be re-run — clips already present in
    ``out_csv`` are skipped. ``**score_opts`` (model, n_frames, …) is passed
    through to the backend.
    """
    out_csv = Path(out_csv)
    done: set[str] = set()
    if out_csv.exists():
        existing = pd.read_csv(out_csv, dtype={"clip_id": str, "rater_id": str})
        done = set(existing.loc[existing["rater_id"] == rater_id, "clip_id"])

    scored = 0
    for clip_id, path in clip_paths.items():
        if clip_id in done:
            continue
        if limit is not None and scored >= limit:
            break
        result = score_clip(
            path, backend=backend, work_dir=work_dir, clip_id=clip_id, **score_opts
        )
        save_rating(out_csv, clip_id, rater_id, int(result["rating"]))
        scored += 1

    if out_csv.exists():
        return pd.read_csv(out_csv, dtype={"clip_id": str, "rater_id": str})
    return pd.DataFrame(columns=list(REQUIRED_COLUMNS))
