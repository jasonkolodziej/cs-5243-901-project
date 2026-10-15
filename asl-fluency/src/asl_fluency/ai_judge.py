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

Backends: only the deterministic ``mock`` backend ships enabled (for testing
the plumbing). To go live, implement ``_score_with_backend`` for your VLM of
choice (Gemini / GPT / Claude video APIs) — the prompt, rendering, and CSV
plumbing are already done.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

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
) -> dict:
    """Score one clip; returns {"rating": int, "confidence": str, "rationale": str}."""
    if backend == "mock":
        return _score_mock(clip_path)
    return _score_with_backend(clip_path, backend, work_dir, clip_id)


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
) -> dict:
    """Live VLM backends — implement this for your provider of choice.

    Suggested shape (any video API works the same way):

        video = render_clip_video(clip_path, Path(work_dir) / f"{clip_id}.mp4")
        response = <provider client>.generate(build_prompt(), video=video)
        payload = json.loads(response.text)   # keys: rating, confidence, rationale
        assert 1 <= int(payload["rating"]) <= 5
        return payload

    Tips: request the JSON-only response format if the provider supports it;
    retry with the contact-sheet fallback (render_contact_sheet) if the video
    upload fails; keep per-clip raw JSON responses under work_dir for
    auditability.
    """
    raise NotImplementedError(
        f"AI backend '{backend}' is not implemented yet. "
        "Implement models-independent scoring in ai_judge._score_with_backend "
        "(render_clip_video + build_prompt are ready), or use --backend mock "
        "to test the plumbing."
    )


def run_ai_rating(
    clip_paths: dict[str, str | Path],
    out_csv: str | Path,
    backend: str = "mock",
    rater_id: str = DEFAULT_AI_RATER_ID,
    work_dir: str | Path | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Score a pool of clips and write/extend a long-format AI ratings CSV.

    Saves incrementally (one row per clip via save_rating) so a long run
    survives interruption and can simply be re-run — clips already present in
    ``out_csv`` are skipped.
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
        result = score_clip(path, backend=backend, work_dir=work_dir, clip_id=clip_id)
        save_rating(out_csv, clip_id, rater_id, int(result["rating"]))
        scored += 1

    if out_csv.exists():
        return pd.read_csv(out_csv, dtype={"clip_id": str, "rater_id": str})
    return pd.DataFrame(columns=list(REQUIRED_COLUMNS))
