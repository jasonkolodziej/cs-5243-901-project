"""Browser-based labeling UIs for human ground truth.

Two modes sharing one stdlib http.server scaffold:

- **rate** (`asl-fluency rate`) — raters watch each clip (looping, speed
  control) and click a 1-5 fluency rating; saves into the long-format intake
  CSV on every click.
- **boundaries** (`asl-fluency label-boundaries`) — one labeler marks
  per-letter [start, end) frame spans (Roadmap Gap 1 ground truth) by
  scrubbing the flipbook and clicking the joints between letters; saves into
  data/ratings/boundary_labels.csv.

Shared design choices:
- **No new dependencies** — stdlib http.server + inline HTML pages.
- **Frame-directory native** — ChicagoFSWild clips are JPEG directories, so
  clips play as a JS flipbook (no codec/container issues); real video files
  are served to a <video> tag instead.
- **Blind to extra context in rate mode** — the transcript is never shown to
  fluency raters. Boundary mode shows the transcript (it IS the task).
- **Per-rater clip order in rate mode** — deterministic but different
  shuffles per rater wash out ordering effects.

Only clips registered in the catalog are served; no directory browsing.
"""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from ..landmarks.extractor import IMAGE_SUFFIXES
from .boundary_labels import load_boundary_labels, save_clip_spans, spans_for
from .fluency_labels import REQUIRED_COLUMNS, save_rating

_CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
}


class ClipCatalog:
    """clip_id -> playable media info for the clips being labeled."""

    def __init__(self, clips: dict[str, Path], default_fps: float = 30.0):
        self.default_fps = default_fps
        self.clips: dict[str, dict] = {}
        for clip_id, path in clips.items():
            path = Path(path)
            if path.is_dir():
                files = sorted(
                    p.name for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
                )
                if files:
                    self.clips[clip_id] = {
                        "kind": "frames",
                        "path": path,
                        "files": files,
                        "fps": default_fps,
                    }
            elif path.is_file():
                self.clips[clip_id] = {"kind": "video", "path": path}

    def info(self, clip_id: str) -> dict | None:
        c = self.clips.get(clip_id)
        if c is None:
            return None
        if c["kind"] == "frames":
            return {
                "clip_id": clip_id,
                "kind": "frames",
                "fps": c["fps"],
                "n_frames": len(c["files"]),
                "frames": [f"/media/{clip_id}/{name}" for name in c["files"]],
            }
        return {
            "clip_id": clip_id,
            "kind": "video",
            "url": f"/media/{clip_id}/{c['path'].name}",
        }

    def media_file(self, clip_id: str, name: str) -> Path | None:
        """Resolve a media request to a file, guarding against traversal."""
        c = self.clips.get(clip_id)
        if c is None or Path(name).name != name:
            return None
        if c["kind"] == "frames" and name in c["files"]:
            return c["path"] / name
        if c["kind"] == "video" and name == c["path"].name:
            return c["path"]
        return None


class RatingStore:
    """Thread-safe read/upsert over the long-format intake CSV."""

    def __init__(self, path: Path, rating_scale: tuple[int, int] = (1, 5)):
        self.path = Path(path)
        self.scale = rating_scale
        self.lock = threading.Lock()

    def load(self) -> pd.DataFrame:
        if self.path.exists():
            return pd.read_csv(self.path, dtype={"clip_id": str, "rater_id": str}).fillna(
                {"rating": ""}
            )
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS))

    def raters(self) -> list[str]:
        return sorted(self.load()["rater_id"].unique())

    def rate(self, clip_id: str, rater_id: str, rating: int | None) -> None:
        with self.lock:
            save_rating(self.path, clip_id, rater_id, rating, self.scale)

    def progress(self, rater_id: str) -> dict:
        df = self.load()
        mine = df[df["rater_id"] == rater_id]
        rated = mine[pd.to_numeric(mine["rating"], errors="coerce").notna()]
        return {
            "rater_id": rater_id,
            "n_total": len(mine),
            "n_rated": len(rated),
            "ratings": dict(zip(rated["clip_id"], rated["rating"].astype(float))),
        }


class BoundaryStore:
    """Thread-safe read/save over boundary_labels.csv."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.Lock()

    def labels(self) -> pd.DataFrame:
        return load_boundary_labels(self.path)

    def spans(self, clip_id: str) -> list[tuple[int, int, str]]:
        return spans_for(self.labels(), clip_id)

    def save(self, clip_id: str, spans: list[tuple[int, int, str]]) -> None:
        with self.lock:
            save_clip_spans(self.path, clip_id, spans)

    def labeled_clip_ids(self) -> set[str]:
        return set(self.labels()["clip_id"].unique())


def per_rater_order(clip_ids: list[str], rater_id: str, seed: int = 42) -> list[str]:
    """Deterministic per-rater shuffle: same order on resume, different
    order across raters, so ordering effects average out."""
    return sorted(
        clip_ids,
        key=lambda c: hashlib.sha1(f"{seed}:{rater_id}:{c}".encode()).hexdigest(),
    )


def make_handler(
    store: RatingStore | None,
    catalog: ClipCatalog,
    seed: int = 42,
    mode: str = "rate",
    boundary_store: BoundaryStore | None = None,
    transcripts: dict[str, str] | None = None,
):
    """Build the request handler bound to stores + catalog."""

    class LabelingHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        # -- helpers ------------------------------------------------------
        def _send(self, body: bytes, content_type: str = "text/html", status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200):
            self._send(json.dumps(obj).encode(), "application/json", status)

        # -- routes -------------------------------------------------------
        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self._send((_BOUNDARY_PAGE if mode == "boundaries" else _PAGE).encode())
            elif url.path == "/api/session":
                if mode == "boundaries":
                    clips = sorted(catalog.clips)
                    labeled = boundary_store.labeled_clip_ids() & set(clips)
                    self._json(
                        {
                            "clips": clips,
                            "progress": {
                                "n_total": len(clips),
                                "n_labeled": len(labeled),
                                "labeled": sorted(labeled),
                            },
                        }
                    )
                else:
                    rater = parse_qs(url.query).get("rater", [""])[0]
                    df = store.load()
                    clips = sorted(df["clip_id"].unique())
                    if rater:
                        clips = per_rater_order(clips, rater, seed)
                    self._json(
                        {
                            "raters": store.raters(),
                            "clips": clips,
                            "progress": store.progress(rater) if rater else None,
                        }
                    )
            elif url.path.startswith("/api/clip_label/"):
                clip_id = url.path.removeprefix("/api/clip_label/")
                spans = boundary_store.spans(clip_id)
                self._json(
                    {
                        "clip_id": clip_id,
                        "transcript": (transcripts or {}).get(clip_id, ""),
                        "spans": [
                            {"letter": l, "start_frame": s, "end_frame": e}
                            for s, e, l in spans
                        ],
                    }
                )
            elif url.path.startswith("/api/clip/"):
                clip_id = url.path.removeprefix("/api/clip/")
                info = catalog.info(clip_id)
                if info is None:
                    self._json({"error": "unknown or unreadable clip"}, 404)
                else:
                    self._json(info)
            elif url.path.startswith("/media/"):
                _, _, clip_id, name = url.path.split("/", 3)
                path = catalog.media_file(clip_id, name)
                if path is None:
                    self._json({"error": "not found"}, 404)
                else:
                    ctype = _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
                    self._send(path.read_bytes(), ctype)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            url = urlparse(self.path)
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            except (json.JSONDecodeError, TypeError) as e:
                self._json({"error": str(e)}, 400)
                return
            if url.path == "/api/rate" and mode == "rate":
                try:
                    rating = body.get("rating")
                    rating = int(rating) if rating is not None else None
                    store.rate(str(body["clip_id"]), str(body["rater_id"]), rating)
                    self._json(
                        {"ok": True, "progress": store.progress(str(body["rater_id"]))}
                    )
                except (ValueError, KeyError) as e:
                    self._json({"error": str(e)}, 400)
            elif url.path == "/api/label" and mode == "boundaries":
                try:
                    spans = [
                        (int(s["start_frame"]), int(s["end_frame"]), str(s["letter"]))
                        for s in body["spans"]
                    ]
                    boundary_store.save(str(body["clip_id"]), spans)
                    self._json({"ok": True})
                except (ValueError, KeyError, TypeError) as e:
                    self._json({"error": str(e)}, 400)
            else:
                self._json({"error": "not found"}, 404)

    return LabelingHandler


def serve(
    ratings_path: str | Path | None,
    clip_paths: dict[str, Path],
    host: str = "127.0.0.1",
    port: int = 8765,
    rating_scale: tuple[int, int] = (1, 5),
    default_fps: float = 30.0,
    seed: int = 42,
    mode: str = "rate",
    boundary_path: str | Path | None = None,
    transcripts: dict[str, str] | None = None,
) -> ThreadingHTTPServer:
    """Create the labeling server (call .serve_forever() or .handle_request())."""
    if mode not in ("rate", "boundaries"):
        raise ValueError(f"unknown UI mode: {mode}")
    store = RatingStore(Path(ratings_path), rating_scale) if mode == "rate" else None
    boundary_store = (
        BoundaryStore(Path(boundary_path)) if mode == "boundaries" else None
    )
    catalog = ClipCatalog(clip_paths, default_fps)
    server = ThreadingHTTPServer(
        (host, port), make_handler(store, catalog, seed, mode, boundary_store, transcripts)
    )
    server.rating_store = store  # exposed for tests
    server.boundary_store = boundary_store
    server.clip_catalog = catalog
    return server


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ASL Fingerspelling Fluency Rating</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, sans-serif; background: #14171c;
         color: #e8ebef; display: flex; min-height: 100vh; }
  aside { width: 300px; padding: 20px; background: #1c2129; border-right: 1px solid #2a313c;
          font-size: 13px; line-height: 1.5; }
  aside h1 { font-size: 16px; margin: 0 0 12px; }
  aside ol { padding-left: 18px; margin: 8px 0; }
  aside li { margin-bottom: 8px; }
  main { flex: 1; display: flex; flex-direction: column; align-items: center;
         padding: 24px; gap: 16px; }
  select, button { font: inherit; }
  #rater { padding: 6px 10px; background: #232a35; color: inherit;
           border: 1px solid #39424f; border-radius: 6px; }
  #stage { width: min(640px, 90%); aspect-ratio: 4/3; background: #000;
           border-radius: 10px; display: flex; align-items: center;
           justify-content: center; overflow: hidden; position: relative; }
  #stage img, #stage video { max-width: 100%; max-height: 100%; }
  #placeholder { color: #5b6675; padding: 20px; text-align: center; }
  #buttons { display: flex; gap: 12px; }
  .rate-btn { width: 64px; height: 64px; font-size: 24px; border-radius: 10px;
              border: 2px solid #39424f; background: #232a35; color: inherit;
              cursor: pointer; transition: all .1s; }
  .rate-btn:hover { border-color: #6ea8fe; transform: translateY(-2px); }
  .rate-btn.active { background: #2f6fed; border-color: #6ea8fe; }
  #nav { display: flex; gap: 12px; align-items: center; }
  #nav button { padding: 8px 18px; border-radius: 8px; border: 1px solid #39424f;
                background: #232a35; color: inherit; cursor: pointer; }
  #nav button:hover { border-color: #6ea8fe; }
  #progressbar { width: min(640px, 90%); height: 6px; background: #232a35;
                 border-radius: 3px; overflow: hidden; }
  #progressbar div { height: 100%; background: #2f6fed; width: 0; transition: width .2s; }
  #status { color: #9aa5b1; font-size: 14px; }
  .hint { color: #5b6675; font-size: 12px; }
  #speed { padding: 4px 8px; background: #232a35; color: inherit;
           border: 1px solid #39424f; border-radius: 6px; }
</style>
</head>
<body>
<aside>
  <h1>Fluency rating rubric</h1>
  <p>Rate <b>how fluently</b> the person fingerspells &mdash; speed, smoothness,
     and consistency of letter transitions. <b>Not</b> what they spell, and not
     whether the spelling is correct.</p>
  <ol>
    <li><b>1 &mdash; Very disfluent:</b> slow, hesitant, long pauses, jerky or
        restarted transitions.</li>
    <li><b>2 &mdash; Disfluent:</b> frequent hesitations, uneven rhythm, rough
        transitions.</li>
    <li><b>3 &mdash; Moderate:</b> understandable rhythm, occasional hesitations,
        moderate pace.</li>
    <li><b>4 &mdash; Fluent:</b> smooth even rhythm, quick consistent transitions,
        minor irregularities.</li>
    <li><b>5 &mdash; Very fluent:</b> fast, effortless, highly consistent, seamless
        transitions.</li>
  </ol>
  <p class="hint">Keyboard: 1&ndash;5 rate &middot; &larr;/&rarr; prev/next &middot;
     space play/pause. Ratings save instantly; you can close and resume any time.</p>
</aside>
<main>
  <div>
    <label for="rater">Rater: </label>
    <select id="rater"></select>
    <label for="speed" style="margin-left:16px">Speed: </label>
    <select id="speed">
      <option value="0.25">0.25&times;</option>
      <option value="0.5">0.5&times;</option>
      <option value="1" selected>1&times;</option>
      <option value="2">2&times;</option>
    </select>
  </div>
  <div id="stage"><div id="placeholder">Select a rater to begin</div></div>
  <div id="buttons"></div>
  <div id="nav">
    <button id="prev">&larr; Prev</button>
    <span id="status"></span>
    <button id="next">Next &rarr;</button>
  </div>
  <div id="progressbar"><div></div></div>
</main>
<script>
let raters = [], clips = [], ratings = {}, rater = "", idx = 0;
let frames = [], frameIdx = 0, timer = null, baseFps = 12, playing = true;

const $ = s => document.querySelector(s);

async function api(path, opts) {
  const r = await fetch(path, opts);
  return r.json();
}

async function init() {
  const sess = await api("/api/session");
  raters = sess.raters;
  const sel = $("#rater");
  sel.innerHTML = '<option value="">— choose —</option>' +
    raters.map(r => `<option>${r}</option>`).join("");
  const btns = $("#buttons");
  for (let v = 1; v <= 5; v++) {
    const b = document.createElement("button");
    b.className = "rate-btn"; b.textContent = v;
    b.onclick = () => rate(v);
    btns.appendChild(b);
  }
}

async function pickRater() {
  rater = $("#rater").value;
  if (!rater) return;
  const sess = await api("/api/session?rater=" + encodeURIComponent(rater));
  clips = sess.clips;
  ratings = sess.progress ? sess.progress.ratings : {};
  idx = 0;
  jumpToFirstUnrated();
  await show();
  updateStatus();
}

function jumpToFirstUnrated() {
  const i = clips.findIndex(c => !(c in ratings));
  idx = i === -1 ? 0 : i;
}

function stopPlayback() {
  if (timer) { clearInterval(timer); timer = null; }
  const v = $("#stage video"); if (v) v.pause();
}

async function show() {
  stopPlayback();
  if (!clips.length) return;
  const clip = clips[idx];
  const info = await api("/api/clip/" + encodeURIComponent(clip));
  const stage = $("#stage");
  stage.innerHTML = "";
  if (info.kind === "frames") {
    const img = document.createElement("img");
    stage.appendChild(img);
    frames = info.frames.map(src => { const im = new Image(); im.src = src; return im; });
    baseFps = info.fps || 12;
    frameIdx = 0; playing = true;
    img.src = frames[0].src;
    startFlipbook(img);
  } else {
    const v = document.createElement("video");
    v.src = info.url; v.loop = true; v.muted = true; v.autoplay = true;
    v.playbackRate = parseFloat($("#speed").value);
    stage.appendChild(v);
  }
  markRated();
  updateStatus();
}

function startFlipbook(img) {
  const step = () => {
    frameIdx = (frameIdx + 1) % frames.length;
    img.src = frames[frameIdx].src;
  };
  const tick = () => 1000 / (baseFps * parseFloat($("#speed").value));
  timer = setInterval(step, tick());
}

function markRated() {
  const cur = clips[idx];
  document.querySelectorAll(".rate-btn").forEach(b =>
    b.classList.toggle("active", Number(b.textContent) === Number(ratings[cur])));
}

function updateStatus() {
  const rated = Object.keys(ratings).length;
  $("#status").textContent = `clip ${idx + 1}/${clips.length} · ${rated}/${clips.length} rated`;
  $("#progressbar div").style.width = (clips.length ? 100 * rated / clips.length : 0) + "%";
}

async function rate(v) {
  if (!rater || !clips.length) return;
  const clip = clips[idx];
  const res = await api("/api/rate", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({rater_id: rater, clip_id: clip, rating: v}),
  });
  if (res.ok) {
    ratings[clip] = v;
    markRated(); updateStatus();
    if (idx < clips.length - 1) { idx++; await show(); }
  }
}

function nav(d) {
  if (!clips.length) return;
  idx = (idx + d + clips.length) % clips.length;
  show();
}

$("#rater").onchange = pickRater;
$("#prev").onclick = () => nav(-1);
$("#next").onclick = () => nav(1);
$("#speed").onchange = () => {
  const v = $("#stage video");
  if (v) v.playbackRate = parseFloat($("#speed").value);
  if (timer) { const img = $("#stage img"); clearInterval(timer); timer = setInterval(() => {
    frameIdx = (frameIdx + 1) % frames.length; img.src = frames[frameIdx].src;
  }, 1000 / (baseFps * parseFloat($("#speed").value))); }
};
document.addEventListener("keydown", e => {
  if (e.key >= "1" && e.key <= "5") rate(Number(e.key));
  else if (e.key === "ArrowRight") nav(1);
  else if (e.key === "ArrowLeft") nav(-1);
  else if (e.key === " ") {
    e.preventDefault();
    const v = $("#stage video");
    if (v) { v.paused ? v.play() : v.pause(); }
    else if (timer) { clearInterval(timer); timer = null; playing = false; }
    else if (frames.length) { playing = true;
      const img = $("#stage img");
      timer = setInterval(() => { frameIdx = (frameIdx + 1) % frames.length;
        img.src = frames[frameIdx].src; }, 1000 / (baseFps * parseFloat($("#speed").value))); }
  }
});
init();
</script>
</body>
</html>
"""


_BOUNDARY_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ASL Letter-Boundary Labeling</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, sans-serif; background: #14171c;
         color: #e8ebef; display: flex; min-height: 100vh; }
  aside { width: 280px; padding: 20px; background: #1c2129; border-right: 1px solid #2a313c;
          font-size: 13px; line-height: 1.5; }
  aside h1 { font-size: 16px; margin: 0 0 12px; }
  main { flex: 1; display: flex; flex-direction: column; align-items: center;
         padding: 24px; gap: 14px; }
  select, button { font: inherit; }
  #stage { width: min(640px, 90%); aspect-ratio: 4/3; background: #000;
           border-radius: 10px; display: flex; align-items: center;
           justify-content: center; overflow: hidden; }
  #stage img { max-width: 100%; max-height: 100%; }
  #scrub { width: min(640px, 90%); }
  #frameLabel { color: #9aa5b1; font-size: 13px; font-variant-numeric: tabular-nums; }
  #letters { display: flex; align-items: stretch; gap: 0; margin-top: 4px;
             flex-wrap: wrap; justify-content: center; }
  .letter { padding: 10px 14px; background: #232a35; border: 2px solid #39424f;
            border-radius: 8px; text-align: center; min-width: 64px; }
  .letter b { font-size: 20px; display: block; }
  .letter span { font-size: 11px; color: #9aa5b1; font-variant-numeric: tabular-nums; }
  .joint { width: 44px; margin: 0 2px; border-radius: 8px; border: 2px dashed #39424f;
           background: #1c2129; color: #9aa5b1; cursor: pointer;
           font-size: 11px; font-variant-numeric: tabular-nums; align-self: center;
           padding: 14px 0; }
  .joint:hover { border-color: #6ea8fe; color: #e8ebef; }
  .joint.set { border-style: solid; border-color: #2f6fed; color: #e8ebef; }
  #nav { display: flex; gap: 12px; align-items: center; }
  #nav button, #controls button { padding: 8px 18px; border-radius: 8px;
    border: 1px solid #39424f; background: #232a35; color: inherit; cursor: pointer; }
  #nav button:hover, #controls button:hover { border-color: #6ea8fe; }
  #controls { display: flex; gap: 10px; align-items: center; }
  #progressbar { width: min(640px, 90%); height: 6px; background: #232a35;
                 border-radius: 3px; overflow: hidden; }
  #progressbar div { height: 100%; background: #2f6fed; width: 0; transition: width .2s; }
  #status { color: #9aa5b1; font-size: 14px; }
  .hint { color: #5b6675; font-size: 12px; }
  #savestate { font-size: 12px; color: #4caf7d; min-height: 16px; }
</style>
</head>
<body>
<aside>
  <h1>Letter-boundary labeling</h1>
  <p>Mark where each letter of the transcript begins and ends. Spans tile the
     clip: each letter ends where the next begins.</p>
  <p><b>How:</b> scrub or play the clip; when the signer finishes a letter,
     click the dashed <i>joint</i> between that letter and the next &mdash; the
     boundary snaps to the current frame and saves automatically.</p>
  <p class="hint">Keyboard: &larr;/&rarr; step frame &middot; space play/pause
     &middot; Enter set next unset joint &middot; PgUp/PgDn prev/next clip.
     Boundaries pre-fill with an even split; adjust as needed. Clicking a set
     joint re-marks it.</p>
</aside>
<main>
  <div id="status">loading…</div>
  <div id="stage"></div>
  <input type="range" id="scrub" min="0" max="0" value="0">
  <div id="controls">
    <button id="play">&#9654;/&#10073;&#10073;</button>
    <button id="stepback">&larr;</button>
    <button id="stepfwd">&rarr;</button>
    <span id="frameLabel">frame 0/0</span>
    <select id="speed">
      <option value="0.25">0.25&times;</option>
      <option value="0.5" selected>0.5&times;</option>
      <option value="1">1&times;</option>
      <option value="2">2&times;</option>
    </select>
  </div>
  <div id="letters"></div>
  <div id="savestate"></div>
  <div id="nav">
    <button id="prev">&larr; Prev clip</button>
    <button id="next">Next clip &rarr;</button>
  </div>
  <div id="progressbar"><div></div></div>
</main>
<script>
let clips = [], labeled = new Set(), idx = 0;
let frames = [], nFrames = 0, cur = 0, timer = null, baseFps = 12;
let letters = [], joints = [];

const $ = s => document.querySelector(s);
async function api(path, opts) { return (await fetch(path, opts)).json(); }

async function init() {
  const sess = await api("/api/session");
  clips = sess.clips;
  labeled = new Set(sess.progress.labeled);
  updateProgress();
  await show();
}

async function show() {
  stop();
  if (!clips.length) { $("#status").textContent = "no clips"; return; }
  const clip = clips[idx];
  const [info, lab] = await Promise.all([
    api("/api/clip/" + encodeURIComponent(clip)),
    api("/api/clip_label/" + encodeURIComponent(clip)),
  ]);
  const stage = $("#stage"); stage.innerHTML = "";
  const img = document.createElement("img"); stage.appendChild(img);
  frames = info.frames.map(src => { const im = new Image(); im.src = src; return im; });
  baseFps = info.fps || 12;
  nFrames = info.n_frames;
  cur = 0;
  $("#scrub").max = nFrames - 1; $("#scrub").value = 0;

  letters = (lab.transcript || "").toUpperCase().replace(/[^A-Z]/g, "").split("");
  if (lab.spans && lab.spans.length === letters.length) {
    joints = [lab.spans[0].start_frame, ...lab.spans.map(s => s.end_frame)];
  } else {  // even-split starting guess
    joints = [];
    for (let i = 0; i <= letters.length; i++)
      joints.push(Math.round(i * nFrames / letters.length));
  }
  img.src = frames[0].src;
  renderLetters();
  updateStatus();
  $("#savestate").textContent = "";
}

function renderLetters() {
  const box = $("#letters"); box.innerHTML = "";
  letters.forEach((L, i) => {
    const chip = document.createElement("div");
    chip.className = "letter";
    chip.innerHTML = `<b>${L}</b><span>${joints[i]}–${joints[i + 1]}</span>`;
    box.appendChild(chip);
    if (i < letters.length - 1) {
      const j = document.createElement("button");
      j.className = "joint" + (joints[i + 1] !== Math.round((i + 1) * nFrames / letters.length) || labeled.has(clips[idx]) ? " set" : "");
      j.textContent = joints[i + 1];
      j.title = "set boundary to current frame";
      j.onclick = () => setJoint(i + 1);
      box.appendChild(j);
    }
  });
}

async function setJoint(k, silent) {
  // clamp: strictly between neighbouring joints
  const lo = joints[k - 1] + 1, hi = joints[k + 1] - 1;
  joints[k] = Math.max(lo, Math.min(hi, cur));
  renderLetters();
  await save();
}

async function setNextUnset() {
  for (let k = 1; k < joints.length - 1; k++) {
    const even = Math.round(k * nFrames / letters.length);
    if (joints[k] === even && !labeled.has(clips[idx])) { await setJoint(k); return; }
  }
}

async function save() {
  const spans = letters.map((L, i) =>
    ({letter: L, start_frame: joints[i], end_frame: joints[i + 1]}));
  const res = await api("/api/label", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({clip_id: clips[idx], spans}),
  });
  if (res.ok) {
    labeled.add(clips[idx]);
    $("#savestate").textContent = "saved ✓";
    updateProgress();
  } else {
    $("#savestate").textContent = "save failed: " + (res.error || "?");
  }
}

function setFrame(i) {
  cur = Math.max(0, Math.min(nFrames - 1, i));
  $("#stage img").src = frames[cur].src;
  $("#scrub").value = cur;
  $("#frameLabel").textContent = `frame ${cur}/${nFrames - 1}`;
}

function stop() { if (timer) { clearInterval(timer); timer = null; } }
function play() {
  if (timer) { stop(); return; }
  timer = setInterval(() => {
    if (cur >= nFrames - 1) { stop(); return; }
    setFrame(cur + 1);
  }, 1000 / (baseFps * parseFloat($("#speed").value)));
}

function updateStatus() {
  $("#status").textContent =
    `clip ${idx + 1}/${clips.length} — ${clips[idx]}` +
    (labeled.has(clips[idx]) ? " (labeled)" : "");
}
function updateProgress() {
  $("#progressbar div").style.width =
    (clips.length ? 100 * labeled.size / clips.length : 0) + "%";
  $("#status").textContent = clips.length
    ? `clip ${idx + 1}/${clips.length} — ${clips[idx]} · ${labeled.size}/${clips.length} labeled`
    : "no clips";
}

function nav(d) {
  if (!clips.length) return;
  idx = (idx + d + clips.length) % clips.length;
  show();
}

$("#play").onclick = play;
$("#stepback").onclick = () => { stop(); setFrame(cur - 1); };
$("#stepfwd").onclick = () => { stop(); setFrame(cur + 1); };
$("#scrub").oninput = e => { stop(); setFrame(Number(e.target.value)); };
$("#prev").onclick = () => nav(-1);
$("#next").onclick = () => nav(1);
document.addEventListener("keydown", e => {
  if (e.key === "ArrowLeft") { stop(); setFrame(cur - 1); }
  else if (e.key === "ArrowRight") { stop(); setFrame(cur + 1); }
  else if (e.key === " ") { e.preventDefault(); play(); }
  else if (e.key === "Enter") { e.preventDefault(); setNextUnset(); }
  else if (e.key === "PageDown") nav(1);
  else if (e.key === "PageUp") nav(-1);
});
init();
</script>
</body>
</html>
"""
