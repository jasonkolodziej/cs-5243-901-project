"""Browser-based rating UI for human fluency labels.

`asl-fluency rate` launches a local web server so raters watch each clip
(looping, speed control) and click a 1-5 rating. Ratings are written
straight into the long-format intake CSV on every click (atomic writes), so
the existing `agreement` / `evaluate` commands work on the file at any time
and a session can be interrupted and resumed.

Design choices:
- **No new dependencies** — stdlib http.server + a single inline HTML page.
- **Frame-directory native** — ChicagoFSWild clips are JPEG directories, so
  clips are played as a JS flipbook of frames (no codec/container issues);
  real video files are served to a <video> tag instead.
- **Blind rating** — the transcript is never shown; fluency is about motion
  quality, not spelling correctness.
- **Per-rater clip order** — each rater sees a deterministic but different
  shuffled order to wash out ordering effects.

Only clips present in the intake sheet are served; no directory browsing.
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
    """clip_id -> playable media info for the clips in an intake sheet."""

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


def per_rater_order(clip_ids: list[str], rater_id: str, seed: int = 42) -> list[str]:
    """Deterministic per-rater shuffle: same order on resume, different
    order across raters, so ordering effects average out."""
    return sorted(
        clip_ids,
        key=lambda c: hashlib.sha1(f"{seed}:{rater_id}:{c}".encode()).hexdigest(),
    )


def make_handler(store: RatingStore, catalog: ClipCatalog, seed: int = 42):
    """Build the request handler bound to a store + catalog."""

    class RatingHandler(BaseHTTPRequestHandler):
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
                self._send(_PAGE.encode())
            elif url.path == "/api/session":
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
            if url.path != "/api/rate":
                self._json({"error": "not found"}, 404)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                rating = body.get("rating")
                rating = int(rating) if rating is not None else None
                store.rate(str(body["clip_id"]), str(body["rater_id"]), rating)
                self._json({"ok": True, "progress": store.progress(str(body["rater_id"]))})
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                self._json({"error": str(e)}, 400)

    return RatingHandler


def serve(
    ratings_path: str | Path,
    clip_paths: dict[str, Path],
    host: str = "127.0.0.1",
    port: int = 8765,
    rating_scale: tuple[int, int] = (1, 5),
    default_fps: float = 30.0,
    seed: int = 42,
) -> ThreadingHTTPServer:
    """Create the rating server (call .serve_forever() or .handle_request())."""
    store = RatingStore(Path(ratings_path), rating_scale)
    catalog = ClipCatalog(clip_paths, default_fps)
    server = ThreadingHTTPServer((host, port), make_handler(store, catalog, seed))
    server.rating_store = store  # exposed for tests
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
