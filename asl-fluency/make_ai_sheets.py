#!/usr/bin/env python3
"""
make_ai_sheets.py — render contact sheets for the clips awaiting ratings and
zip them for upload to an external VLM judge.

For each unique clip_id in data/ratings/ratings_intake.csv:
  - resolves the local media path via data/interim/clip_index.csv
    (same logic as the CLI's rating UI)
  - renders a 16-frame, 4-column contact sheet (evenly sampled, temporal
    order, reading left-to-right top-to-bottom)
  - records it in a manifest

Blindness: the bundle contains ONLY clip_id -> image. Transcripts are
deliberately excluded — fluency must be judged from motion, not from knowing
the target word (same rule as ai_judge.build_prompt).

Run from the repo root:
  uv run python make_ai_sheets.py
  # -> ../asl-fluency-ai-sheets-<date>.zip  (upload this file)

Re-runs resume: sheets that already exist are not re-rendered.
"""

import csv
import sys
import time
import zipfile
from pathlib import Path

import cv2

from asl_fluency.ai_judge import render_contact_sheet
from asl_fluency.config import project_root

INTAKE = "data/ratings/ratings_intake.csv"
INDEX = "data/interim/clip_index.csv"
OUT_DIR = "data/interim/ai_sheets"
MAX_FRAMES = 16
COLS = 4
JPEG_QUALITY = 85


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def main() -> int:
    root = project_root()
    intake = root / INTAKE
    index = root / INDEX
    for p in (intake, index):
        if not p.exists():
            sys.exit(f"error: {p} not found — run from the repo root after "
                     "`asl-fluency index` and `asl-fluency rating-sheet`.")

    wanted = []
    with open(intake) as f:
        for row in csv.DictReader(f):
            if row["clip_id"] not in wanted:
                wanted.append(row["clip_id"])

    paths = {}
    with open(index) as f:
        for row in csv.DictReader(f):
            p = Path(row["video_path"])
            paths[row["clip_id"]] = p if p.is_absolute() else root / p

    out_dir = root / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest, missing, rendered, skipped = [], [], 0, 0
    for i, clip_id in enumerate(wanted, 1):
        sheet = out_dir / f"{clip_id}.jpg"
        src = paths.get(clip_id)
        if src is None or not src.exists():
            missing.append(clip_id)
            continue
        if sheet.exists():
            skipped += 1
        else:
            try:
                # render_contact_sheet picks format from the extension; write
                # PNG then re-encode as quality JPEG to keep the zip small
                tmp_png = sheet.with_suffix(".png")
                render_contact_sheet(src, tmp_png, max_frames=MAX_FRAMES, cols=COLS)
                img = cv2.imread(str(tmp_png))
                cv2.imwrite(str(sheet), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                tmp_png.unlink()
                rendered += 1
            except (OSError, ValueError, cv2.error) as e:  # unreadable clip: report, keep going
                missing.append(f"{clip_id} ({e})")
                continue
        manifest.append({"clip_id": clip_id, "sheet": sheet.name})
        print(f"\r{i}/{len(wanted)} clips processed", end="", flush=True)
    print()

    if missing:
        print(f"\nWARNING: {len(missing)} clips had no readable local media:")
        for m in missing:
            print(f"  - {m}")

    manifest_path = out_dir / "sheets_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "sheet"])
        w.writeheader()
        w.writerows(manifest)

    stamp = time.strftime("%Y%m%d")
    zip_path = root.parent / f"asl-fluency-ai-sheets-{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(manifest_path, arcname="sheets_manifest.csv")
        for row in manifest:
            zf.write(out_dir / row["sheet"], arcname=f"sheets/{row['sheet']}")

    print(f"\nwrote {zip_path}  ({human(zip_path.stat().st_size)})")
    print(f"  sheets: {len(manifest)}  (rendered {rendered}, reused {skipped}, "
          f"missing {len(missing)})")
    print("\nUpload the zip. Ask for: 1-5 fluency ratings per the intake rubric,")
    print("one sentence rationale each, long-format CSV, rater_id = vlm_kimi.")
    return 0 if manifest else 1


if __name__ == "__main__":
    raise SystemExit(main())
