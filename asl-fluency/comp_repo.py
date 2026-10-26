#!/usr/bin/env python3
"""
comp_repo.py — package the asl-fluency repo into a compressed archive
suitable for uploading (or any file-upload gate with size limits).

Included by default:
  - all source code (src/, tests/), configs/, notebooks/
  - README.md, TREE.md, pyproject.toml, uv.lock
  - data CSVs: clip_index.csv, features.csv, ablation/stratified results,
    everything under data/ratings/  (small, and they ARE the project state)
  - .gitkeep placeholders, current git commit hash (if available)

Excluded by default (heavy or regenerable):
  - .git/, .venv/, __pycache__/, tool caches
  - data/raw/ frame directories, videos, and dataset archives (.tgz/.tar.gz)
  - data/interim/landmarks/*.npz   (re-derivable via `asl-fluency landmarks`)
  - assets/hand_landmarker.task    (~8 MB; re-downloadable, README has the curl)

Usage:
  python comp_repo.py                          # -> asl-fluency-share.zip
  python comp_repo.py --format tar.gz
  python comp_repo.py --include-asset          # also pack hand_landmarker.task
  python comp_repo.py --root /path/to/asl-fluency --out /tmp/share.zip
"""

import argparse
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

# --- policy -----------------------------------------------------------------

EXCLUDE_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", ".idea", ".vscode", "node_modules", ".ipynb_checkpoints",
}

# heavy / regenerable file types, anywhere in the tree
EXCLUDE_SUFFIXES = {
    ".pyc", ".pyo",
    ".npz",                      # landmark tensors — re-derivable
    ".jpg", ".jpeg", ".png",     # raw dataset frames
    ".mp4", ".avi", ".mov", ".webm",
    ".tgz", ".zip",              # nested dataset archives
    ".task",                     # MediaPipe model (unless --include-asset)
}

EXCLUDE_NAMES = {".DS_Store", "Thumbs.db"}

WARN_BYTES = 90 * 1024 * 1024    # warn if the archive exceeds ~90 MB


def git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def iter_files(root: Path, include_asset: bool):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        parts = set(rel.parts)
        if parts & EXCLUDE_DIRS:
            continue
        if path.name in EXCLUDE_NAMES:
            continue
        suffix = path.suffix.lower()
        if suffix in {".tar.gz", ".tar.bz2"} or path.name.endswith((".tar.gz", ".tar.bz2")):
            continue
        if suffix == ".task" and include_asset:
            pass  # explicitly allowed
        elif suffix in EXCLUDE_SUFFIXES:
            continue
        yield rel


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def main() -> int:
    ap = argparse.ArgumentParser(description="Package asl-fluency for upload.")
    ap.add_argument("--root", default=".", help="repo root (default: cwd)")
    ap.add_argument("--out", default=None, help="output archive path")
    ap.add_argument("--format", choices=["zip", "tar.gz"], default="zip")
    ap.add_argument("--include-asset", action="store_true",
                    help="also include assets/hand_landmarker.task (~8 MB)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not (root / "pyproject.toml").exists():
        sys.exit(f"error: {root} does not look like the asl-fluency repo root")

    stamp = time.strftime("%Y%m%d-%H%M")
    out = Path(args.out) if args.out else root.parent / f"asl-fluency-share-{stamp}.{args.format}"

    files = list(iter_files(root, args.include_asset))
    if not files:
        sys.exit("error: nothing matched the include policy")

    # manifest stamped into the archive so recipients know what this snapshot is
    manifest = (
        f"asl-fluency share snapshot\n"
        f"created:  {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"git:      {git_commit(root)}\n"
        f"files:    {len(files)}\n"
        f"excluded: .git, .venv, caches, raw frames/video, *.npz landmarks"
        f"{'' if args.include_asset else ', hand_landmarker.task'}\n"
    )

    if args.format == "zip":
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            zf.writestr("asl-fluency/SHARE-MANIFEST.txt", manifest)
            for rel in files:
                zf.write(root / rel, arcname=f"asl-fluency/{rel.as_posix()}")
    else:
        with tarfile.open(out, "w:gz") as tf:
            import io
            data = manifest.encode()
            info = tarfile.TarInfo("asl-fluency/SHARE-MANIFEST.txt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
            for rel in files:
                tf.add(root / rel, arcname=f"asl-fluency/{rel.as_posix()}")

    # --- report ---------------------------------------------------------------
    sizes = sorted(((root / r).stat().st_size, r) for r in files)
    raw_total = sum(s for s, _ in sizes)
    arc_size = out.stat().st_size

    print(f"\nwrote {out}")
    print(f"  files:        {len(files)}")
    print(f"  uncompressed: {human(raw_total)}")
    print(f"  archive:      {human(arc_size)}")
    print("\n  largest entries:")
    for s, r in sizes[-8:][::-1]:
        print(f"    {human(s):>10}  {r.as_posix()}")

    if arc_size > WARN_BYTES:
        print(f"\n  WARNING: archive exceeds {human(WARN_BYTES)}.")
        print("  If the upload is rejected, drop bulky CSVs by adding them to")
        print("  EXCLUDE_SUFFIXES / a name check, or share features.csv separately.")
        return 1
    print("\n  OK — within typical upload limits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
