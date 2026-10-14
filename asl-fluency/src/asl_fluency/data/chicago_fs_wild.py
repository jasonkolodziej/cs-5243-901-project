"""Loaders for the ChicagoFSWild / ChicagoFSWild+ datasets.

Both datasets ship a CSV-style frame-level annotation file. This module
normalises them into a single per-clip DataFrame with the columns the
pipeline expects:

    clip_id, video_path, transcript, signer_id, n_frames, fps, split

ChicagoFSWild:   https://ttic.uchicago.edu/~klivescu/ChicagoFSWild.htm
ChicagoFSWild+:  same page, extended crowdsourced release (55,232 clips).

The datasets must be downloaded manually (see README). Point paths.raw_dir
in the config at the folder that contains them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tarfile
import warnings

import pandas as pd

# Column aliases seen across the two releases / common unpack layouts.
_COLUMN_ALIASES = {
    "filename": "video_path",
    "file": "video_path",
    "video": "video_path",
    "vid_name": "video_path",
    "label": "transcript",
    "label_proc": "transcript",      # ChicagoFSWild.csv
    "text": "transcript",
    "word": "transcript",
    "signer": "signer_id",           # ChicagoFSWild.csv
    "speaker": "signer_id",
    "partition": "split",            # ChicagoFSWild.csv (train/dev/test)
    "number_of_frames": "n_frames",  # ChicagoFSWild.csv
}


@dataclass
class ClipIndex:
    """A per-clip index over a fingerspelling corpus."""

    frame: pd.DataFrame

    def __len__(self) -> int:
        return len(self.frame)

    def signer_disjoint_split(
        self, seed: int = 42, train: float = 0.7, val: float = 0.15
    ) -> pd.DataFrame:
        """Assign train/val/test by *signer* so no signer leaks across splits.

        The proposal's hypothesis is about generalising to new signers, so
        random clip-level splits would overstate performance.
        """
        df = self.frame.copy()
        signers = df["signer_id"].dropna().unique()
        rng = pd.Series(signers).sample(frac=1.0, random_state=seed).to_numpy()
        n = len(rng)
        n_train, n_val = int(train * n), int(val * n)
        split_of = {}
        for i, s in enumerate(rng):
            split_of[s] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
        df["split"] = df["signer_id"].map(split_of).fillna("unassigned")
        return df


def load_clip_index(
    raw_dir: str | Path,
    dataset: str = "chicagofswild",
    base_dir: str | Path | None = None,
) -> ClipIndex:
    """Build the clip index for a locally downloaded corpus.

    Scans raw_dir for the corpus annotation CSV (ChicagoFSWild.csv /
    ChicagoFSWildPlus.csv, or any CSV that contains recognisable
    transcript + video columns) and normalises column names. Clip paths are
    resolved against common frame-extraction layouts (see _resolve_clip_paths);
    note ChicagoFSWild clips are frame *directories*, not video files.

    `base_dir` (usually the project root) makes stored `video_path` values
    relative to that directory when the clip lives under it, so the emitted
    index is portable across machines. Paths outside `base_dir` stay absolute.

    If a ChicagoFSWild frames archive is present locally
    (ChicagoFSWild-Frames.tgz or .tar.gz), it is unpacked automatically.
    """
    raw_dir = Path(raw_dir)
    _maybe_extract_frames_archives(raw_dir)
    candidates = sorted(raw_dir.glob("**/*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No annotation CSV found under {raw_dir}. "
            f"Download {dataset} first (see README 'Datasets' section)."
        )
    df = None
    for cand in candidates:
        trial = pd.read_csv(cand)
        trial = trial.rename(columns={c: _COLUMN_ALIASES[c.lower()] for c in trial.columns
                                      if c.lower() in _COLUMN_ALIASES})
        trial.columns = [c.lower() for c in trial.columns]
        if {"transcript", "video_path"} <= set(trial.columns):
            df = trial
            break
    if df is None:
        raise ValueError(
            f"No CSV under {raw_dir} had recognisable transcript/video columns. "
            f"Tried: {[c.name for c in candidates]}. Add aliases in _COLUMN_ALIASES."
        )
    resolved = _resolve_clip_paths(df, raw_dir, base_dir)
    found = int(resolved["exists"].sum())
    total = int(len(resolved))
    if found == 0:
        raise FileNotFoundError(
            "Found annotation CSV, but no clip frames/video files were found. "
            "Download/extract ChicagoFSWild-Frames.tgz (or equivalent) under data/raw so "
            "paths like 'aslized/<clip>/0001.jpg' exist."
        )
    if found < total:
        missing = total - found
        warnings.warn(
            f"Skipping {missing} clips with missing local media; using {found}/{total} available clips.",
            RuntimeWarning,
            stacklevel=2,
        )
    df = df.loc[resolved["exists"]].copy()
    df["video_path"] = resolved.loc[resolved["exists"], "video_path"].to_numpy()
    df["clip_id"] = df["video_path"].map(lambda p: Path(p).name).astype(str)
    if "signer_id" not in df.columns:
        df["signer_id"] = None
    df["transcript"] = df["transcript"].astype(str).str.upper().str.replace(r"[^A-Z]", "", regex=True)
    keep = ["clip_id", "video_path", "transcript", "signer_id"] + [
        c for c in ("fps", "n_frames", "split") if c in df.columns
    ]
    return ClipIndex(df[keep])


def _maybe_extract_frames_archives(raw_dir: Path) -> None:
    """Unpack known ChicagoFSWild frame archives when present.

    This is intentionally best-effort and idempotent: if no archive exists,
    nothing happens; if the destination appears populated, extraction is
    skipped.
    """
    archive_names = (
        "ChicagoFSWild-Frames.tgz",
        "ChicagoFSWild-Frames.tar.gz",
        "ChicagoFSWildPlus-Frames.tgz",
        "ChicagoFSWildPlus-Frames.tar.gz",
    )
    archives: list[Path] = []
    for name in archive_names:
        archives.extend(sorted(raw_dir.glob(f"**/{name}")))
    for archive in archives:
        _extract_frames_archive_if_needed(archive, raw_dir)


def _extract_frames_archive_if_needed(archive_path: Path, raw_dir: Path) -> None:
    """Safely extract a frame archive to raw_dir if not already unpacked."""
    stem = archive_path.name
    if stem.endswith(".tar.gz"):
        stem = stem[:-7]
    elif stem.endswith(".tgz"):
        stem = stem[:-4]
    expected_root = raw_dir / stem

    if expected_root.is_dir() and any(expected_root.iterdir()):
        return

    extract_to = raw_dir
    with tarfile.open(archive_path, "r:gz") as tar:
        _safe_extract_tar(tar, extract_to)


def _safe_extract_tar(tar: tarfile.TarFile, destination: Path) -> None:
    """Extract tar members while preventing path traversal."""
    destination = destination.resolve()
    for member in tar.getmembers():
        member_path = (destination / member.name).resolve()
        if not str(member_path).startswith(str(destination)):
            raise ValueError(
                f"Unsafe archive member path: {member.name}. "
                "Refusing to extract outside the target directory."
            )
    tar.extractall(path=destination)


def _resolve_clip_paths(
    df: pd.DataFrame, raw_dir: Path, base_dir: str | Path | None = None
) -> pd.DataFrame:
    """Map the CSV's relative `filename` to an existing local path.

    The CSV stores paths like `aslized/elsie_stecker_0001` relative to the
    frames root, but the nested archives get extracted in different layouts.
    Try the common roots in order; fall back to raw_dir/rel (which will fail
    loudly later if the frames really are missing).

    When `base_dir` is given, paths that lie under it are stored as relative
    to `base_dir` so the index stays portable across machines.
    """
    roots = [
        raw_dir,
        raw_dir / "aslized",
        raw_dir / "frames",
        raw_dir / "ChicagoFSWild-Frames",
        raw_dir / "ChicagoFSWildPlus-Frames",
        raw_dir / "ChicagoFSWild",
        raw_dir / "ChicagoFSWild" / "aslized",
        raw_dir / "ChicagoFSWild" / "frames",
        raw_dir / "ChicagoFSWildPlus",
        raw_dir / "ChicagoFSWildPlus" / "aslized",
        raw_dir / "ChicagoFSWildPlus" / "frames",
    ]
    roots = [r for r in roots if r.is_dir()]
    base = Path(base_dir) if base_dir else None

    def _make_relative(p: Path) -> str:
        if base is None:
            return str(p)
        try:
            return str(p.relative_to(base))
        except ValueError:
            return str(p)

    def _relative_variants(rel: str) -> list[Path]:
        p = Path(rel)
        variants = [p]
        if p.parts and p.parts[0].lower() in {"aslized", "frames"}:
            variants.append(Path(*p.parts[1:]))
        # preserve order while removing duplicates
        unique = []
        seen = set()
        for v in variants:
            key = str(v)
            if key not in seen:
                seen.add(key)
                unique.append(v)
        return unique

    def resolve(rel: str) -> tuple[str, bool]:
        rel_variants = _relative_variants(rel)
        for root in roots:
            for rel_path in rel_variants:
                p = root / rel_path
                if p.exists():
                    return _make_relative(p), True
        fallback = raw_dir / rel
        return _make_relative(fallback), fallback.exists()

    resolved = df["video_path"].map(resolve)
    return pd.DataFrame(
        {
            "video_path": resolved.map(lambda t: t[0]),
            "exists": resolved.map(lambda t: t[1]),
        },
        index=df.index,
    )
