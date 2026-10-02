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

import pandas as pd

# Column aliases seen across the two releases / common unpack layouts.
_COLUMN_ALIASES = {
    "filename": "video_path",
    "file": "video_path",
    "video": "video_path",
    "vid_name": "video_path",
    "label": "transcript",
    "text": "transcript",
    "word": "transcript",
    "signer": "signer_id",
    "speaker": "signer_id",
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


def load_clip_index(raw_dir: str | Path, dataset: str = "chicagofswild") -> ClipIndex:
    """Build the clip index for a locally downloaded corpus.

    Scans raw_dir for annotation CSVs (annotation*.csv, *labels*.csv, or any
    csv at the top level) and normalises column names.
    """
    raw_dir = Path(raw_dir)
    candidates = sorted(raw_dir.glob("**/*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No annotation CSV found under {raw_dir}. "
            f"Download {dataset} first (see README 'Datasets' section)."
        )
    df = pd.concat((pd.read_csv(c) for c in candidates), ignore_index=True)
    df = df.rename(columns={c: _COLUMN_ALIASES[c.lower()] for c in df.columns
                            if c.lower() in _COLUMN_ALIASES})
    df.columns = [c.lower() for c in df.columns]
    if "transcript" not in df.columns or "video_path" not in df.columns:
        raise ValueError(
            f"Could not find transcript/video columns in {candidates}; "
            f"got columns {list(df.columns)}. Add an alias in _COLUMN_ALIASES."
        )
    df["video_path"] = df["video_path"].map(lambda p: str(raw_dir / p))
    df["clip_id"] = df.get("clip_id", pd.Series(range(len(df)))).astype(str)
    if "signer_id" not in df.columns:
        df["signer_id"] = None
    df["transcript"] = df["transcript"].astype(str).str.upper().str.replace(r"[^A-Z]", "", regex=True)
    return ClipIndex(df[["clip_id", "video_path", "transcript", "signer_id"]
                        + [c for c in ("fps", "n_frames", "split") if c in df.columns]])
