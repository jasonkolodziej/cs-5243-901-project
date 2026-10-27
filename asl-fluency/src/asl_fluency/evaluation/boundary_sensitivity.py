"""Boundary-error sensitivity analysis (Roadmap Gap 1, step 2).

Answers "how much does letter-boundary error actually hurt my features?"
before any effort goes into a better aligner. Procedure:

1. Segment each clip with the configured method (uniform / energy_valley).
2. Perturb the interior boundaries with uniform random jitter of ±j frames
   (j in e.g. {2, 4, 6}), several repetitions per level.
3. Recompute the kinematic features on the perturbed spans.
4. Rank-correlate (Spearman) each perturbed feature column against its
   unperturbed self across clips.

Features that stay near rho = 1 need no better aligner; features that
collapse tell you the alignment quality you must reach — or that the feature
is an aligner artifact (e.g. letter-duration variance under uniform
segmentation is near-constant by construction, so jitter scrambles it).

Typical finding: whole-clip features (LDJ, pause counts, path efficiency)
are perfectly robust; transition statistics degrade gracefully; per-letter
duration variance is the fragile one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..features.kinematics import kinematic_features
from ..features.pipeline import KINEMATIC_FEATURES
from ..landmarks.extractor import LandmarkSequence
from ..segmentation.aligner import segment_letters


def jitter_spans(
    spans: list[tuple[int, int, str]], jitter: int, rng: np.random.Generator
) -> list[tuple[int, int, str]]:
    """Perturb interior boundaries by a uniform draw in [-jitter, +jitter].

    Endpoints (clip start/end) are fixed; boundaries are re-clamped to keep
    spans ordered and non-empty, mimicking small aligner error.
    """
    if jitter <= 0:
        return list(spans)
    b = [spans[0][0]] + [s[1] for s in spans]
    for i in range(1, len(b) - 1):
        b[i] += int(rng.integers(-jitter, jitter + 1))
    for i in range(1, len(b)):
        b[i] = max(b[i], b[i - 1] + 1)
    return [(b[i], b[i + 1], s[2]) for i, s in enumerate(spans)]


def sensitivity_report(
    landmark_dir: str | Path,
    transcripts: dict[str, str],
    jitters: tuple[int, ...] = (2, 4, 6),
    reps: int = 5,
    seg_method: str = "uniform",
    seed: int = 0,
    limit: int | None = None,
    **feature_kw,
) -> pd.DataFrame:
    """One row per feature: Spearman(jittered vs. unperturbed) per jitter level.

    Columns: feature, plus one `rho_jit<j>` column per jitter level (mean over
    repetitions), and `min_rho` for quick triage.
    """
    paths = sorted(Path(landmark_dir).glob("*.npz"))[:limit]
    base_rows, pert_rows = [], {j: [] for j in jitters}
    for path in paths:
        seq = LandmarkSequence.load(path)
        transcript = transcripts.get(seq.clip_id)
        if not transcript:
            continue
        try:
            spans = segment_letters(seq, transcript, method=seg_method)
        except ValueError:
            continue
        base_rows.append(
            {"clip_id": seq.clip_id, **kinematic_features(seq, spans, **feature_kw)}
        )
        for j in jitters:
            acc: dict[str, list[float]] = {f: [] for f in KINEMATIC_FEATURES}
            for rep in range(reps):
                rng = np.random.default_rng(seed + 1000 * j + rep)
                perturbed = jitter_spans(spans, j, rng)
                row = kinematic_features(seq, perturbed, **feature_kw)
                for f in KINEMATIC_FEATURES:
                    acc[f].append(row[f])
            pert_rows[j].append(
                {"clip_id": seq.clip_id, **{f: float(np.mean(v)) for f, v in acc.items()}}
            )

    base = pd.DataFrame(base_rows).set_index("clip_id")
    # Clips whose unperturbed features contain NaN (tracking too sparse to
    # smooth) would poison the whole column's correlation — drop them first,
    # same cleanliness standard as build_feature_table.
    n_total = len(base)
    base = base.dropna()
    if (dropped := n_total - len(base)) > 0:
        import logging

        logging.getLogger(__name__).warning(
            "excluded %d/%d clips with NaN base features from the sensitivity analysis",
            dropped, n_total,
        )
    out = []
    for f in KINEMATIC_FEATURES:
        row = {"feature": f}
        for j in jitters:
            pj = pd.DataFrame(pert_rows[j]).set_index("clip_id")[f].reindex(base.index)
            mask = base[f].notna() & pj.notna()  # pairwise-complete
            x, y = base[f][mask], pj[mask]
            # A constant column (zero variance after masking) has no defined
            # rank correlation — scipy would emit ConstantInputWarning and
            # return NaN anyway. Record NaN deliberately: this is the artifact
            # signature (e.g. letter_dur_* under uniform segmentation), not a
            # numeric accident.
            rho = (
                stats.spearmanr(x, y).statistic
                if len(x) >= 3 and x.nunique() > 1 and y.nunique() > 1
                else np.nan
            )
            row[f"rho_jit{j}"] = float(rho) if np.isfinite(rho) else np.nan
        vals = [row[f"rho_jit{j}"] for j in jitters]
        row["min_rho"] = float(np.nanmin(vals)) if np.isfinite(vals).any() else np.nan
        out.append(row)
    report = pd.DataFrame(out).sort_values("min_rho").reset_index(drop=True)
    report.attrs["n_clips"] = len(base)
    report.attrs["n_dropped"] = n_total - len(base)
    return report
