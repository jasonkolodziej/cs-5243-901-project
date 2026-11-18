"""Project webpage: recompute headline results from committed data and render HTML.

`asl-fluency site --out _site` is what the GitHub Pages workflow runs on every
push that touches ratings, features or code, so the published numbers can never
drift from the repository. The page frames itself from the data:

    pilot          fewer than 2 human raters with ratings -> "single-rater pilot"
    low_agreement  >= 2 raters but ICC(2,k) below the threshold -> results shown
                   with a reliability warning, no consensus claim
    consensus      >= 2 raters and ICC(2,k) >= threshold -> final framing

Only aggregate statistics are published — never per-clip or per-rater scores.
"""

from __future__ import annotations

import html
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .data.fluency_labels import consensus_scores, split_ai_raters
from .evaluation.ablation import (
    BASELINE_ARM,
    DEFAULT_ARMS,
    PRIMARY_ARM,
    marginal_stratified_metrics,
    run_ablation,
)
from .evaluation.agreement import agreement_report

ICC_THRESHOLD = 0.6
ALPHA = 0.05

ARM_LABELS = {
    "mean_baseline": ("Mean baseline", "—"),
    "naive_speed_only": ("Speed-only baseline", "1"),
    "core_ridge_cv": ("Ridge-CV, core (primary)", "3"),
    "core_ordinal": ("Ordinal, core", "3"),
    "kinematic_robust_ridge_cv": ("Ridge-CV, robust kinematic", "16"),
    "kinematic_ridge": ("Ridge α = 1 (legacy)", "18"),
    "kinematic_robust_ridge": ("Ridge α = 1, robust (legacy)", "16"),
    "kinematic_gbm": ("Gradient boosting (legacy)", "18"),
}


def framing(n_active_raters: int, icc: float | None) -> str:
    """Which story the page tells: pilot | low_agreement | consensus."""
    if n_active_raters < 2 or icc is None or math.isnan(icc):
        return "pilot"
    return "consensus" if icc >= ICC_THRESHOLD else "low_agreement"


def compute_site_results(
    sheet: pd.DataFrame,
    feats: pd.DataFrame,
    *,
    context: pd.DataFrame | None = None,
    boundary: pd.DataFrame | None = None,
    n_splits: int = 5,
    bootstrap_iters: int = 1000,
) -> dict:
    """All numbers shown on the page, as a JSON-serializable dict.

    `sheet` is the long intake sheet (clip_id, rater_id, rating; blanks allowed).
    """
    sheet = sheet.copy()
    sheet["rating"] = pd.to_numeric(sheet["rating"], errors="coerce")
    human_sheet, _ = split_ai_raters(sheet)
    progress = (
        human_sheet.assign(done=human_sheet["rating"].notna())
        .groupby("rater_id")["done"].agg(rated="sum", assigned="size")
        .reset_index()
    )
    rated = human_sheet.dropna(subset=["rating"])
    active = progress[progress["rated"] > 0]
    n_active = len(active)

    agreement = None
    if n_active >= 2:
        agreement = {k: _num(v) for k, v in agreement_report(rated).items()}
    icc = agreement["icc_2k"] if agreement else None
    mode = framing(n_active, icc)

    y = consensus_scores(rated)
    labeled = feats[feats["clip_id"].isin(y.index)].reset_index(drop=True)
    results: dict = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "framing": mode,
        "icc_threshold": ICC_THRESHOLD,
        "raters": progress.to_dict("records"),
        "n_active_raters": n_active,
        "n_rated_clips": len(labeled),
        "n_signers": int(labeled["signer_id"].nunique()) if len(labeled) else 0,
        "n_feature_clips": len(feats),
        "rating_distribution": {
            str(int(k)): int(v) for k, v in y.round().value_counts().sort_index().items()
        },
        "rating_mean": _num(y.mean()),
        "rating_sd": _num(y.std()),
        "agreement": agreement,
        "ablation": [],
        "hypothesis": None,
        "fairness": [],
        "boundary": [],
        "per_rater": [],
    }

    n_groups = labeled["signer_id"].nunique() if len(labeled) else 0
    if len(labeled) >= 10 and n_groups >= 2:
        summary, details = run_ablation(
            labeled, y, group_col="signer_id", n_splits=min(n_splits, n_groups),
            bootstrap_iters=bootstrap_iters,
        )
        results["ablation"] = [_ablation_row(r) for r in summary.to_dict("records")]
        primary = next((r for r in results["ablation"] if r["arm"] == PRIMARY_ARM), None)
        if primary:
            results["hypothesis"] = {
                "arm": PRIMARY_ARM, "baseline": BASELINE_ARM,
                "delta": primary["delta_spearman"], "ci": primary["delta_ci"],
                "p": primary["p_vs_naive"],
                "significant": bool(primary["p_vs_naive"] is not None
                                    and primary["p_vs_naive"] < ALPHA),
                # Pre-registered: no consensus claim unless ICC(2,k) >= threshold.
                "supported": bool(mode == "consensus" and primary["p_vs_naive"] is not None
                                  and primary["p_vs_naive"] < ALPHA),
            }
        if context is not None and len(context):
            results["fairness"] = _fairness(y, details[PRIMARY_ARM]["oof"], context, labeled)
        if n_active >= 2:
            results["per_rater"] = _per_rater(
                rated, feats, n_splits=n_splits, bootstrap_iters=bootstrap_iters
            )

    if boundary is not None and len(boundary):
        results["boundary"] = [
            {"feature": r["feature"], "min_rho": _num(r["min_rho"])}
            for r in boundary.sort_values("min_rho").to_dict("records")
        ]
    return results


def _per_rater(rated, feats, *, n_splits, bootstrap_iters) -> list[dict]:
    """Primary vs. baseline against each rater alone (the low-ICC fallback)."""
    arms = [a for a in DEFAULT_ARMS if a["name"] in (BASELINE_ARM, PRIMARY_ARM)]
    rows = []
    for rater_id, g in rated.groupby("rater_id"):
        y = consensus_scores(g)
        lab = feats[feats["clip_id"].isin(y.index)].reset_index(drop=True)
        n_groups = lab["signer_id"].nunique()
        if len(lab) < 10 or n_groups < 2:
            continue
        summary, _ = run_ablation(lab, y, arms=arms, group_col="signer_id",
                                  n_splits=min(n_splits, n_groups),
                                  bootstrap_iters=bootstrap_iters)
        by = {r["arm"]: _ablation_row(r) for r in summary.to_dict("records")}
        rows.append({
            "rater_id": rater_id,
            "mean_rating": _num(y.mean()),
            "baseline_spearman": by[BASELINE_ARM]["spearman"],
            "primary_spearman": by[PRIMARY_ARM]["spearman"],
            "primary_ci": by[PRIMARY_ARM]["spearman_ci"],
            "delta": by[PRIMARY_ARM]["delta_spearman"],
            "p": by[PRIMARY_ARM]["p_vs_naive"],
        })
    return rows


def _fairness(y, oof, context, labeled) -> list[dict]:
    strata = marginal_stratified_metrics(y, oof, context, ["skin_tone", "lighting"])
    det = context.merge(labeled[["clip_id", "detection_rate"]], on="clip_id")
    rows = []
    for r in strata.to_dict("records"):
        col = r["factor"]
        missing = pd.isna(r["stratum"]) or r["stratum"] == "nan"
        sel = det[det[col].isna()] if missing else det[det[col].astype(str) == r["stratum"]]
        rows.append({
            "factor": col,
            "stratum": "undefined" if missing else str(r["stratum"]),
            "n": int(r["n"]),
            "detection_rate": _num(sel["detection_rate"].mean()),
            "mean_rating": _num(y.reindex(sel["clip_id"]).mean()),
            "bias": _num(r["mean_signed_error"]),
        })
    return rows


def _ablation_row(r: dict) -> dict:
    return {
        "arm": r["arm"],
        "spearman": _num(r["spearman"]),
        "spearman_ci": [_num(v) for v in r["spearman_ci"]],
        "mae": _num(r["mae"]),
        "delta_spearman": _num(r.get("delta_spearman")),
        "delta_ci": [_num(v) for v in r.get("delta_ci", (np.nan, np.nan))],
        "p_vs_naive": _num(r.get("p_vs_naive")),
    }


def _num(v) -> float | None:
    """JSON-safe float (NaN/None -> None)."""
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) else v


# --------------------------------------------------------------------------- render


CSS = """
:root{--bg:#F7F3E8;--ink:#27231F;--accent:#355C52;--muted:#8A8276;--warn:#8C3B36;
--rule:#C9C2B2;--dark:#22403A;--gold:#C9A869}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.6 Arial,Helvetica,sans-serif}main{max-width:960px;margin:auto;padding:0 24px 64px}
header.hero{background:var(--dark);color:var(--bg);padding:64px 24px 48px}
header.hero>div{max-width:960px;margin:auto}.kicker{letter-spacing:.15em;font-size:12px;
font-weight:700;text-transform:uppercase;color:#C9D4C4}h1{font-size:40px;line-height:1.15;margin:12px 0}
.authors{color:#DCE3DC}.links a{display:inline-block;margin:16px 12px 0 0;padding:8px 16px;
border:1px solid var(--gold);color:var(--gold);text-decoration:none;border-radius:4px}
.links a.off{opacity:.45;pointer-events:none}h2{color:var(--accent);font-size:13px;letter-spacing:.15em;
text-transform:uppercase;border-bottom:1px solid var(--rule);padding-bottom:6px;margin-top:48px}
h3{margin:24px 0 8px}.banner{margin:32px 0 0;padding:14px 18px;border-left:4px solid var(--gold);
background:#fff8e6}.banner.consensus{border-color:var(--accent);background:#eef3ef}
.banner.low_agreement{border-color:var(--warn);background:#f8ecea}
table{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0}th{text-align:left;
color:var(--accent);font-size:12px}th,td{padding:6px 8px;border-bottom:1px solid var(--rule)}
tr.primary td{font-weight:700}td.num{font-variant-numeric:tabular-nums}.bad{color:var(--warn)}
.muted{color:var(--muted);font-size:13px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}
.big{font-size:44px;font-weight:700;color:var(--accent)}code,pre{background:#efe9da;border-radius:3px}
pre{padding:12px;overflow-x:auto;font-size:13px}code{padding:1px 4px}img{max-width:100%}
ol.pipe{padding-left:20px}@media(max-width:700px){.grid{grid-template-columns:1fr}h1{font-size:30px}}
"""


def render_site(results: dict, out_dir: str | Path, meta: dict) -> Path:
    """Write index.html, results.json and figures into out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    figs = _figures(results, out)
    (out / "index.html").write_text(_html(results, meta, figs))
    (out / ".nojekyll").write_text("")
    return out / "index.html"


def _figures(results: dict, out: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "axes.spines.top": False, "axes.spines.right": False})
    figs = {}
    rows = [r for r in results["ablation"] if r["spearman"] is not None]
    if rows:
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(rows) + 0.8))
        names = [ARM_LABELS.get(r["arm"], (r["arm"],))[0] for r in rows]
        rho = [r["spearman"] for r in rows]
        lo = [r["spearman"] - (r["spearman_ci"][0] or r["spearman"]) for r in rows]
        hi = [(r["spearman_ci"][1] or r["spearman"]) - r["spearman"] for r in rows]
        colors = ["#355C52" if r["arm"] == PRIMARY_ARM else
                  "#8A8276" if r["arm"] == BASELINE_ARM else "#9FB3A3" for r in rows]
        ax.barh(names, rho, xerr=[lo, hi], color=colors, ecolor="#27231F", capsize=3)
        ax.axvline(0, color="#27231F", lw=0.8)
        ax.invert_yaxis()
        ax.set_xlabel("Out-of-fold Spearman ρ with human ratings (95% bootstrap CI)")
        fig.tight_layout()
        fig.savefig(out / "ablation.svg", transparent=True)
        plt.close(fig)
        figs["ablation"] = "ablation.svg"
    dist = results["rating_distribution"]
    if dist:
        fig, ax = plt.subplots(figsize=(4.2, 2.6))
        ks = [str(k) for k in range(1, 6)]
        vals = [dist.get(k, 0) for k in ks]
        ax.bar(ks, vals, color="#355C52")
        ax.set_xlabel("Fluency rating (consensus, rounded)")
        ax.set_ylabel("Clips")
        fig.tight_layout()
        fig.savefig(out / "ratings.svg", transparent=True)
        plt.close(fig)
        figs["ratings"] = "ratings.svg"
    return figs


def _f(v, fmt="{:.2f}", signed=False) -> str:
    if v is None:
        return "—"
    s = fmt.format(v)
    return ("+" + s) if signed and v > 0 else s.replace("-", "−")


def _p(v) -> str:
    return "—" if v is None else ("< .001" if v < 0.001 else f"{v:.3f}".lstrip("0"))


def _html(r: dict, meta: dict, figs: dict) -> str:
    e = html.escape
    mode = r["framing"]
    total = len(r["raters"])
    ag = r["agreement"] or {}
    icc = ag.get("icc_2k")
    banner = {
        "pilot": f"<strong>Preliminary — single-rater pilot</strong> ({r['n_active_raters']} of "
                 f"{total} raters have submitted ratings). Every number on this page is recomputed "
                 "automatically from the repository whenever new ratings are committed.",
        "low_agreement": f"<strong>Ratings complete, low agreement</strong> — ICC(2,k) = "
                         f"{_f(icc)} is below {r['icc_threshold']}, so the consensus is unreliable; "
                         "results are shown for transparency, not as a confirmed finding.",
        "consensus": f"<strong>Final — consensus of {r['n_active_raters']} raters</strong>, "
                     f"ICC(2,k) = {_f(icc)} (≥ {r['icc_threshold']}).",
    }[mode]
    scope = {
        "pilot": "single-rater pilot",
        "low_agreement": f"mean of {r['n_active_raters']} raters, low agreement — exploratory",
        "consensus": f"consensus of {r['n_active_raters']} raters",
    }[mode]

    h = r["hypothesis"]
    if h is None:
        verdict = "Not enough rated clips yet to run the hypothesis test."
    else:
        stats_txt = (f"Δρ = {_f(h['delta'], signed=True)}, 95% CI [{_f(h['ci'][0], signed=True)}, "
                     f"{_f(h['ci'][1], signed=True)}], one-sided p = {_p(h['p'])}")
        if mode == "low_agreement":
            sig = "significant" if h.get("significant") else "not significant"
            verdict = ("<strong>No consensus verdict (pre-registered):</strong> raters disagree "
                       f"(ICC(2,k) = {_f(icc)} &lt; {r['icc_threshold']}), so the consensus test "
                       f"is reported as exploratory only — {stats_txt}, {sig} at α = .05. "
                       "See per-rater results below.")
        elif h["supported"]:
            verdict = ("<strong>Supported:</strong> the kinematic model outperforms the speed-only "
                       f"baseline ({stats_txt}).")
        elif h["delta"] is not None and h["delta"] > 0:
            verdict = ("<strong>Directionally consistent, not yet significant:</strong> the "
                       f"kinematic model beats the speed-only baseline in the predicted direction "
                       f"({stats_txt}), but not at α = .05.")
        else:
            verdict = ("<strong>Not supported:</strong> the kinematic model does not outperform "
                       f"the speed-only baseline ({stats_txt}).")

    rater_rows = "".join(
        f"<tr><td>{e(x['rater_id'])}</td><td class=num>{x['rated']} / {x['assigned']}</td></tr>"
        for x in r["raters"]
    )
    agree_txt = (f"Fleiss' κ = {_f(ag.get('fleiss_kappa'))}, ICC(2,k) = {_f(icc)}"
                 if ag else "Agreement needs ≥ 2 raters — pending.")

    abl_rows = ""
    for a in r["ablation"]:
        name, nfeat = ARM_LABELS.get(a["arm"], (a["arm"], "?"))
        ci = (f" [{_f(a['spearman_ci'][0])}, {_f(a['spearman_ci'][1])}]"
              if a["spearman"] is not None else "")
        floor = next((x["mae"] for x in r["ablation"] if x["arm"] == "mean_baseline"), None)
        bad = " bad" if floor is not None and a["mae"] is not None and a["mae"] > floor + 0.02 else ""
        delta = ("—" if a["arm"] in ("mean_baseline", BASELINE_ARM) or a["delta_spearman"] is None
                 else f"{_f(a['delta_spearman'], signed=True)} (p = {_p(a['p_vs_naive'])})")
        cls = " class=primary" if a["arm"] == PRIMARY_ARM else ""
        abl_rows += (f"<tr{cls}><td>{e(name)}</td><td class=num>{nfeat}</td>"
                     f"<td class=num>{_f(a['spearman'])}{ci}</td>"
                     f"<td class='num{bad}'>{_f(a['mae'], '{:.3f}')}</td>"
                     f"<td class=num>{delta}</td></tr>")

    fair_rows = "".join(
        f"<tr><td>{e(x['factor'].replace('_', ' '))} — {e(x['stratum'])}</td>"
        f"<td class=num>{x['n']}</td><td class=num>{_f(x['detection_rate'], '{:.1%}')}</td>"
        f"<td class=num>{_f(x['mean_rating'])}</td><td class=num>{_f(x['bias'], signed=True)}</td></tr>"
        for x in r["fairness"]
    )
    per_rater_rows = "".join(
        f"<tr><td>{e(x['rater_id'])}</td><td class=num>{_f(x['mean_rating'])}</td>"
        f"<td class=num>{_f(x['baseline_spearman'])}</td>"
        f"<td class=num>{_f(x['primary_spearman'])} [{_f(x['primary_ci'][0])}, "
        f"{_f(x['primary_ci'][1])}]</td>"
        f"<td class=num>{_f(x['delta'], signed=True)} (p = {_p(x['p'])})</td></tr>"
        for x in r["per_rater"]
    )
    per_rater_html = "" if not per_rater_rows else f"""
<h2>Results — per rater</h2>
<table><tr><th>Rater</th><th>Mean rating</th><th>Speed-only ρ</th>
<th>Primary model ρ [95% CI]</th><th>Δρ vs. speed-only (p)</th></tr>{per_rater_rows}</table>
<p class=muted>Primary model (ridge-CV, core features) and speed-only baseline evaluated against
each rater alone, signer-disjoint CV. Pre-registered fallback when raters disagree.</p>"""
    bnd_rows = "".join(
        f"<tr><td><code>{e(x['feature'])}</code></td><td class=num>{_f(x['min_rho'])}</td></tr>"
        for x in r["boundary"]
    )

    def link(label, url):
        return (f"<a href='{e(url)}'>{e(label)}</a>" if url
                else f"<a class=off>{e(label)} (coming soon)</a>")

    links = "".join([
        link("Code repository", meta.get("repo_url")),
        link("Presentation slides", meta.get("slides_url")),
        link("Presentation video", meta.get("video_url")),
        link("Analysis notebook", meta.get("notebook_url")),
    ])
    sha = meta.get("commit", "")
    ratings_fig = (f"<img src='{figs['ratings']}' alt='Rating distribution'>"
                   if "ratings" in figs else "")
    ablation_fig = (f"<img src='{figs['ablation']}' alt='Ablation: Spearman by model'>"
                    if "ablation" in figs else "")

    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>{e(meta['title'])}</title><meta name=description content="{e(meta['tagline'])}">
<style>{CSS}</style></head><body>
<header class=hero><div><div class=kicker>{e(meta['course'])} · Final Project</div>
<h1>{e(meta['title'])}</h1><div class=authors>{e(' · '.join(meta['authors']))}</div>
<div class=links>{links}</div></div></header><main>
<div class="banner {mode}">{banner}</div>

<h2>Abstract</h2>
<p>ASL fingerspelling fluency — speed, smoothness and consistency of letter transitions — is
judged by eye today. We score <em>how well</em>, not <em>what</em>, someone fingerspells: MediaPipe
hand landmarks → letter segmentation → kinematic fluency features → small, pre-registered models
evaluated against human 1–5 ratings with signer-disjoint cross-validation, on {r['n_feature_clips']:,}
in-the-wild ChicagoFSWild clips. Hypothesis: kinematics predict human-perceived fluency better than
signing speed alone.</p>
<p><strong>Result ({scope}, n = {r['n_rated_clips']} clips, {r['n_signers']} signers):</strong> {verdict}</p>

<h2>Motivation</h2>
<p>Prior computer-vision work on fingerspelling targets <em>recognition</em> — transcribing letters,
e.g. on ChicagoFSWild. None of it labels or predicts fluency, and no public dataset has fluency
ratings. Instructors assess proficiency subjectively; an objective, automatic score would support
feedback for learners at scale.</p>
<p><strong>Research question:</strong> can kinematic features from hand-landmark tracking — signing
rate, transition smoothness, pause patterns — predict human-perceived fingerspelling fluency?</p>

<h2>Method</h2>
<ol class=pipe>
<li><strong>Landmarks</strong> — MediaPipe Hands, 21 keypoints per frame; the most active hand is the signing hand.</li>
<li><strong>Segmentation</strong> — letter boundaries from the transcript (uniform / energy-valley).</li>
<li><strong>Features</strong> — pre-registered core set: letters per second, log dimensionless jerk
(smoothness), total pause time; 14 further kinematic features as ablations.</li>
<li><strong>Models</strong> — mean and speed-only baselines; ridge with nested-CV shrinkage and an
ordinal (proportional-odds) model on the core set; legacy 18-feature ridge / gradient boosting.</li>
<li><strong>Evaluation</strong> — signer-disjoint CV, Spearman / MAE with bootstrap CIs, paired-bootstrap
Δρ vs. the speed-only baseline, inter-rater agreement, stratified fairness metrics.</li>
</ol>

<h2>Data &amp; ground truth</h2>
<div class=grid><div>
<p>{r['n_rated_clips']} stratified clips from {r['n_signers']} signers, rated 1–5 for fluency in a
browser UI. Mean rating {_f(r['rating_mean'])} (SD {_f(r['rating_sd'])}).</p>
<table><tr><th>Rater</th><th>Clips rated</th></tr>{rater_rows}</table>
<p class=muted>{agree_txt}</p></div><div>{ratings_fig}</div></div>

<h2>Results — ablation ({scope})</h2>
<table><tr><th>Model</th><th>Features</th><th>Spearman ρ [95% CI]</th><th>MAE</th>
<th>Δρ vs. speed-only (p)</th></tr>{abl_rows}</table>
{ablation_fig}
<p class=muted>Out-of-fold predictions, signer-disjoint CV. Δρ: paired bootstrap of
Spearman(model) − Spearman(speed-only) on the same clips, one-sided. MAE in red is worse than
predicting the mean. The primary arm was fixed before ratings were collected.</p>

{per_rater_html}

<h2>Fairness (skin tone &amp; lighting)</h2>
<table><tr><th>Stratum</th><th>Clips</th><th>Hand detection</th><th>Mean rating</th>
<th>Model bias (pred − true)</th></tr>{fair_rows}</table>
<p class=muted>Skin-tone proxy: Individual Typology Angle of palm pixels located by the landmarks;
lighting: frame-brightness tertiles. A proxy confounded with capture conditions, not a demographic label.</p>

<h2>Robustness to letter-boundary error</h2>
<table><tr><th>Feature</th><th>Min ρ under ±2/4/6-frame jitter</th></tr>{bnd_rows}</table>

<h2>Limitations &amp; future work</h2>
<ul><li>Small, restricted-range rated set (mostly fluent signers) caps power and correlation.</li>
<li>Uniform segmentation; a CTC forced aligner on landmarks would make transition features trustworthy.</li>
<li>ITA skin-tone proxy from compressed web video; calibrated annotation would strengthen fairness claims.</li>
<li>Next: more raters and adaptive rating batches; a learned pose encoder pretrained on ChicagoFSWild+.</li></ul>

<h2>Reproduce</h2>
<pre>cd asl-fluency && uv sync
uv run asl-fluency agreement data/ratings/ratings_intake.csv
uv run asl-fluency evaluate  data/ratings/ratings_intake.csv
uv run asl-fluency site --out _site      # this page</pre>

<p class=muted>Generated {e(r['generated_at'])}{' from commit <code>' + e(sha[:7]) + '</code>' if sha else ''}
· raw numbers: <a href=results.json>results.json</a></p>
</main></body></html>"""
