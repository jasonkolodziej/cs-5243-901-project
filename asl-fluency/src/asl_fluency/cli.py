"""Command-line interface: the full experiment pipeline as subcommands.

    asl-fluency index        build the clip index from a downloaded corpus
    asl-fluency landmarks    extract MediaPipe landmarks for all clips
    asl-fluency features     compute the kinematic feature table
    asl-fluency rating-sheet generate an empty sheet for human raters
    asl-fluency agreement    inter-rater agreement on collected ratings
    asl-fluency evaluate     ablation + metrics against human ratings
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from .config import load_config, resolve_path

app = typer.Typer(help="ASL fingerspelling fluency scoring pipeline.")


@app.command()
def index(
    config: Path | None = typer.Option(None, "--config", "-c"),
    dataset: str = typer.Option("chicagofswild"),
    split: bool = typer.Option(True, help="add signer-disjoint train/val/test split"),
) -> None:
    """Build data/interim/clip_index.csv from the raw corpus."""
    from .data.chicago_fs_wild import load_clip_index

    cfg = load_config(config)
    ci = load_clip_index(resolve_path(cfg, "raw_dir"), dataset)
    df = ci.signer_disjoint_split(cfg["project"]["seed"]) if split else ci.frame
    out = resolve_path(cfg, "interim_dir") / "clip_index.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    typer.echo(f"{len(df)} clips, {df['signer_id'].nunique()} signers -> {out}")


@app.command()
def landmarks(
    config: Path | None = typer.Option(None, "--config", "-c"),
    limit: int | None = typer.Option(None, help="process only the first N clips"),
) -> None:
    """Extract per-frame hand landmarks -> data/interim/landmarks/*.npz."""
    from .data.chicago_fs_wild import ClipIndex
    from .landmarks.extractor import batch_extract

    cfg = load_config(config)
    index_path = resolve_path(cfg, "interim_dir") / "clip_index.csv"
    df = pd.read_csv(index_path, dtype={"clip_id": str})
    if limit:
        df = df.head(limit)
    out_dir = resolve_path(cfg, "interim_dir") / "landmarks"
    saved = batch_extract(ClipIndex(df), out_dir, **cfg["landmarks"])
    typer.echo(f"{len(saved)} landmark files in {out_dir}")


@app.command()
def features(
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Compute the kinematic feature table -> data/processed/features.csv."""
    from .features.pipeline import build_feature_table

    cfg = load_config(config)
    idx = pd.read_csv(resolve_path(cfg, "interim_dir") / "clip_index.csv",
                      dtype={"clip_id": str})
    transcripts = dict(zip(idx["clip_id"], idx["transcript"]))
    table = build_feature_table(
        resolve_path(cfg, "interim_dir") / "landmarks",
        transcripts,
        seg_method=cfg["segmentation"]["method"],
        refine_radius=cfg["segmentation"]["refine_radius_frames"],
        smooth_window=cfg["features"]["smooth_window_frames"],
        smooth_polyorder=cfg["features"]["smooth_polyorder"],
        pause_speed_percentile=cfg["features"]["pause_speed_percentile"],
        min_pause_frames=cfg["features"]["min_pause_frames"],
    )
    # carry signer_id + split through for grouped CV
    table = table.merge(idx[["clip_id", "signer_id", "split"]], on="clip_id", how="left")
    out = resolve_path(cfg, "processed_dir") / "features.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    typer.echo(f"{len(table)} rows -> {out}")


@app.command()
def rating_sheet(
    config: Path | None = typer.Option(None, "--config", "-c"),
    raters: str = typer.Option(..., help="comma-separated rater ids"),
    n_clips: int = typer.Option(100, help="how many clips to sample for rating"),
) -> None:
    """Generate the empty intake sheet for human fluency raters."""
    from .data.fluency_labels import rating_sheet_template

    cfg = load_config(config)
    idx = pd.read_csv(resolve_path(cfg, "interim_dir") / "clip_index.csv",
                      dtype={"clip_id": str})
    sampled = idx["clip_id"].sample(min(n_clips, len(idx)),
                                    random_state=cfg["project"]["seed"]).tolist()
    sheet = rating_sheet_template(sampled, [r.strip() for r in raters.split(",")])
    out = resolve_path(cfg, "ratings_dir") / "ratings_intake.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(out, index=False)
    typer.echo(f"{len(sheet)} rows ({len(sampled)} clips x {len(raters.split(','))} raters) -> {out}")


@app.command()
def agreement(
    ratings: Path = typer.Argument(..., help="long-format ratings CSV"),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Ground-truth sanity check: Fleiss' kappa / ICC on collected ratings."""
    from .data.fluency_labels import load_ratings
    from .evaluation.agreement import agreement_report

    cfg = load_config(config)
    scale = tuple(cfg["labels"]["rating_scale"])
    report = agreement_report(load_ratings(ratings, scale))
    for k, v in report.items():
        typer.echo(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")


@app.command()
def evaluate(
    ratings: Path = typer.Argument(..., help="long-format ratings CSV"),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Ablation: naive vs. kinematic arms against consensus human ratings."""
    from .data.fluency_labels import consensus_scores, load_ratings
    from .evaluation.ablation import run_ablation, stratified_metrics

    cfg = load_config(config)
    scale = tuple(cfg["labels"]["rating_scale"])
    y = consensus_scores(load_ratings(ratings, scale))
    feats = pd.read_csv(resolve_path(cfg, "processed_dir") / "features.csv",
                        dtype={"clip_id": str})
    feats = feats[feats["clip_id"].isin(y.index)].reset_index(drop=True)
    summary, details = run_ablation(
        feats, y,
        group_col="signer_id" if cfg["model"]["cv"] == "signer_group" else None,
        n_splits=cfg["model"]["n_splits"],
    )
    typer.echo(summary.to_string(index=False))
    out = resolve_path(cfg, "processed_dir") / "ablation_results.csv"
    summary.to_csv(out, index=False)

    # Stratified fairness evaluation (proposal Section 7), if context exists.
    best = max(details, key=lambda k: details[k]["folds"]["spearman"].mean())
    oof = details[best]["oof"]
    context_path = resolve_path(cfg, "ratings_dir") / "clip_context.csv"
    if context_path.exists():
        ctx = pd.read_csv(context_path, dtype={"clip_id": str})
        strata = stratified_metrics(y, oof, ctx, cfg["evaluation"]["stratify_by"])
        typer.echo("\nStratified metrics:")
        typer.echo(strata.to_string(index=False))
        strata.to_csv(resolve_path(cfg, "processed_dir") / "stratified_results.csv",
                      index=False)
    typer.echo(f"results -> {out}")


if __name__ == "__main__":
    app()
