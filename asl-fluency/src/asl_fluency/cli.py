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
    lm_cfg = dict(cfg["landmarks"])
    model_path = _project_path(lm_cfg.pop("model_path", None), cfg)
    lm_cfg.pop("backend", None)
    lm_cfg.pop("static_image_mode", None)
    saved = batch_extract(ClipIndex(df), out_dir, model_path=model_path, **lm_cfg)
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


@app.command()
def demo(
    video: Path = typer.Argument(..., help="single fingerspelling video file"),
    transcript: str = typer.Option(..., help="the spelled word, e.g. INTERDISCIPLINARY"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    annotated_out: Path | None = typer.Option(None, help="save skeleton-overlay mp4"),
) -> None:
    """Full single-clip report: landmarks -> segmentation -> fluency features.

    The quick way to test the pipeline on any video file (e.g. one reassembled
    from ChicagoFSWild frames) without building the whole corpus index.
    """
    from .features.pipeline import compute_feature_row
    from .landmarks.extractor import extract_landmarks

    cfg = load_config(config)
    lm_cfg = dict(cfg["landmarks"])
    model_path = _project_path(lm_cfg.pop("model_path", None), cfg)
    lm_cfg.pop("backend", None)
    lm_cfg.pop("static_image_mode", None)

    seq = extract_landmarks(video, video.stem, model_path=model_path, **lm_cfg)
    row = compute_feature_row(
        seq, transcript,
        seg_method=cfg["segmentation"]["method"],
        refine_radius=cfg["segmentation"]["refine_radius_frames"],
        smooth_window=cfg["features"]["smooth_window_frames"],
        smooth_polyorder=cfg["features"]["smooth_polyorder"],
        pause_speed_percentile=cfg["features"]["pause_speed_percentile"],
        min_pause_frames=cfg["features"]["min_pause_frames"],
    )
    typer.echo(f"clip: {video.name}  transcript: {transcript.upper()}")
    typer.echo(f"frames: {seq.n_frames} @ {seq.fps:.1f} fps "
               f"(detected hand in {seq.detection_rate * 100:.0f}%)")
    for k, v in row.items():
        if k not in ("clip_id", "n_letters", "detection_rate"):
            typer.echo(f"  {k:22s} {v:.4f}" if isinstance(v, float) else f"  {k:22s} {v}")

    if annotated_out:
        _render_annotated(video, seq, annotated_out)
        typer.echo(f"annotated video -> {annotated_out}")


@app.command()
def live(
    source: str = typer.Option("0", help="webcam index ('0') or video file path"),
    transcript: str | None = typer.Option(None, help="word being spelled"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    display: bool = typer.Option(True, "--display/--no-display"),
    save_video: Path | None = typer.Option(None, help="save annotated stream mp4"),
    report: bool = typer.Option(True, help="print full feature report at the end"),
) -> None:
    """Live fluency monitoring: webcam (default) or a video file as the source.

    Shows/saves a skeleton overlay with live wrist-speed, detection-rate, and
    letters-per-second HUD; on exit, runs the full offline feature pipeline
    over the captured stream.
    """
    from .landmarks.live import LiveFluencyMonitor, run_live

    cfg = load_config(config)
    lm_cfg = cfg["landmarks"]
    monitor = LiveFluencyMonitor(
        model_path=_project_path(lm_cfg.get("model_path"), cfg),
        transcript=transcript,
        max_num_hands=lm_cfg.get("max_num_hands", 2),
        min_detection_confidence=lm_cfg.get("min_detection_confidence", 0.5),
        min_tracking_confidence=lm_cfg.get("min_tracking_confidence", 0.5),
    )
    src: int | str = int(source) if source.isdigit() else source
    seq = run_live(src, monitor, display=display, save_video=save_video)
    typer.echo(f"captured {seq.n_frames} frames @ {seq.fps:.1f} fps, "
               f"detection rate {seq.detection_rate * 100:.0f}%")

    if report and transcript:
        from .features.pipeline import compute_feature_row

        row = compute_feature_row(
            seq, transcript,
            seg_method=cfg["segmentation"]["method"],
            refine_radius=cfg["segmentation"]["refine_radius_frames"],
            smooth_window=cfg["features"]["smooth_window_frames"],
            smooth_polyorder=cfg["features"]["smooth_polyorder"],
            pause_speed_percentile=cfg["features"]["pause_speed_percentile"],
            min_pause_frames=cfg["features"]["min_pause_frames"],
        )
        for k, v in row.items():
            if k not in ("clip_id", "n_letters", "detection_rate"):
                typer.echo(f"  {k:22s} {v:.4f}" if isinstance(v, float) else f"  {k:22s} {v}")


def _project_path(p: str | None, cfg: dict) -> Path:
    """Resolve a config path relative to the project root."""
    from .config import DEFAULT_CONFIG_PATH

    root = DEFAULT_CONFIG_PATH.parents[1]
    return root / p if p else root / "assets" / "hand_landmarker.task"


def _render_annotated(video: Path, seq, out: Path) -> None:
    import cv2

    from .landmarks.extractor import draw_landmarks, open_frame_source

    cap, _ = open_frame_source(video)
    writer = None
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok or i >= seq.n_frames:
            break
        if writer is None:
            h, w = bgr.shape[:2]
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                     seq.fps or 30.0, (w, h))
        writer.write(draw_landmarks(bgr, seq.landmarks[i] if seq.frame_ok[i] else None))
        i += 1
    cap.release()
    if writer:
        writer.release()


if __name__ == "__main__":
    app()
