"""Command-line interface: the full experiment pipeline as subcommands.

    asl-fluency index        build the clip index from a downloaded corpus
    asl-fluency landmarks    extract MediaPipe landmarks for all clips
    asl-fluency features     compute the kinematic feature table
    asl-fluency ai-rate      AI-judge pre-scores (stratification aid, NOT ground truth)
    asl-fluency rating-sheet stratified intake sheet for human raters (batchable)
    asl-fluency rate         browser UI: watch clips, click 1-5, saves to the sheet
    asl-fluency label-boundaries  browser UI: mark per-letter frame spans
    asl-fluency boundary-sensitivity  how much does boundary error hurt each feature?
    asl-fluency agreement    inter-rater agreement on collected ratings
    asl-fluency clip-context objective skin-tone (ITA) / lighting strata (Section 7)
    asl-fluency evaluate     ablation + metrics against human ratings
    asl-fluency site         project webpage (GitHub Pages) from committed data
    asl-fluency train        fit the chosen arm on all rated data, persist artifact
    asl-fluency score        apply a persisted scorer to new clips / feature rows
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from .config import load_config, project_root, resolve_path

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
    ci = load_clip_index(
        resolve_path(cfg, "raw_dir"), dataset, base_dir=project_root()
    )
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
    # video_path may be stored relative to project root; resolve for extraction
    root = project_root()
    df["video_path"] = df["video_path"].apply(
        lambda p: str(root / p) if not Path(p).is_absolute() else p
    )
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
    if table.empty:
        typer.echo("Warning: no feature rows produced (all clips failed segmentation).")
        out = resolve_path(cfg, "processed_dir") / "features.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        typer.echo(f"0 rows -> {out}")
        return
    # carry signer_id + split through for grouped CV
    table = table.merge(idx[["clip_id", "signer_id", "split"]], on="clip_id", how="left")
    out = resolve_path(cfg, "processed_dir") / "features.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    typer.echo(f"{len(table)} rows -> {out}")


def _load_clip_paths(cfg: dict) -> dict[str, Path]:
    """clip_id -> absolute media path (frame dir or video), from the clip index."""
    idx = pd.read_csv(resolve_path(cfg, "interim_dir") / "clip_index.csv",
                      dtype={"clip_id": str})
    root = project_root()
    return {
        c: (root / p) if not Path(p).is_absolute() else Path(p)
        for c, p in zip(idx["clip_id"], idx["video_path"])
    }


@app.command()
def rating_sheet(
    config: Path | None = typer.Option(None, "--config", "-c"),
    raters: str = typer.Option(..., help="comma-separated rater ids"),
    n_clips: int = typer.Option(50, help="how many clips to sample for this batch"),
    batch: int = typer.Option(1, help="batch number (adaptive collection: 1, 2, ...)"),
    stratify_by: str = typer.Option(
        "auto", help="ai_score | letters_per_second | auto | none"
    ),
    min_detection_rate: float = typer.Option(
        0.8, help="exclude clips with worse hand-tracking than this"
    ),
    max_per_signer: int | None = typer.Option(
        None, help="optional cap on clips per signer in one batch"
    ),
) -> None:
    """Generate/extend the intake sheet with a stratified clip batch.

    Samples from features.csv (never the raw clip index), so every rated clip
    is guaranteed to have model features. Stratification spans the fluency
    range: AI-judge score quintiles when data/ratings/ratings_ai.csv exists,
    otherwise letters-per-second quintiles. Re-running with --batch 2, 3, ...
    adds fresh clips and never re-samples ones already in the sheet — this is
    the adaptive top-up loop: rate a pilot, check agreement/correlation, then
    add a batch only if the confidence intervals are still too wide.
    """
    from .data.fluency_labels import rating_sheet_template, stratified_rating_sample

    cfg = load_config(config)
    seed = cfg["project"]["seed"]
    ratings_dir = resolve_path(cfg, "ratings_dir")
    out = ratings_dir / "ratings_intake.csv"
    rater_ids = [r.strip() for r in raters.split(",")]

    already: set[str] = set()
    if out.exists():
        already = set(pd.read_csv(out, dtype={"clip_id": str})["clip_id"].unique())

    strat_col: str | None
    ai_path = ratings_dir / "ratings_ai.csv"
    feats_path = resolve_path(cfg, "processed_dir") / "features.csv"
    feats = (
        pd.read_csv(feats_path, dtype={"clip_id": str})
        if feats_path.exists()
        else pd.DataFrame()
    )

    if stratify_by == "none":
        strat_col = None
    elif stratify_by == "auto":
        strat_col = "ai_score" if ai_path.exists() else "letters_per_second"
    else:
        strat_col = stratify_by

    if strat_col is None or feats.empty:
        if strat_col is not None:
            typer.echo(
                "Warning: features.csv missing/empty; falling back to a uniform "
                "sample from clip_index.csv. Run `asl-fluency features` first so "
                "rated clips have model features."
            )
        idx = pd.read_csv(resolve_path(cfg, "interim_dir") / "clip_index.csv",
                          dtype={"clip_id": str})
        pool = idx[~idx["clip_id"].isin(already)]
        sampled_df = pool.sample(min(n_clips, len(pool)), random_state=seed + batch)
        sampled = sampled_df["clip_id"].tolist()
        batch_log = sampled_df[["clip_id", "signer_id"]].assign(
            batch=batch, stratum=-1, stratum_score=float("nan")
        )
    else:
        work = feats.copy()
        if strat_col == "ai_score":
            ai = pd.read_csv(ai_path, dtype={"clip_id": str})
            score = ai.groupby("clip_id")["rating"].mean().rename("ai_score")
            work = work.merge(score, on="clip_id", how="left")
        picked = stratified_rating_sample(
            work,
            n_clips,
            seed=seed + batch,
            score_col=strat_col,
            min_detection_rate=min_detection_rate,
            exclude_clip_ids=already,
            max_per_signer=max_per_signer,
        )
        sampled = picked["clip_id"].tolist()
        batch_log = picked.assign(batch=batch)
        typer.echo(
            f"stratified on {strat_col}: "
            + ", ".join(
                f"stratum {s}: {n}" for s, n in picked.groupby("stratum").size().items()
            )
        )

    sheet = rating_sheet_template(sampled, rater_ids)
    if len(sampled) < n_clips:
        typer.echo(
            f"note: only {len(sampled)} of the requested {n_clips} clips were "
            "available (pool exhausted by earlier batches, the detection-rate "
            "gate, or missing AI scores)."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    combined = (
        pd.concat([pd.read_csv(out, dtype={"clip_id": str}), sheet], ignore_index=True)
        if out.exists()
        else sheet
    )
    combined.to_csv(out, index=False)

    log_path = ratings_dir / "rating_batches.csv"
    batch_log.to_csv(
        log_path, mode="a", index=False, header=not log_path.exists()
    )
    typer.echo(
        f"batch {batch}: +{len(sheet)} rows ({len(sampled)} clips x {len(rater_ids)} "
        f"raters); total sheet now {len(combined)} rows -> {out}"
    )
    typer.echo("next: asl-fluency rate   (browser UI to collect the ratings)")


@app.command()
def rate(
    config: Path | None = typer.Option(None, "--config", "-c"),
    ratings: Path | None = typer.Option(
        None, help="intake sheet to fill (default: data/ratings/ratings_intake.csv)"
    ),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8765),
    open_browser: bool = typer.Option(True, "--open/--no-open"),
) -> None:
    """Launch the browser rating UI: watch each clip, click 1-5.

    Serves the clips listed in the intake sheet (frame directories play as a
    looping flipbook, video files as <video>), one rater at a time, in a
    per-rater shuffled order. Every click is saved to the sheet immediately,
    so sessions can be stopped and resumed freely.
    """
    import webbrowser

    from .data.ratings_ui import serve

    cfg = load_config(config)
    ratings_path = ratings or resolve_path(cfg, "ratings_dir") / "ratings_intake.csv"
    if not Path(ratings_path).exists():
        raise typer.BadParameter(
            f"{ratings_path} not found — generate it with `asl-fluency rating-sheet` first."
        )
    sheet = pd.read_csv(ratings_path, dtype={"clip_id": str})
    paths = _load_clip_paths(cfg)
    clips = {c: paths[c] for c in sheet["clip_id"].unique() if c in paths}
    missing = set(sheet["clip_id"].unique()) - set(clips)
    if missing:
        typer.echo(f"Warning: {len(missing)} sheet clips have no media in the index; "
                   "they will be skipped in the UI.")

    scale = tuple(cfg["labels"]["rating_scale"])
    fps = float(cfg["landmarks"].get("default_fps", 30.0))
    server = serve(ratings_path, clips, host=host, port=port,
                   rating_scale=scale, default_fps=fps, seed=cfg["project"]["seed"])
    url = f"http://{host}:{port}/"
    typer.echo(f"Rating UI at {url}  (Ctrl+C to stop; progress is saved per click)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        typer.echo(f"ratings saved in {ratings_path}")


@app.command()
def ai_rate(
    config: Path | None = typer.Option(None, "--config", "-c"),
    backend: str | None = typer.Option(
        None, help="mock | grok — default: ai_judge.backend in the config"
    ),
    model: str | None = typer.Option(
        None, help="VLM model id — default: ai_judge.model in the config"
    ),
    n_frames: int | None = typer.Option(
        None, help="frames sampled per clip — default: ai_judge.n_frames"
    ),
    limit: int | None = typer.Option(None, help="score at most N new clips"),
    rater_id: str = typer.Option("vlm_judge"),
    min_detection_rate: float = typer.Option(
        0.8, help="only score clips with at least this tracking quality"
    ),
) -> None:
    """AI-judge pre-scores over the feature pool -> data/ratings/ratings_ai.csv.

    These scores are an AID for stratified sampling and a secondary analysis
    axis — they are never part of the human ground truth (consensus_scores
    excludes vlm_* raters automatically). Re-runs resume where they stopped.

    Backend settings come from the ai_judge section of the config; CLI flags
    override them. The grok backend needs XAI_API_KEY in .env (see
    example.env) and the openai package (uv add openai).
    """
    from .ai_judge import run_ai_rating
    from .config import ai_judge_config

    cfg = load_config(config)
    judge = ai_judge_config(cfg)
    backend = backend or judge["backend"]
    score_opts = {
        "model": model or judge["model"],
        "n_frames": n_frames or judge["n_frames"],
        "temperature": judge["temperature"],
        "jpeg_quality": judge["jpeg_quality"],
    }
    feats_path = resolve_path(cfg, "processed_dir") / "features.csv"
    if not feats_path.exists():
        raise typer.BadParameter("features.csv not found — run `asl-fluency features` first.")
    feats = pd.read_csv(feats_path, dtype={"clip_id": str})
    pool = feats[feats["detection_rate"] >= min_detection_rate]["clip_id"].unique()
    paths = _load_clip_paths(cfg)
    clip_paths = {c: paths[c] for c in pool if c in paths}

    out = resolve_path(cfg, "ratings_dir") / "ratings_ai.csv"
    df = run_ai_rating(
        clip_paths, out, backend=backend, rater_id=rater_id,
        work_dir=resolve_path(cfg, "interim_dir") / "ai_clips", limit=limit,
        **score_opts,
    )
    typer.echo(f"{len(df)} AI-rated rows (rater '{rater_id}') -> {out}")


@app.command()
def label_boundaries(
    config: Path | None = typer.Option(None, "--config", "-c"),
    clips_csv: Path | None = typer.Option(
        None, "--clips", help="CSV with a clip_id column (default: the intake sheet)"
    ),
    out: Path | None = typer.Option(
        None, help="default: data/ratings/boundary_labels.csv"
    ),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8766),
    open_browser: bool = typer.Option(True, "--open/--no-open"),
) -> None:
    """Browser UI for hand-labeling per-letter frame spans (Gap 1 ground truth).

    Labels the same clips that get fluency ratings (default: the intake
    sheet). Scrub the flipbook, click the joint between two letters to set
    that boundary to the current frame; spans auto-save per change. Start
    from an even split and adjust — much faster than marking from scratch.
    """
    import webbrowser

    from .data.ratings_ui import serve

    cfg = load_config(config)
    ratings_dir = resolve_path(cfg, "ratings_dir")
    clips_csv = clips_csv or ratings_dir / "ratings_intake.csv"
    out = out or ratings_dir / "boundary_labels.csv"
    if not Path(clips_csv).exists():
        raise typer.BadParameter(f"{clips_csv} not found — nothing to label.")
    wanted = pd.read_csv(clips_csv, dtype={"clip_id": str})["clip_id"].unique()

    idx = pd.read_csv(resolve_path(cfg, "interim_dir") / "clip_index.csv",
                      dtype={"clip_id": str})
    transcripts = dict(zip(idx["clip_id"], idx["transcript"]))
    paths = _load_clip_paths(cfg)
    clips = {c: paths[c] for c in wanted if c in paths and c in transcripts}
    missing = set(wanted) - set(clips)
    if missing:
        typer.echo(f"Warning: {len(missing)} clips missing media or transcript; skipped.")

    fps = float(cfg["landmarks"].get("default_fps", 30.0))
    server = serve(None, clips, host=host, port=port, default_fps=fps,
                   mode="boundaries", boundary_path=out, transcripts=transcripts)
    url = f"http://{host}:{port}/"
    typer.echo(f"Boundary-labeling UI at {url}  (Ctrl+C to stop; saves per change)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        typer.echo(f"boundary labels saved in {out}")


@app.command()
def boundary_sensitivity(
    config: Path | None = typer.Option(None, "--config", "-c"),
    jitters: str = typer.Option("2,4,6", help="comma-separated frame jitter levels"),
    reps: int = typer.Option(5, help="jitter repetitions per level (averaged)"),
    seg_method: str | None = typer.Option(None, help="default: config segmentation.method"),
    limit: int | None = typer.Option(None, help="analyze at most N clips"),
) -> None:
    """How much does boundary error hurt each feature? (Gap 1, step 2)

    Perturbs letter boundaries with ±j-frame jitter, recomputes features, and
    reports Spearman(perturbed vs. unperturbed) per feature. Features near
    1.0 need no better aligner; features that collapse are aligner artifacts
    or set the alignment-quality bar for a CTC aligner.
    """
    from .evaluation.boundary_sensitivity import sensitivity_report

    cfg = load_config(config)
    idx = pd.read_csv(resolve_path(cfg, "interim_dir") / "clip_index.csv",
                      dtype={"clip_id": str})
    transcripts = dict(zip(idx["clip_id"], idx["transcript"]))
    levels = tuple(int(x) for x in jitters.split(","))
    report = sensitivity_report(
        resolve_path(cfg, "interim_dir") / "landmarks",
        transcripts,
        jitters=levels,
        reps=reps,
        seg_method=seg_method or cfg["segmentation"]["method"],
        seed=cfg["project"]["seed"],
        limit=limit,
        smooth_window=cfg["features"]["smooth_window_frames"],
        smooth_polyorder=cfg["features"]["smooth_polyorder"],
        pause_speed_percentile=cfg["features"]["pause_speed_percentile"],
        min_pause_frames=cfg["features"]["min_pause_frames"],
    )
    if report.empty:
        typer.echo("no clips analyzed — run `asl-fluency landmarks` first.")
        raise typer.Exit(code=1)
    n_clips = report.attrs.get("n_clips", 0)
    n_dropped = report.attrs.get("n_dropped", 0)
    typer.echo(
        f"analyzed {n_clips} clips"
        + (f" ({n_dropped} excluded: NaN base features)" if n_dropped else "")
    )
    typer.echo(report.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    out = resolve_path(cfg, "processed_dir") / "boundary_sensitivity.csv"
    report.to_csv(out, index=False)
    fragile = report[report["min_rho"] < 0.8]["feature"].tolist()
    if fragile:
        typer.echo(f"\nfragile features (min rho < 0.8): {', '.join(fragile)}")
    typer.echo(f"report -> {out}")


@app.command()
def train(
    config: Path | None = typer.Option(None, "--config", "-c"),
    ratings: Path | None = typer.Option(
        None, help="default: data/ratings/ratings_intake.csv"
    ),
    kind: str | None = typer.Option(
        None, help="mean | naive_speed | ridge | ridge_cv | ordinal | random_forest | "
        "gradient_boosting (default: config)"
    ),
    feature_set: str | None = typer.Option(
        None, help="naive | core | kinematic | kinematic_robust | full (default: config)"
    ),
    out: Path | None = typer.Option(
        None, help="default: data/processed/fluency_scorer.joblib"
    ),
) -> None:
    """Fit the chosen arm on ALL rated data and persist the scorer artifact.

    Run this AFTER `evaluate` has identified the winning arm via CV — the
    artifact inherits the small-sample caveats (Section 7): a research
    scorer, validated by the cross-validated metrics, not production.
    """
    from .data.fluency_labels import consensus_scores, load_ratings
    from .models.persist import fit_scorer, save_scorer

    cfg = load_config(config)
    ratings = ratings or resolve_path(cfg, "ratings_dir") / "ratings_intake.csv"
    scale = tuple(cfg["labels"]["rating_scale"])
    y = consensus_scores(load_ratings(ratings, scale))  # human-only consensus
    feats = pd.read_csv(resolve_path(cfg, "processed_dir") / "features.csv",
                        dtype={"clip_id": str})

    kind = kind or cfg["model"]["kind"]
    feature_set = feature_set or cfg["model"]["feature_set"]
    model, labeled = fit_scorer(feats, y, kind, feature_set,
                                seed=cfg["project"]["seed"])
    out = out or resolve_path(cfg, "processed_dir") / "fluency_scorer.joblib"
    save_scorer(model, out, kind=kind, feature_set=feature_set,
                n_train_clips=len(labeled), rating_scale=scale)
    typer.echo(f"trained {kind}/{feature_set} on {len(labeled)} rated clips -> {out}")
    typer.echo("validate with `evaluate` CV metrics; apply with `asl-fluency score`.")


@app.command()
def score(
    clip: Path | None = typer.Argument(
        None, help="a clip (frame dir or video) to score; omit with --from-features"
    ),
    transcript: str | None = typer.Option(None, help="spelled word, required for clip scoring"),
    from_features: bool = typer.Option(
        False, "--from-features", help="score every row of features.csv instead"
    ),
    model_path: Path | None = typer.Option(
        None, "--model", help="default: data/processed/fluency_scorer.joblib"
    ),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Apply a persisted scorer (from `train`) to new data."""
    from .models.persist import load_scorer, predict_scores

    cfg = load_config(config)
    model_path = model_path or resolve_path(cfg, "processed_dir") / "fluency_scorer.joblib"
    if not model_path.exists():
        raise typer.BadParameter(f"{model_path} not found — run `asl-fluency train` first.")
    artifact = load_scorer(model_path)

    if from_features:
        feats = pd.read_csv(resolve_path(cfg, "processed_dir") / "features.csv",
                            dtype={"clip_id": str})
        preds = predict_scores(artifact, feats)
        out = resolve_path(cfg, "processed_dir") / "scored_features.csv"
        preds.rename_axis("clip_id").reset_index().to_csv(out, index=False)
        typer.echo(preds.describe().to_string())
        typer.echo(f"scores -> {out}")
        return

    if clip is None or transcript is None:
        raise typer.BadParameter("score a clip: asl-fluency score CLIP --transcript WORD")
    from .features.pipeline import compute_feature_row
    from .landmarks.extractor import extract_landmarks

    lm_cfg = dict(cfg["landmarks"])
    landmarker_path = _project_path(lm_cfg.pop("model_path", None), cfg)
    lm_cfg.pop("backend", None)
    lm_cfg.pop("static_image_mode", None)
    seq = extract_landmarks(clip, clip.stem, model_path=landmarker_path, **lm_cfg)
    row = compute_feature_row(
        seq, transcript,
        seg_method=cfg["segmentation"]["method"],
        refine_radius=cfg["segmentation"]["refine_radius_frames"],
        smooth_window=cfg["features"]["smooth_window_frames"],
        smooth_polyorder=cfg["features"]["smooth_polyorder"],
        pause_speed_percentile=cfg["features"]["pause_speed_percentile"],
        min_pause_frames=cfg["features"]["min_pause_frames"],
    )
    pred = predict_scores(artifact, pd.DataFrame([row]))
    typer.echo(f"{clip.stem}: predicted fluency {pred.iloc[0]:.2f} "
               f"({artifact['kind']}/{artifact['feature_set']}, "
               f"trained on {artifact['n_train_clips']} clips)")


@app.command()
def agreement(
    ratings: Path = typer.Argument(..., help="long-format ratings CSV"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    include_ai: bool = typer.Option(
        False, "--include-ai", help="include vlm_* AI-judge rows in the report"
    ),
) -> None:
    """Ground-truth sanity check: Fleiss' kappa / ICC on collected ratings.

    Human raters only by default. To check the AI judge against the humans,
    run this on a file that contains both and pass --include-ai.
    """
    from .data.fluency_labels import load_ratings, split_ai_raters
    from .evaluation.agreement import agreement_report

    cfg = load_config(config)
    scale = tuple(cfg["labels"]["rating_scale"])
    df = load_ratings(ratings, scale)
    human, ai = split_ai_raters(df)
    if not ai.empty and not include_ai:
        typer.echo(
            f"note: {len(ai)} AI-judge rows excluded from the human agreement report "
            "(use --include-ai to fold them in)"
        )
    target = df if include_ai else human
    if target.empty:
        typer.echo("no ratings to report on yet.")
        return
    report = agreement_report(target)
    for k, v in report.items():
        typer.echo(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")


@app.command()
def clip_context(
    config: Path | None = typer.Option(None, "--config", "-c"),
    ratings: Path | None = typer.Option(
        None, help="clips to describe (default: data/ratings/ratings_intake.csv)"
    ),
    all_clips: bool = typer.Option(
        False, "--all", help="every clip with landmarks, not just the rated ones"
    ),
    out: Path | None = typer.Option(None, help="default: data/ratings/clip_context.csv"),
    force: bool = typer.Option(False, help="overwrite an existing clip_context.csv"),
) -> None:
    """Objective skin-tone (ITA) / lighting strata for the Section 7 fairness eval.

    Measures palm-pixel ITA and frame luminance from the clip media, located
    by the extracted landmarks — no human annotation needed. A proxy, not a
    demographic label: see landmarks/context.py for the caveats.
    """
    from .landmarks.context import build_clip_context

    cfg = load_config(config)
    out = out or resolve_path(cfg, "ratings_dir") / "clip_context.csv"
    if out.exists() and not force:
        typer.echo(f"{out} exists (may be hand-annotated); pass --force to overwrite.")
        raise typer.Exit(code=1)
    media = _load_clip_paths(cfg)
    if not all_clips:
        ratings = ratings or resolve_path(cfg, "ratings_dir") / "ratings_intake.csv"
        rated = set(pd.read_csv(ratings, dtype={"clip_id": str})["clip_id"])
        media = {c: p for c, p in media.items() if c in rated}
    ctx = build_clip_context(
        media,
        resolve_path(cfg, "interim_dir") / "landmarks",
        default_fps=float(cfg["landmarks"].get("default_fps", 30.0)),
    )
    if ctx.empty:
        typer.echo("No clips with both media and landmarks; run `landmarks` first.")
        raise typer.Exit(code=1)
    ctx.to_csv(out, index=False)
    typer.echo(ctx[["skin_tone", "lighting"]].value_counts(dropna=False).to_string())
    typer.echo(f"{len(ctx)} clips -> {out}")


@app.command()
def evaluate(
    ratings: Path = typer.Argument(..., help="long-format ratings CSV"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    rater: str | None = typer.Option(
        None, help="score against one rater only (single-rater pilot / low-ICC fallback)"
    ),
) -> None:
    """Ablation: naive vs. kinematic arms against consensus human ratings."""
    from .data.fluency_labels import consensus_scores, load_ratings, split_ai_raters
    from .evaluation.ablation import (
        PRIMARY_ARM,
        marginal_stratified_metrics,
        run_ablation,
    )

    cfg = load_config(config)
    scale = tuple(cfg["labels"]["rating_scale"])
    df = load_ratings(ratings, scale)
    if rater:
        df = df[df["rater_id"] == rater]
        if df.empty:
            typer.echo(f"No ratings from rater '{rater}' in {ratings}.")
            raise typer.Exit(code=1)
        typer.echo(f"single-rater evaluation: {rater} ({len(df)} clips)")
    human, ai = split_ai_raters(df)
    if not ai.empty:
        typer.echo(f"note: {len(ai)} AI-judge rows excluded from the human consensus.")
    y = consensus_scores(df)  # human-only by default: vlm_* rows never enter ground truth
    per_clip = human.groupby("clip_id")["rater_id"].nunique()
    min_raters = cfg["labels"].get("min_raters_per_clip", 3)
    thin = per_clip[per_clip < min_raters]
    if len(thin) and not rater:
        typer.echo(
            f"Warning: {len(thin)} clips have fewer than {min_raters} human raters; "
            "their consensus scores are noisy."
        )
    feats = pd.read_csv(resolve_path(cfg, "processed_dir") / "features.csv",
                        dtype={"clip_id": str})
    feats = feats[feats["clip_id"].isin(y.index)].reset_index(drop=True)
    if feats.empty:
        typer.echo(
            "No overlap between rated clips and features.csv. If the sheet was "
            "generated before features existed, regenerate it (it samples from "
            "features.csv now) or run `asl-fluency features` on the rated clips."
        )
        raise typer.Exit(code=1)
    group_col = "signer_id" if cfg["model"]["cv"] == "signer_group" else None
    n_splits = cfg["model"]["n_splits"]
    if group_col:
        n_groups = feats["signer_id"].nunique()
        if n_groups < 2:
            typer.echo(
                "Warning: only 1 signer in the rated set; signer-disjoint CV is "
                "meaningless — falling back to plain k-fold for this run."
            )
            group_col = None
        elif n_groups < n_splits:
            typer.echo(
                f"Warning: only {n_groups} signers in the rated set; reducing CV "
                f"folds from {n_splits} to {n_groups}."
            )
            n_splits = n_groups
    summary, details = run_ablation(
        feats, y, group_col=group_col, n_splits=n_splits,
        bootstrap_iters=cfg["evaluation"].get("bootstrap_iters", 1000),
    )
    with pd.option_context("display.width", 200, "display.max_columns", None):
        typer.echo(summary.round(3).to_string(index=False))
    suffix = f"_{rater}" if rater else ""
    out = resolve_path(cfg, "processed_dir") / f"ablation_results{suffix}.csv"
    summary.to_csv(out, index=False)

    # Stratified fairness evaluation (proposal Section 7), if context exists.
    # Uses the pre-registered primary arm — picking the CV winner here would
    # be a post-hoc choice on the same held-out predictions.
    oof = details[PRIMARY_ARM]["oof"] if PRIMARY_ARM in details else next(
        iter(details.values()))["oof"]
    context_path = resolve_path(cfg, "ratings_dir") / "clip_context.csv"
    if context_path.exists():
        ctx = pd.read_csv(context_path, dtype={"clip_id": str})
        strata = marginal_stratified_metrics(y, oof, ctx, cfg["evaluation"]["stratify_by"])
        typer.echo(f"\nStratified metrics ({PRIMARY_ARM}):")
        typer.echo(strata.round(3).to_string(index=False))
        strata.to_csv(
            resolve_path(cfg, "processed_dir") / f"stratified_results{suffix}.csv",
            index=False,
        )
    typer.echo(f"results -> {out}")


@app.command()
def site(
    out: Path = typer.Option(Path("_site"), help="output directory for the static site"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    ratings: Path | None = typer.Option(
        None, help="default: data/ratings/ratings_intake.csv"
    ),
) -> None:
    """Project webpage: recompute results from committed data, render static HTML.

    Run by the GitHub Pages workflow on every push that changes ratings, so the
    published numbers (and the pilot / consensus framing) always match the repo.
    """
    import os

    from .site import compute_site_results, render_site

    cfg = load_config(config)
    ratings = ratings or resolve_path(cfg, "ratings_dir") / "ratings_intake.csv"
    sheet = pd.read_csv(ratings, dtype={"clip_id": str, "rater_id": str})
    feats = pd.read_csv(resolve_path(cfg, "processed_dir") / "features.csv",
                        dtype={"clip_id": str})
    ctx_path = resolve_path(cfg, "ratings_dir") / "clip_context.csv"
    bnd_path = resolve_path(cfg, "processed_dir") / "boundary_sensitivity.csv"
    results = compute_site_results(
        sheet, feats,
        context=pd.read_csv(ctx_path, dtype={"clip_id": str}) if ctx_path.exists() else None,
        boundary=pd.read_csv(bnd_path) if bnd_path.exists() else None,
        n_splits=cfg["model"]["n_splits"],
        bootstrap_iters=cfg["evaluation"].get("bootstrap_iters", 1000),
    )
    meta = dict(cfg.get("site", {}))
    meta["commit"] = os.environ.get("GITHUB_SHA", "")
    index = render_site(results, out, meta)
    typer.echo(f"framing: {results['framing']} | {results['n_rated_clips']} rated clips "
               f"| {results['n_active_raters']} active raters -> {index}")


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
        delegate=lm_cfg.get("delegate", "cpu"),
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
