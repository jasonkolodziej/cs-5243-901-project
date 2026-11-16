# ASL Fingerspelling Fluency Scoring

Companion code for the research proposal **"Automated Fluency and Speed Scoring for ASL
Fingerspelling"** (Steele, Kolodziej, Fogeh). The system scores *how fluently* someone
fingerspells — speed, smoothness, and consistency of letter transitions — directly from
video, instead of recognizing *what* they spelled.

**Research question:** can kinematic features from hand-landmark tracking (signing rate,
inter-letter transition smoothness, pause patterns) predict human-perceived fingerspelling
fluency?

**Hypothesis:** a model combining hand-motion kinematics with letter-segment timing
correlates significantly with human fluency ratings, outperforming a naive baseline using
only overall signing speed (letters per second).

## Pipeline (proposal Section 4)

```
video ──► 1. landmarks    MediaPipe Hands, 21 keypoints/frame       (landmarks/)
      ──► 2. segmentation letter boundaries from the transcript     (segmentation/)
      ──► 3. features     rate, velocity/jerk, pauses, smoothness   (features/)
      ──► 4. model        naive baseline vs. shrunken ridge/ordinal  (models/)
      ──► 5. evaluation   Spearman/Pearson, MAE, ablations, κ/ICC   (evaluation/)
```

## Setup

```bash
cd asl-fluency
uv sync                     # core pipeline
uv sync --extra dev         # + pytest, ruff
uv sync --extra fused       # + torch, for the video-fusion ablation arm

# one-time: MediaPipe hand landmarker model (~8 MB)
curl -L -o assets/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

### Platform note: MediaPipe on macOS

The project depends on MediaPipe's native hand-landmarker runtime. On some macOS
installations, the native GraphService/Metal initialization aborts before any clip is
processed, producing the error seen in the issue report:

```
F ... graph_service.h:139] Check failed: service_ Service is unavailable.
```

This is a MediaPipe runtime issue in the underlying Metal/AVFoundation stack, not a bad
model file or a broken project path. In practice, the pipeline is most reliable on a
Linux/WSL environment or any host where MediaPipe can initialize its native graph service
correctly. If the command crashes immediately on macOS, the fix is to run the project on a
supported MediaPipe runtime rather than changing the dataset or CLI arguments.

## Try it on real video

Place any fingerspelling video (or a clip extracted from ChicagoFSWild) under
`data/raw/fs_sample/` and run the demo commands below. A good test clip is
`gallaudet/mj_bienvenu_4509` from ChicagoFSWild (transcript **INTERDISCIPLINARY**,
79 frames, fluent signing); once extracted it can be reassembled to
`data/raw/fs_sample/mj_bienvenu_4509.mp4`.

```bash
# single-clip report: landmarks -> letter segmentation -> fluency features
asl-fluency demo data/raw/fs_sample/mj_bienvenu_4509.mp4 \
  --transcript INTERDISCIPLINARY --annotated-out annotated.mp4

# live mode: webcam (default --source 0) or any video file as a stand-in stream
asl-fluency live                                   # webcam, skeleton + HUD, Q to stop
asl-fluency live --source path/to/clip.mp4 --transcript HELLO --no-display \
  --save-video overlay.mp4                         # headless / file source
```

Expected on a fluent sample clip: 100% detection, ~6.5 letters/sec, letter-duration CV ≈ 0.10.

The hand selector (`SigningHandSelector`) picks the *most active* hand via fingertip-motion
EMA — on clips with a visible resting hand this is what keeps features computed on the
signing hand (continuity-only selection locks onto the wrong hand).

## Datasets (proposal Section 5)

Download manually into `data/raw/`:

| Dataset | Size | Use |
|---|---|---|
| [ChicagoFSWild](https://ttic.uchicago.edu/~klivescu/ChicagoFSWild.htm) | 7,304 clips, 160 signers | main corpus |
| ChicagoFSWild+ | 55,232 clips, 260 signers | scale-up if compute allows |
| WLASL / MS-ASL | isolated signs | hand-pose pretraining only |

**Important: ChicagoFSWild(+) clips are directories of JPEG frames, not video files.**
The pipeline handles this natively — extract the nested frames archive so clips sit in
one of these layouts (the loader auto-detects them):

```
data/raw/ChicagoFSWild.csv                     <- annotation CSV (any depth works)
data/raw/ChicagoFSWild-Frames/<signer>/<clip>/0001.jpg ...   # or
data/raw/<signer>/<clip>/0001.jpg ...                        # or
data/raw/frames/<signer>/<clip>/0001.jpg ...
```

`ChicagoFSWild.csv` alone is not sufficient; it is metadata only. You must also have
the extracted clip media (frame directories or videos) on disk.

If `ChicagoFSWild-Frames.tgz` (or `.tar.gz`) is present under `data/raw/` at any depth
(for example `data/raw/ChicagoFSWild/ChicagoFSWild-Frames.tgz`), `asl-fluency index`
will auto-extract it into `data/raw/` before resolving clip paths.

`asl-fluency index` now keeps only clips that exist locally. If no clips are found, it
fails with a setup error explaining that frame extraction is missing.

Frame directories have no native frame rate, so `landmarks.default_fps` (30) is used for
rate features. Run `asl-fluency demo data/raw/ChicagoFSWild-Frames/<signer>/<clip>
--transcript WORD` to sanity-check one clip before the batch run.

No dataset has fluency labels — collect a small human-rated subset (1–5 scale,
instructors or fluent signers) with the built-in rating-sheet tooling and the
browser rating UI (`asl-fluency rate`).

**How small can the human-rated set be?** The evaluation is a correlation
between model scores and human consensus ratings, so the sample size is a
power question: n ≈ 47 clips detects ρ = 0.4 and n ≈ 30 detects ρ = 0.5
(80% power, α = .05, two-sided). Combined with stratified sampling and the
adaptive top-up loop (`--batch 2`, `--batch 3`, …), ~50–70 clips × 3 raters
is usually enough — far below the naive 100 × 3 default. Clips are short
(median < 1 s), so a 50-clip batch is ~10 minutes per rater.

## Running the experiments

```bash
# 1. index the corpus (signer-disjoint train/val/test split)
asl-fluency index

# If this reports skipped clips, your extraction is partial; this is expected.
# It will process only clips that currently exist on disk.

# 2. extract landmarks (data/interim/landmarks/*.npz)
asl-fluency landmarks            # add --limit 50 for a quick smoke run

# 3. compute the kinematic feature table
asl-fluency features

# 4. (optional) AI-judge pre-scores over the feature pool — aids stratified
#    sampling; never part of the human ground truth. Two backends ship:
#    "mock" (deterministic, plumbing tests) and "grok" (xAI vision model).
asl-fluency ai-rate --backend mock --limit 200

#    grok backend: Grok has no video-input API, so clips are scored from N
#    evenly-sampled frames sent in temporal order (default 16). One-time setup:
#      uv add openai                 # API client dependency
#      cp example.env .env           # then put your XAI_API_KEY in .env
#    Settings live in the ai_judge section of configs/default.yaml (model,
#    n_frames, temperature); CLI flags --backend/--model/--n-frames override
#    them. Raw per-clip JSON responses are kept under data/interim/ai_clips/
#    for auditability. Pilot on ~30-50 clips first and check the scores
#    rank-correlate with letters/sec (and later human ratings via
#    `asl-fluency agreement`) before trusting them for stratification.
asl-fluency ai-rate --backend grok --limit 200

# 5. stratified intake sheet (samples from features.csv — every rated clip is
#    guaranteed evaluable). Auto-stratifies on the AI score when present,
#    else letters/sec quintiles. ~50 clips x 3 raters ≈ 10 min per rater and
#    is powered to detect rho >= 0.4 at 80%.
asl-fluency rating-sheet --raters instructor_a,instructor_b,fluent_c --n-clips 50

# 6. collect ratings in the browser UI (watch, click 1-5, auto-advance;
#    saves per click, stop/resume any time)
asl-fluency rate

# 7. sanity-check ground truth: Fleiss' kappa + ICC(2,k), human raters only.
#    If ICC looks fine but the model correlation CI is too wide, top up with
#    another batch: asl-fluency rating-sheet ... --batch 2 (never re-samples
#    clips already in the sheet).
asl-fluency agreement data/ratings/ratings_intake.csv

# 8a. objective fairness strata (Section 7), no human annotation: palm-pixel
#     ITA skin-tone proxy + frame-luminance lighting -> data/ratings/clip_context.csv
asl-fluency clip-context              # rated clips; --all for the whole corpus

# 8b. ablation vs. consensus ratings: the pre-registered primary arm
#     (core_ridge_cv) vs. naive speed-only, a paired-bootstrap delta-Spearman
#     test, the mean-baseline MAE floor, secondary arms, and marginal stratified
#     metrics (+ mean signed error) if clip_context.csv exists.
#     AI-judge (vlm_*) rows are automatically excluded from the consensus.
asl-fluency evaluate data/ratings/ratings_intake.csv
asl-fluency evaluate data/ratings/ratings_intake.csv --rater fluent_c   # one rater only

# 9. boundary ground truth + sensitivity (Roadmap Gap 1): label letter spans
#    on the same rated clips in the browser, then measure how much boundary
#    error would hurt each feature
asl-fluency label-boundaries          # saves data/ratings/boundary_labels.csv
asl-fluency boundary-sensitivity      # -> data/processed/boundary_sensitivity.csv

# 10. fit the pre-registered primary arm on all rated data and persist the
#     scorer (config default: ridge_cv / core), then apply it to new clips or
#     the whole feature table
asl-fluency train                     # or --kind ordinal --feature-set core, ...
asl-fluency score data/raw/fs_sample/mj_bienvenu_4509.mp4 --transcript INTERDISCIPLINARY
asl-fluency score --from-features     # score every row of features.csv
```

Ratings format (long CSV, one row per rater × clip) and the optional
`clip_context.csv` (per-clip `skin_tone`, `lighting` for the stratified fairness
evaluation) are exemplified in `data/ratings/*.example.csv`. `clip-context`
writes the latter automatically; it refuses to overwrite an existing
(possibly hand-annotated) file without `--force`.

## Modelling and pre-registered analysis plan

The first ablation run (one rater, 50 clips) showed every multi-feature arm at
ρ ≈ 0 with MAE *worse than predicting the mean*: fixed-α ridge and GBM over ~18
features overfit ~40 training clips per signer-disjoint fold (GBM train MAE ≈
0.02 vs. held-out 0.83). The fix, consistent with Section 7's small-label
warning:

- **`core` feature set, chosen a priori** — one feature per construct named in
  the RQ: rate (`letters_per_second`), smoothness (`log_dimless_jerk`), pauses
  (`pause_total_s`). All three are whole-clip features, robust to boundary
  error (boundary-sensitivity ρ ≥ 0.998). *Not* selected by correlation with
  the collected ratings, so the primary test is not tuned on its own data.
- **`ridge_cv`** — shrinkage chosen by efficient leave-one-out *inside* each
  training fold (nested; no test leakage).
- **`ordinal`** — proportional-odds model (statsmodels `OrderedModel`) for the
  ordered 1–5 scale; predicts E[rating].
- **`mean`** — training-fold mean; the MAE floor. Its correlation is reported
  as NaN (fold means are anti-correlated with held-out ratings by construction).
- **Paired bootstrap Δ Spearman** vs. `naive_speed_only` on the same
  out-of-fold predictions (`delta_spearman`, `delta_ci`, `p_vs_naive` columns) —
  the actual hypothesis test, which separate per-arm CIs cannot provide.

Plan fixed **before** the instructor ratings are seen (see
`evaluation/ablation.py`, `PRIMARY_ARM`):

| Item | Decision |
|---|---|
| Ground truth | mean of human raters; valid only if ICC(2,k) ≥ 0.6 |
| Primary test | `core_ridge_cv` vs. `naive_speed_only`, paired-bootstrap Δρ, one-sided α = .05 |
| Secondary | `core_ordinal`, `kinematic_robust_ridge_cv`, legacy α=1 ridge / GBM — reported, never used to re-pick a winner |
| Fairness | marginal metrics + mean signed error per skin-tone / lighting stratum, primary arm |
| ICC < 0.6 or ratings late | per-rater results (`evaluate --rater …`) + agreement as findings; no consensus claim |

**Single-rater pilot (`fluent_c`, n = 50, 27 signers)** — directionally
consistent with the hypothesis, not yet significant:

| arm | Spearman [95% CI] | MAE | Δρ vs. naive (p) |
|---|---|---|---|
| mean_baseline | — | 0.663 | — |
| naive_speed_only | 0.06 [−0.23, 0.36] | 0.647 | — |
| **core_ridge_cv** (primary) | 0.19 [−0.11, 0.46] | 0.663 | +0.13 (p = .14) |
| core_ordinal | 0.26 [−0.02, 0.50] | 0.661 | +0.20 (p = .11) |
| kinematic_ridge (α = 1, legacy) | −0.00 [−0.29, 0.28] | 0.858 | −0.06 |
| kinematic_gbm (legacy) | −0.12 [−0.37, 0.14] | 0.825 | −0.18 |

Fairness pilot: detection rate ≥ 0.98 in every skin-tone / lighting stratum
(no tracker-quality gap); skin-tone signed errors |bias| ≤ 0.15. Full analysis,
diagnostics and a simulated 3-rater dress rehearsal are in
[`notebooks/01_pilot_modelling_and_fairness.ipynb`](notebooks/01_pilot_modelling_and_fairness.ipynb)
— re-run it top to bottom once the instructor ratings land:

```bash
uv sync --extra dev
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace \
  notebooks/01_pilot_modelling_and_fairness.ipynb
```

## Experiment map → proposal

| Proposal item | Code |
|---|---|
| RQ / hypothesis test (kinematics vs. naive speed) | `evaluation/ablation.py`: primary `core_ridge_cv` vs. `naive_speed_only`, paired bootstrap (`metrics.paired_bootstrap_delta`) |
| Step 1 landmarks | `landmarks/extractor.py` (Tasks API, `SigningHandSelector`), `landmarks/live.py` (webcam/stream) |
| Step 2 letter boundaries | `segmentation/aligner.py` (`uniform` + `energy_valley` shipped; `forced_alignment` is a stub hook — see Roadmap Gap 1); ground truth via `asl-fluency label-boundaries`, error tolerance via `asl-fluency boundary-sensitivity` |
| Step 3 fluency features | `features/kinematics.py` (letters/s, transition velocity/jerk, pauses, log dimensionless jerk, path efficiency) |
| Step 4 learned scorer | `models/baseline.py` (`mean`, `naive_speed`, `ridge_cv`, `ordinal`, ridge/RF/GBM), `models/fused.py` (fusion arm, `--extra fused`; deferred — see Roadmap Gap 2) |
| Step 5 human validation | `data/fluency_labels.py` (stratified sampling, human-only consensus), `asl-fluency rating-sheet` + `asl-fluency rate` (browser UI) |
| AI-judge pre-scoring (sampling aid, not ground truth) | `ai_judge.py`, `asl-fluency ai-rate` |
| Metrics: Spearman/Pearson, MAE | `evaluation/metrics.py` (+ bootstrap CIs) |
| Inter-rater agreement (κ / ICC) | `evaluation/agreement.py` |
| Skin-tone / lighting stratified eval (Section 7) | `landmarks/context.py` + `asl-fluency clip-context` (ITA / luminance strata), `evaluation/ablation.marginal_stratified_metrics`, `landmarks/quality.py` |

## Roadmap

Two known gaps, in the order they should be tackled. Everything else in the
pipeline (index → landmarks → features → ratings → agreement → evaluate) is
already runnable end-to-end.

### Gap 1: letter-boundary segmentation (do this first)

Every transition-zone feature (`trans_vel/jerk_*`, pause placement) is computed
relative to letter boundaries — and the RQ is precisely about whether those
features predict fluency. Global features (`letters_per_second`, duration CV,
total pause time) are far more robust to boundary error, so the first question
is not "how do I get perfect alignment" but **"how much does boundary error
actually hurt my features?"**

1. Hand-label letter boundaries on ~30–50 clips (ideally the same clips that
   get fluency ratings — one session serves both): `(clip_id, letter,
   start_frame, end_frame)` CSV under `data/ratings/`. **Tooling shipped:**
   `asl-fluency label-boundaries` opens the browser UI (scrub the flipbook,
   click the joint between letters; starts from an even split, auto-saves to
   `boundary_labels.csv`).
2. **Sensitivity analysis — shipped** as `asl-fluency boundary-sensitivity`:
   jitters uniform/energy-valley boundaries by ±2/±4/±6 frames, recomputes
   features, and rank-correlates each feature against its unperturbed self.
   Features that survive jitter need no better aligner; features that
   collapse set the alignment quality bar. This may kill the whole problem.
   First look on 20 real clips: whole-clip features (LDJ, pauses, path
   efficiency) are perfectly robust; `trans_*` statistics degrade gracefully;
   `letter_dur_std_s` / `letter_dur_cv` collapse (near-artifacts under
   uniform segmentation). **Full-corpus run (7,236 clips) confirms this**:
   ρ = 1.00 for whole-clip features, 0.71–0.90 for `trans_*`, but ≈ 0.0 /
   0.48 for `letter_dur_std_s` / `letter_dur_cv`. The artifact features are
   excluded from the `kinematic_robust` feature set, which `evaluate` runs
   as an extra ablation arm (`kinematic_robust_ridge`) so the keep/drop
   decision is made empirically against human ratings.
3. Only if needed: **CTC forced alignment** — train a letter recognizer on
   *landmark sequences* (not pixels) over ChicagoFSWild+ (clip-level
   transcripts, no frame labels: exactly the CTC setting), then Viterbi-align
   against the known transcript. Plugs directly into the existing
   `method="forced_alignment"` hook in `segmentation/aligner.py`.
4. Validate against the hand-labeled set and report boundary error in the
   paper — this directly answers Section 7's "boundary errors propagate".

### Gap 2: fusion arm (defer until labels exist)

Its only job is the kinematic-only vs. kinematic+learned ablation row, which
is meaningless until human ratings exist and ICC ≥ ~0.6 (`asl-fluency
agreement`). Section 7's warning rules out training a video model from
scratch on ~100 labels, so the choice is:

- **Frozen backbone + learned head** (safest): pretrained video encoder
  (r3d_18/Kinetics; better: a WLASL-pretrained I3D/VideoMAE, which the
  proposal already earmarks for pretraining) over each letter span, frozen,
  embeddings concatenated with the kinematic row; train only the ridge/GBM
  head. Trainable parameters stay in the dozens.
- **Learned pose encoder** (preferred): a small temporal model (TCN or
  2-layer GRU) over the 21×3 landmark sequences — still the hand-engineered
  vs. learned comparison, far cheaper than pixels, sidesteps the pixel-level
  skin-tone/lighting bias, and pretrainable self-supervised (e.g.
  masked-keypoint prediction) on all 55k unlabeled ChicagoFSWild+ clips.

Rules for either: same signer-disjoint folds as the other arms
(`cross_validate` already enforces this); compare arms on held-out test
**once** (repeated peeking with ~100 labels manufactures spurious wins);
augment only in fluency-preserving ways (brightness, crop, horizontal jitter —
never speed or temporal warping, which changes the very quantity raters
scored).

### Also shipped

- `train` / `score` commands: `evaluate` fits models only inside CV folds and
  discards them; `train` fits the chosen arm on all rated data and persists
  it to `data/processed/fluency_scorer.joblib` (joblib), and `score` applies
  the artifact to a new clip (`--transcript`) or the whole feature table
  (`--from-features`).

### Suggested overall order

1. Ratings pilot: `features` on ~1,000 clips → `rating-sheet --n-clips 50` →
   `rate` → `agreement` (ICC ≥ 0.6?). The "~100+ clips" target is reached
   adaptively: top up with `--batch 2`, `--batch 3`, … only while the Spearman
   CIs are too wide — don't pre-commit human hours the statistics may not need.
2. Boundary labels on the same clips → `boundary-sensitivity`.
3. Hypothesis test: `clip-context` → `evaluate` — the pre-registered table,
   run **once** on the final consensus (single-rater pilot already done; see
   "Modelling and pre-registered analysis plan").
4. Branch: boundaries the bottleneck → CTC aligner; else skip and say so in
   the paper. CIs too wide → next rating batch.
5. Fusion arm last — frozen/pretrained only.

## Design decisions tied to Section 7 (challenges)

- **Signer-disjoint CV everywhere** — fluency must generalize to unseen signers.
- **Interpretable scorers first** — the primary arm is 3 a-priori features with
  nested-CV shrinkage, because ~18 features overfit ~50 rated clips; ridge/GBM over
  the full set and the end-to-end fused model are secondary ablation arms.
- **Tracking-quality covariates** (`detection_rate`, dropout runs, interframe jumps)
  travel with every feature row so degraded-tracking clips can be filtered or stratified.
- **Stratified metrics** by skin tone / lighting rather than aggregate-only reporting,
  sliced one factor at a time (the joint grid is too sparse at n≈50) and with mean
  signed error, since systematic over/under-scoring is the bias that matters. Strata
  come from pixels (ITA proxy + luminance), not subjective annotation — a proxy
  confounded with capture conditions, and reported as such.

## Project webpage (GitHub Pages)

`asl-fluency site --out _site` recomputes every headline number from committed
data (`ratings_intake.csv`, `features.csv`, `clip_context.csv`,
`boundary_sensitivity.csv`) and renders a static page plus `results.json` and
SVG figures. Only aggregates are published, never per-clip or per-rater scores.

[`.github/workflows/pages.yml`](../.github/workflows/pages.yml) runs it (after
the test suite) on every push to `main` that touches ratings, features, code or
config, and deploys to GitHub Pages. The page frames itself from the data:

| Raters with ratings | ICC(2,k) | Page says |
|---|---|---|
| < 2 | — | "Preliminary — single-rater pilot" |
| ≥ 2 | ≥ 0.6 | "Final — consensus of N raters" |
| ≥ 2 | < 0.6 | "Ratings complete, low agreement" (no consensus claim) |

The hypothesis verdict (supported / directionally consistent / not supported)
is also derived from the paired test, so committing the instructor ratings is
the only step needed to update the published result. Slide / video links live
in the `site:` section of `configs/default.yaml` (empty = "coming soon").

One-time setup: repo **Settings → Pages → Source: GitHub Actions** (or
`gh api -X POST repos/<owner>/<repo>/pages -f build_type=workflow`). Pages on a
private repo needs GitHub Pro / Team / Enterprise (free with the Student
Developer Pack); the published site is public either way.

Preview locally:

```bash
uv run asl-fluency site --out _site && python3 -m http.server -d _site 8000
```

## Tests

```bash
uv run pytest          # synthetic end-to-end tests, no video data required
```

## Layout

```
.
├── assets/                          # one-time-download binary artifacts (not in git)
│   └── hand_landmarker.task         # MediaPipe HandLandmarker model (~8 MB)
│
├── configs/                         # experiment configuration overrides
│   └── default.yaml                 # all knobs: paths, model, CV, features, segmentation
│
├── data/                            # experiment data — raw, interim, processed, ratings
│   ├── interim/                     # artefacts produced during pipeline stages
│   │   ├── .gitkeep
│   │   ├── clip_index.csv           # per-clip manifest with *relative* video_path
│   │   └── landmarks/               # per-frame MediaPipe hand landmarks (*.npz)
│   │
│   ├── processed/                   # final tables ready for modelling / evaluation
│   │   └── .gitkeep                 # features.csv, ablation_results.csv, stratified_results.csv
│   │
│   ├── ratings/                     # human fluency ratings and optional context
│   │   ├── .gitkeep
│   │   ├── ratings_intake.csv       # the sheet the browser UI writes into
│   │   ├── rating_batches.csv       # per-batch strata log (adaptive collection)
│   │   ├── ratings_ai.csv           # AI-judge scores (vlm_* rows; NOT ground truth)
│   │   ├── clip_context.csv         # auto skin_tone (ITA) / lighting strata (`clip-context`)
│   │   ├── clip_context.example.csv # example: per-clip skin_tone / lighting for stratified eval
│   │   └── ratings.example.csv      # example: long-format rater × clip ratings
│   │
│   └── raw/                         # manually-downloaded corpora (not in git)
│       ├── .gitkeep
│       ├── ChicagoFSWild/           # annotation CSVs from the official release
│       │   ├── ChicagoFSWild.csv    # main metadata: filename, label_proc, signer, partition, …
│       │   ├── HandAnnotation.csv   # hand-annotation metadata
│       │   ├── README               # official dataset README
│       │   └── individual.csv       # per-signer demographic info
│       ├── aslized/                 # extracted frame directories (<signer>/<clip>/0001.jpg)
│       └── fs_sample/               # drop a single reassembled video here for quick demo runs
│
├── notebooks/                       # exploratory analysis (Jupyter / Quarto)
│   └── 01_pilot_modelling_and_fairness.ipynb  # analysis plan, modelling fix, fairness, rehearsal
│
├── pyproject.toml                   # uv project metadata, deps, scripts, tool config
├── README.md                        # setup, usage, experiment map, design decisions
├── TREE.md                          # this file — annotated directory layout
├── uv.lock                          # reproducible dependency lockfile
│
├── src/
│   └── asl_fluency/                 # main pipeline package (CLI entrypoint: asl-fluency)
│       ├── __init__.py
│       ├── cli.py                   # Typer CLI: index, landmarks, features, rating-sheet,
│       │                            #   rate (browser UI), ai-rate, agreement, evaluate
│       ├── ai_judge.py              # AI-judge pre-scoring: rubric prompt, clip rendering,
│       │                            #   backends: mock (plumbing) + grok (xAI vision API)
│       ├── config.py                # YAML loading, deep-merge, path resolution, project_root()
│       ├── site.py                  # project webpage: recompute results, render static HTML
│       │
│       ├── data/                    # dataset loaders & rating utilities
│       │   ├── __init__.py
│       │   ├── chicago_fs_wild.py   # ClipIndex, load_clip_index, signer-disjoint split, relative-path logic
│       │   ├── fluency_labels.py    # stratified sampling, rating sheets, human-only consensus, save_rating
│       │   ├── boundary_labels.py   # hand-labeled letter spans (Gap 1 ground truth) schema + storage
│       │   └── ratings_ui.py        # stdlib browser UIs: fluency rating + boundary labeling
│       │
│       ├── evaluation/              # metrics, agreement, ablation runner
│       │   ├── __init__.py
│       │   ├── ablation.py          # pre-registered arms, paired test, (marginal) stratified metrics
│       │   ├── agreement.py         # Fleiss' kappa, ICC(2,k), Cohen's kappa
│       │   ├── boundary_sensitivity.py  # boundary-jitter feature robustness (Gap 1)
│       │   └── metrics.py           # Spearman/Pearson/MAE, bootstrap CIs, paired bootstrap Δρ
│       │
│       ├── features/                # kinematic feature engineering
│       │   ├── __init__.py
│       │   ├── kinematics.py        # letters/s, velocity/jerk, pauses, path efficiency, LDJ
│       │   └── pipeline.py          # compute_feature_row, build_feature_table
│       │
│       ├── landmarks/               # MediaPipe hand-landmark extraction
│       │   ├── __init__.py
│       │   ├── context.py           # ITA skin-tone proxy + luminance lighting strata
│       │   ├── extractor.py         # HandLandmarker Tasks API, SigningHandSelector, batch_extract
│       │   ├── live.py              # webcam / video-stream fluency monitor
│       │   └── quality.py           # tracking-quality covariates
│       │
│       ├── models/                  # regressors & optional fused video arm
│       │   ├── __init__.py
│       │   ├── baseline.py          # mean, naive speed, ridge(_cv), ordinal, RF, GBM
│       │   ├── persist.py           # fit/persist/load the final scorer (train/score)
│       │   └── fused.py             # kinematic + raw-video embedding ablation (needs --extra fused)
│       │
│       └── segmentation/            # letter-boundary alignment inside a clip
│           ├── __init__.py
│           └── aligner.py           # uniform, energy_valley, forced_alignment hook
│
└── tests/                           # synthetic end-to-end tests (no video data required)
    └── test_pipeline.py             # frame-dir capture, loader, segmentation, features, metrics, agreement, ablation
```
