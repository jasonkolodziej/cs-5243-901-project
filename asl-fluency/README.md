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
      ──► 4. model        naive baseline vs. ridge/GBM scorers      (models/)
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

Frame directories have no native frame rate, so `landmarks.default_fps` (30) is used for
rate features. Run `asl-fluency demo data/raw/ChicagoFSWild-Frames/<signer>/<clip>
--transcript WORD` to sanity-check one clip before the batch run.

No dataset has fluency labels — collect a small human-rated subset (1–5 scale,
instructors or fluent signers) with the built-in rating-sheet tooling.

## Running the experiments

```bash
# 1. index the corpus (signer-disjoint train/val/test split)
asl-fluency index

# 2. extract landmarks (data/interim/landmarks/*.npz)
asl-fluency landmarks            # add --limit 50 for a quick smoke run

# 3. compute the kinematic feature table
asl-fluency features

# 4. prepare the human-rating intake sheet, then collect 1-5 ratings
asl-fluency rating-sheet --raters instructor_a,instructor_b,fluent_c --n-clips 100

# 5. sanity-check ground truth: Fleiss' kappa + ICC(2,k)
asl-fluency agreement data/ratings/ratings_intake.csv

# 6. ablation vs. consensus ratings: naive speed-only vs. kinematic arms,
#    plus stratified metrics if data/ratings/clip_context.csv exists
asl-fluency evaluate data/ratings/ratings_intake.csv
```

Ratings format (long CSV, one row per rater × clip) and the optional
`clip_context.csv` (per-clip `skin_tone`, `lighting` for the stratified fairness
evaluation) are exemplified in `data/ratings/*.example.csv`.

## Experiment map → proposal

| Proposal item | Code |
|---|---|
| RQ / hypothesis test (kinematics vs. naive speed) | `evaluation/ablation.py` arms `naive_speed_only` vs. `kinematic_*` |
| Step 1 landmarks | `landmarks/extractor.py` (Tasks API, `SigningHandSelector`), `landmarks/live.py` (webcam/stream) |
| Step 2 letter boundaries | `segmentation/aligner.py` (`uniform`, `energy_valley`, forced-alignment hook) |
| Step 3 fluency features | `features/kinematics.py` (letters/s, transition velocity/jerk, pauses, log dimensionless jerk, path efficiency) |
| Step 4 learned scorer | `models/baseline.py`, `models/fused.py` (video-fusion arm, `--extra fused`) |
| Step 5 human validation | `data/fluency_labels.py`, `asl-fluency rating-sheet` |
| Metrics: Spearman/Pearson, MAE | `evaluation/metrics.py` (+ bootstrap CIs) |
| Inter-rater agreement (κ / ICC) | `evaluation/agreement.py` |
| Skin-tone / lighting stratified eval (Section 7) | `evaluation/ablation.stratified_metrics`, `landmarks/quality.py` |

## Design decisions tied to Section 7 (challenges)

- **Signer-disjoint CV everywhere** — fluency must generalize to unseen signers.
- **Interpretable scorers first** (ridge/GBM over ~18 kinematic features) because the
  rated set is small; the end-to-end fused model is an isolated ablation arm.
- **Tracking-quality covariates** (`detection_rate`, dropout runs, interframe jumps)
  travel with every feature row so degraded-tracking clips can be filtered or stratified.
- **Stratified metrics** by skin tone / lighting rather than aggregate-only reporting.

## Tests

```bash
uv run pytest          # synthetic end-to-end tests, no video data required
```

## Layout

```
configs/default.yaml        all experiment knobs (model, CV, features, paths)
src/asl_fluency/            the pipeline package (CLI: asl-fluency)
data/raw|interim|processed|ratings/
notebooks/                  exploratory analysis
tests/                      synthetic end-to-end tests

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
│   └── .gitkeep
│
├── pyproject.toml                   # uv project metadata, deps, scripts, tool config
├── README.md                        # setup, usage, experiment map, design decisions
├── TREE.md                          # this file — annotated directory layout
├── uv.lock                          # reproducible dependency lockfile
│
├── src/
│   └── asl_fluency/                 # main pipeline package (CLI entrypoint: asl-fluency)
│       ├── __init__.py
│       ├── cli.py                   # Typer CLI: index, landmarks, features, rating-sheet, …
│       ├── config.py                # YAML loading, deep-merge, path resolution, project_root()
│       │
│       ├── data/                    # dataset loaders & rating utilities
│       │   ├── __init__.py
│       │   ├── chicago_fs_wild.py   # ClipIndex, load_clip_index, signer-disjoint split, relative-path logic
│       │   └── fluency_labels.py    # rating-sheet template, consensus scores, inter-rater loading
│       │
│       ├── evaluation/              # metrics, agreement, ablation runner
│       │   ├── __init__.py
│       │   ├── ablation.py          # naive vs. kinematic arms, stratified_metrics
│       │   ├── agreement.py         # Fleiss' kappa, ICC(2,k), Cohen's kappa
│       │   └── metrics.py           # Spearman/Pearson/MAE + bootstrap CIs
│       │
│       ├── features/                # kinematic feature engineering
│       │   ├── __init__.py
│       │   ├── kinematics.py        # letters/s, velocity/jerk, pauses, path efficiency, LDJ
│       │   └── pipeline.py          # compute_feature_row, build_feature_table
│       │
│       ├── landmarks/               # MediaPipe hand-landmark extraction
│       │   ├── __init__.py
│       │   ├── extractor.py         # HandLandmarker Tasks API, SigningHandSelector, batch_extract
│       │   ├── live.py              # webcam / video-stream fluency monitor
│       │   └── quality.py           # tracking-quality covariates
│       │
│       ├── models/                  # regressors & optional fused video arm
│       │   ├── __init__.py
│       │   ├── baseline.py          # naive speed, ridge, random forest, gradient boosting
│       │   └── fused.py             # kinematic + raw-video embedding ablation (needs --extra fused)
│       │
│       └── segmentation/            # letter-boundary alignment inside a clip
│           ├── __init__.py
│           └── aligner.py           # uniform, energy_valley, forced_alignment hook
│
└── tests/                           # synthetic end-to-end tests (no video data required)
    └── test_pipeline.py             # frame-dir capture, loader, segmentation, features, metrics, agreement, ablation
```
