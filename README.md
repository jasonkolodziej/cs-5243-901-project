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

## Try it on real video (no dataset download needed)

`data/raw/fs_sample/` ships one real ChicagoFSWild clip (79 frames reassembled to
`mj_bienvenu_4509.mp4`, transcript "INTERDISCIPLINARY", a fluent signer) plus the full
`ChicagoFSWild.csv` metadata, so every stage can be smoke-tested on real data:

```bash
# single-clip report: landmarks -> letter segmentation -> fluency features
asl-fluency demo data/raw/fs_sample/mj_bienvenu_4509.mp4 \
  --transcript INTERDISCIPLINARY --annotated-out annotated.mp4

# live mode: webcam (default --source 0) or any video file as a stand-in stream
asl-fluency live                                   # webcam, skeleton + HUD, Q to stop
asl-fluency live --source path/to/clip.mp4 --transcript HELLO --no-display \
  --save-video overlay.mp4                         # headless / file source
```

Expected on the sample clip: 100% detection, ~6.5 letters/sec, letter-duration CV ≈ 0.10 —
fast, highly consistent signing, exactly what a fluent clip should look like.

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
```
