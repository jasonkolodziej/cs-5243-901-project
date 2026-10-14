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
instructors or fluent signers) with the built-in rating-sheet tooling.

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

## Gaps

> [!TIP]
> Gap 1 (forced alignment) is a *correctness* problem — it de-risks your main hypothesis. Gap 2 (video fusion) is a *scope* problem — it's the ablation arm, and the proposal itself warns it's the most likely place to overfit. Approach them in that order, and don't build Gap 2 until the labels exist.

### Gap 1: Letter-boundary segmentation (do this first)

**Why it matters more than it looks.** Every transition-zone feature (`trans_vel/jerk_*`, pause placement) is computed relative to boundaries. If boundaries are off, those features are noise — and your RQ is precisely about whether those features predict fluency. But note the asymmetry: global features (`letters_per_second`, duration CV, total pause time) are quite robust to boundary error. So the first question isn't "how do I get perfect alignment" — it's "how much does boundary error actually hurt my features?"

**Recommended sequence:**

1. **Hand-label letter boundaries on ~30–50 clips.** Small, tedious, but it converts segmentation from an assumption into a measurable quantity. Put them in `data/ratings/` as a CSV (`clip_id, letter, start_frame, end_frame`).
2. **Run a sensitivity analysis.** Perturb the uniform/energy-valley boundaries with random jitter (±2, ±4, ±6 frames), recompute features, and measure rank correlation of each feature against its unperturbed self. Features that survive jitter need no better aligner; features that collapse tell you what alignment quality you must reach. This is a few hours of work and might kill the whole problem.
3. **Only if needed, CTC forced alignment.** ChicagoFSWild(+) has clip-level transcripts but no frame-level letter labels — which is exactly the setting CTC was built for. Train a letter recognizer on landmark sequences (not pixels — cheaper, and you already have the extraction pipeline) over the 55k ChicagoFSWild+ clips, then run Viterbi forced alignment with the known transcript to get per-letter spans. This plugs directly into the existing `method="forced_alignment"` hook in `segmentation/aligner.py`.
4. Validate whatever you build against the hand-labeled set from step 1 and report boundary error in the paper — it directly addresses the Section 7 "boundary errors propagate" concern.

### Gap 2: Video-fusion arm (defer until labels exist)

**Don't write this code yet.** Its only job is the ablation table comparing kinematic-only vs. kinematic+video, and you can't run that meaningfully until you've collected the human ratings (target ~100+ clips × 3 raters, after checking ICC ≥ ~0.6 with the existing `agreement` command).

When you do build it, the honest constraint is Section 7's warning: ~100 labeled clips cannot support end-to-end video learning. So rule out training a video CNN from scratch, and choose between:

- **Frozen backbone + learned head (safest).** Run a pretrained video encoder (r3d_18 Kinetics via torchvision; better: a sign-pretrained I3D/VideoMAE checkpoint, e.g. WLASL-pretrained, which the proposal already earmarks for pretraining) over each letter span, freeze it, concatenate the embedding with your kinematic row, and train only the ridge/GBM head. Trainable parameters stay in the dozens — appropriate for your label count.
- **Learned pose encoder (my actual recommendation).** Instead of raw pixels, feed the 21×3 landmark sequences into a small temporal model (TCN or 2-layer GRU) and learn kinematic *embeddings* end-to-end. It's still the "fusion" comparison (hand-engineered vs. learned kinematic features), it's far cheaper than video, it sidesteps the lighting/skin-tone pixel-level bias concerns from Section 7, and it can be pretrained self-supervised on all 55k unlabeled ChicagoFSWild+ clips (e.g., masked-keypoint prediction) before touching your labels.

Whichever you pick: keep it on the **same signer-disjoint folds** as the other arms (the scaffold's `cross_validate` already enforces this), and compare arms on held-out test only once — repeated peeking at test with three arms and ~100 labels is how spurious "improvements" happen. One more subtlety: for the video/pose arm, augment only in ways that preserve fluency (brightness, crop, horizontal jitter) — never speed or temporal warping, since that changes the very quantity the raters scored.

### Suggested overall order

1. Collect ratings (use `rating-sheet`), check ICC/Fleiss' κ
2. Hand-label boundaries on the same ~50 clips → sensitivity analysis
3. Hypothesis test: kinematic vs. naive baseline (already runnable end-to-end)
4. If boundaries are the bottleneck → CTC aligner; if not → skip it and say so in the paper
5. Fused arm last, frozen/pretrained only
