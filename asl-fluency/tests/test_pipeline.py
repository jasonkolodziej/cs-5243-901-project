"""End-to-end tests on synthetic landmark sequences (no video needed)."""

import os
import tarfile
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from asl_fluency.evaluation.ablation import (
    marginal_stratified_metrics,
    run_ablation,
    stratified_metrics,
)
from asl_fluency.evaluation.agreement import agreement_report, cohen_kappa, fleiss_kappa, icc_2k
from asl_fluency.evaluation.metrics import bootstrap_ci, compute_metrics
from asl_fluency.features.pipeline import KINEMATIC_FEATURES, compute_feature_row
from asl_fluency.landmarks.extractor import N_LANDMARKS, LandmarkSequence
from asl_fluency.segmentation.aligner import segment_letters


def make_seq(clip_id="c0", n_frames=120, fps=30.0, dropout=0.0, seed=0):
    """Synthetic landmark sequence: smooth wrist arc + mild noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n_frames)
    lm = np.zeros((n_frames, N_LANDMARKS, 3), dtype=np.float32)
    lm[:, 0, 0] = 0.5 + 0.2 * np.sin(t) + rng.normal(0, 0.005, n_frames)  # wrist x
    lm[:, 0, 1] = 0.5 + 0.1 * np.cos(t) + rng.normal(0, 0.005, n_frames)  # wrist y
    lm[:, 1:, :] = lm[:, [0], :] + rng.normal(0, 0.02, (n_frames, N_LANDMARKS - 1, 3))
    frame_ok = np.ones(n_frames, bool)
    if dropout > 0:
        frame_ok[rng.random(n_frames) < dropout] = False
        lm[~frame_ok] = np.nan
    return LandmarkSequence(clip_id, lm, frame_ok, fps)


class TestFrameDirectorySource:
    def test_frame_dir_capture_reads_sorted(self, tmp_path):
        import cv2

        from asl_fluency.landmarks.extractor import open_frame_source

        for i in (3, 1, 2):  # write out of order; reader must sort
            img = np.full((8, 8, 3), i * 80, dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"{i:04d}.jpg"), img)
        cap, fps = open_frame_source(tmp_path, default_fps=25.0)
        assert fps == 25.0
        vals = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            vals.append(int(frame[0, 0, 0]))
        assert vals == [80, 160, 240]

    def test_loader_resolves_frame_roots(self, tmp_path):
        from asl_fluency.data.chicago_fs_wild import load_clip_index

        frames = tmp_path / "ChicagoFSWild-Frames" / "aslized" / "clip_0001"
        frames.mkdir(parents=True)
        (tmp_path / "ChicagoFSWild.csv").write_text(
            ",filename,label_proc,signer\n0,aslized/clip_0001,hello,0000\n"
        )
        ci = load_clip_index(tmp_path)
        assert len(ci) == 1
        row = ci.frame.iloc[0]
        assert row["video_path"] == str(frames)  # resolved into frames root
        assert row["transcript"] == "HELLO"
        assert row["clip_id"] == "clip_0001"

    def test_loader_ignores_non_clip_csvs(self, tmp_path):
        from asl_fluency.data.chicago_fs_wild import load_clip_index

        # individual.csv has no transcript/video columns -> must not be picked
        (tmp_path / "individual.csv").write_text("signer,age\n0000,30\n")
        clip_dir = tmp_path / "a" / "b_0001"
        clip_dir.mkdir(parents=True)
        (tmp_path / "ChicagoFSWild.csv").write_text(
            ",filename,label_proc,signer\n0,a/b_0001,hi,0000\n"
        )
        ci = load_clip_index(tmp_path)
        assert len(ci) == 1
        assert ci.frame.iloc[0]["transcript"] == "HI"

    def test_loader_drops_missing_clips_when_partial_extract(self, tmp_path):
        from asl_fluency.data.chicago_fs_wild import load_clip_index

        present = tmp_path / "ChicagoFSWild-Frames" / "aslized" / "clip_0001"
        present.mkdir(parents=True)
        (tmp_path / "ChicagoFSWild.csv").write_text(
            ",filename,label_proc,signer\n"
            "0,aslized/clip_0001,hello,0000\n"
            "1,aslized/clip_9999,world,0001\n"
        )
        ci = load_clip_index(tmp_path)
        assert len(ci) == 1
        assert ci.frame.iloc[0]["clip_id"] == "clip_0001"

    def test_loader_auto_extracts_frames_archive(self, tmp_path):
        from asl_fluency.data.chicago_fs_wild import load_clip_index

        build_root = tmp_path / "build"
        clip_dir = build_root / "ChicagoFSWild-Frames" / "aslized" / "clip_0001"
        clip_dir.mkdir(parents=True)
        (clip_dir / "0001.jpg").write_bytes(b"x")

        archive = tmp_path / "ChicagoFSWild" / "ChicagoFSWild-Frames.tgz"
        archive.parent.mkdir(parents=True)
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(build_root / "ChicagoFSWild-Frames", arcname="ChicagoFSWild-Frames")

        (tmp_path / "ChicagoFSWild.csv").write_text(
            ",filename,label_proc,signer\n0,aslized/clip_0001,hello,0000\n"
        )

        ci = load_clip_index(tmp_path)
        assert len(ci) == 1
        assert (tmp_path / "ChicagoFSWild-Frames" / "aslized" / "clip_0001").is_dir()


class TestSegmentation:
    def test_uniform_covers_letters(self):
        seq = make_seq()
        spans = segment_letters(seq, "HELLO", method="uniform")
        assert [s[2] for s in spans] == list("HELLO")
        assert spans[0][0] < spans[0][1]
        assert all(s1[1] <= s2[0] for s1, s2 in pairwise(spans))

    def test_energy_valley_keeps_count(self):
        spans = segment_letters(make_seq(), "WORLD", method="energy_valley")
        assert len(spans) == 5

    def test_rejects_empty_transcript(self):
        with pytest.raises(ValueError):
            segment_letters(make_seq(), "123", method="uniform")


class TestFeatures:
    def test_feature_row_has_all_columns(self):
        row = compute_feature_row(make_seq(), "HELLO", seg_method="uniform")
        for f in KINEMATIC_FEATURES + ["letters_per_second", "detection_rate"]:
            assert f in row, f"missing feature {f}"
        assert row["letters_per_second"] > 0
        assert row["detection_rate"] == 1.0

    def test_dropout_lowers_detection_rate(self):
        row = compute_feature_row(make_seq(dropout=0.2, seed=3), "HI")
        assert 0.7 < row["detection_rate"] < 0.9


class TestMetrics:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 3.0])
        m = compute_metrics(y, y)
        assert m.spearman == pytest.approx(1.0)
        assert m.mae == pytest.approx(0.0)

    def test_bootstrap_ci_brackets_point_estimate(self):
        rng = np.random.default_rng(0)
        y = rng.uniform(1, 5, 50)
        pred = y + rng.normal(0, 0.5, 50)
        ci = bootstrap_ci(y, pred, iters=200)
        m = compute_metrics(y, pred)
        assert ci["spearman"][0] <= m.spearman <= ci["spearman"][1]


class TestAgreement:
    def test_perfect_agreement(self):
        wide = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4], "c": [1, 2, 3, 4]})
        assert fleiss_kappa(wide, [1, 2, 3, 4]) == pytest.approx(1.0)
        assert icc_2k(wide) == pytest.approx(1.0)
        assert cohen_kappa(wide["a"], wide["b"]) == pytest.approx(1.0)

    def test_report_keys(self):
        ratings = pd.DataFrame({
            "clip_id": ["c0", "c0", "c1", "c1", "c2", "c2"],
            "rater_id": ["r1", "r2"] * 3,
            "rating": [3, 4, 2, 2, 5, 4],
        })
        rep = agreement_report(ratings)
        assert {"n_clips", "n_raters", "fleiss_kappa", "icc_2k", "cohen_kappa_quad"} <= set(rep)


class TestAblation:
    def test_kinematic_beats_chance_on_signal(self):
        """Synthetic ground truth driven by letters-per-second + smoothness."""
        rng = np.random.default_rng(0)
        rows, y = {}, {}
        for i in range(60):
            cid = f"c{i:03d}"
            seq = make_seq(cid, n_frames=int(rng.integers(60, 180)), seed=i)
            tr = "ABC"
            rows[cid] = compute_feature_row(seq, tr, seg_method="uniform")
            rows[cid]["signer_id"] = f"s{i % 12}"
            # fluent = fast + smooth
            y[cid] = np.clip(
                3 + 2 * (rows[cid]["letters_per_second"] - 1.0) + rng.normal(0, 0.3),
                1, 5,
            )
        feats = pd.DataFrame(rows.values())
        y = pd.Series(y)
        summary, _ = run_ablation(feats, y, n_splits=3, bootstrap_iters=200)
        arms = summary.set_index("arm")
        assert {"mean_baseline", "core_ridge_cv", "core_ordinal",
                "kinematic_robust_ridge"} <= set(arms.index)
        assert arms.loc["kinematic_ridge", "spearman"] > 0.3  # signal recoverable
        assert arms.loc["core_ridge_cv", "spearman"] > 0.3
        assert arms.loc["core_ordinal", "spearman"] > 0.3
        # paired hypothesis-test columns vs. the naive baseline
        assert arms.loc["naive_speed_only", "delta_spearman"] == 0.0
        assert 0.0 <= arms.loc["core_ridge_cv", "p_vs_naive"] <= 1.0
        # mean baseline: MAE floor only, its fold-mean "correlation" is an artifact
        assert np.isnan(arms.loc["mean_baseline", "spearman"])
        assert np.isfinite(arms.loc["mean_baseline", "mae"])

    def test_ridge_cv_shrinks_noise_features(self):
        """Pure-noise features at n~50: nested shrinkage must not overfit
        worse than the mean baseline (the failure mode of fixed alpha=1)."""
        from asl_fluency.models.baseline import make_model

        rng = np.random.default_rng(1)
        n = 50
        feats = pd.DataFrame(rng.normal(size=(n, len(KINEMATIC_FEATURES) + 1)),
                             columns=["letters_per_second"] + KINEMATIC_FEATURES)
        y = rng.integers(2, 6, n).astype(float)
        tr, te = slice(0, 40), slice(40, None)
        mae = {}
        for kind in ("mean", "ridge_cv"):
            m = make_model(kind, "kinematic").fit(feats.iloc[tr], y[tr])
            mae[kind] = np.abs(m.predict(feats.iloc[te]) - y[te]).mean()
        assert mae["ridge_cv"] <= mae["mean"] * 1.1

    def test_paired_bootstrap_delta(self):
        from asl_fluency.evaluation.metrics import _bootstrap_indices, paired_bootstrap_delta

        rng = np.random.default_rng(0)
        y = rng.normal(size=80)
        good = y + rng.normal(0, 0.3, 80)
        noise = rng.normal(size=80)
        same = paired_bootstrap_delta(y, good, good, iters=200)
        assert same["delta"] == 0.0 and same["p_one_sided"] == 1.0
        better = paired_bootstrap_delta(y, good, noise, iters=200)
        assert better["delta"] > 0.5 and better["ci_lo"] > 0 and better["p_one_sided"] < 0.05
        groups = np.array(["a", "a", "b", "b", "b", "c"])
        sampled = _bootstrap_indices(np.random.default_rng(2), len(groups), groups)
        counts = np.bincount(sampled, minlength=len(groups))
        assert counts[0] == counts[1]
        assert counts[2] == counts[3] == counts[4]

    def test_run_ablation_passes_aligned_signer_ids_to_bootstraps(self, monkeypatch):
        import asl_fluency.evaluation.ablation as ablation

        ids = ["c0", "c1", "c2", "c3"]
        features = pd.DataFrame({
            "clip_id": ids,
            "signer_id": ["s1", "s1", "s2", "s3"],
            "letters_per_second": [1.0, 2.0, 3.0, 4.0],
        })
        y = pd.Series([1.0, 2.0, 3.0, 4.0], index=ids)
        captured = []

        def fake_cv(factory, features, y, **kwargs):
            return pd.Series(y.to_numpy(), index=features["clip_id"]), pd.DataFrame()

        def fake_ci(y_true, y_pred, **kwargs):
            captured.append(np.asarray(kwargs["groups"]))
            return {"spearman": (0.0, 1.0), "pearson": (0.0, 1.0), "mae": (0.0, 1.0)}

        def fake_delta(y_true, pred_a, pred_b, **kwargs):
            captured.append(np.asarray(kwargs["groups"]))
            return {"delta": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "p_one_sided": 1.0}

        monkeypatch.setattr(ablation, "cross_validate", fake_cv)
        monkeypatch.setattr(ablation, "bootstrap_ci", fake_ci)
        monkeypatch.setattr(ablation, "paired_bootstrap_delta", fake_delta)
        arms = [
            {"name": ablation.BASELINE_ARM, "kind": "naive_speed", "feature_set": "naive"},
            {"name": "core", "kind": "ridge_cv", "feature_set": "core"},
        ]
        ablation.run_ablation(features, y, arms=arms, bootstrap_iters=1, group_col=None)
        assert len(captured) == 3
        assert all(groups.tolist() == features["signer_id"].tolist() for groups in captured)

    def test_ordinal_degenerate_fold_falls_back_to_mean(self):
        from asl_fluency.models.baseline import make_model

        feats = pd.DataFrame({"letters_per_second": [1.0, 2.0, 3.0],
                              "log_dimless_jerk": [0.0, 1.0, 2.0],
                              "pause_total_s": [0.1, 0.2, 0.3]})
        m = make_model("ordinal", "core").fit(feats, np.array([4.0, 4.0, 4.0]))
        assert np.allclose(m.predict(feats), 4.0)

    @pytest.mark.parametrize(
        ("converged", "params"), [(False, [0.0]), (True, [np.nan])]
    )
    def test_ordinal_rejects_unusable_fit(self, monkeypatch, converged, params):
        from types import SimpleNamespace

        from statsmodels.miscmodels.ordinal_model import OrderedModel

        from asl_fluency.features.pipeline import FEATURE_SETS
        from asl_fluency.models.baseline import OrdinalScorer

        monkeypatch.setattr(
            OrderedModel,
            "fit",
            lambda *args, **kwargs: SimpleNamespace(
                mle_retvals={"converged": converged}, params=np.asarray(params)
            ),
        )
        cols = FEATURE_SETS["core"]
        feats = pd.DataFrame(np.arange(18, dtype=float).reshape(6, 3), columns=cols)
        model = OrdinalScorer(cols).fit(feats, np.array([1, 2, 3, 4, 5, 3], dtype=float))
        assert model.res_ is None
        assert np.allclose(model.predict(feats), 3.0)

    def test_marginal_stratified_metrics_reports_signed_bias(self):
        ids = [f"c{i}" for i in range(8)]
        y_true = pd.Series([1, 2, 3, 4, 1, 2, 3, 4], index=ids, dtype=float)
        y_pred = y_true + np.r_[[0.0] * 4, [1.0] * 4]  # "dark" over-scored by +1
        ctx = pd.DataFrame({"clip_id": ids,
                            "skin_tone": ["light"] * 4 + ["dark"] * 4,
                            "lighting": ["low", "bright"] * 4})
        out = marginal_stratified_metrics(y_true, y_pred, ctx, ["skin_tone", "lighting"])
        assert set(out["factor"]) == {"skin_tone", "lighting"}
        bias = out[out.factor == "skin_tone"].set_index("stratum")["mean_signed_error"]
        assert bias["dark"] == pytest.approx(1.0) and bias["light"] == pytest.approx(0.0)

    def test_marginal_stratified_metrics_skips_unmatched_context(self):
        y_true = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
        y_pred = y_true + 0.1
        context = pd.DataFrame({"clip_id": ["x", "y"], "skin_tone": ["light", "dark"]})
        assert marginal_stratified_metrics(
            y_true, y_pred, context, ["skin_tone"]
        ).empty

    def test_stratified(self):
        y_true = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})
        y_pred = pd.Series({"a": 1.1, "b": 2.1, "c": 3.1, "d": 3.9})
        ctx = pd.DataFrame({
            "clip_id": ["a", "b", "c", "d"],
            "skin_tone": ["light", "light", "dark", "dark"],
        })
        out = stratified_metrics(y_true, y_pred, ctx, ["skin_tone"])
        assert set(out["skin_tone"]) == {"light", "dark"}


class TestRelativePathResolution:
    def test_loader_makes_paths_relative_when_under_base_dir(self, tmp_path):
        from asl_fluency.data.chicago_fs_wild import load_clip_index

        frames = tmp_path / "ChicagoFSWild-Frames" / "aslized" / "clip_0001"
        frames.mkdir(parents=True)
        (tmp_path / "ChicagoFSWild.csv").write_text(
            ",filename,label_proc,signer\n0,aslized/clip_0001,hello,0000\n"
        )
        # When base_dir is the parent of tmp_path, the resolved path is under it
        ci = load_clip_index(tmp_path, base_dir=tmp_path)
        assert len(ci) == 1
        row = ci.frame.iloc[0]
        # Path should be relative to base_dir
        assert not os.path.isabs(row["video_path"])
        assert row["video_path"] == "ChicagoFSWild-Frames/aslized/clip_0001"
        assert row["transcript"] == "HELLO"

    def test_loader_keeps_absolute_when_outside_base_dir(self, tmp_path):
        from asl_fluency.data.chicago_fs_wild import load_clip_index

        # frames live outside base_dir
        other = tmp_path / "other" / "frames" / "a" / "clip_0002"
        other.mkdir(parents=True)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "ChicagoFSWild.csv").write_text(
            ",filename,label_proc,signer\n0,a/clip_0002,world,0001\n"
        )
        # Create a symlink-ish resolution: raw_dir/a -> other/frames/a
        (raw_dir / "a").mkdir()
        (raw_dir / "a" / "clip_0002").mkdir()
        # Actually simpler: just use base_dir that does NOT contain the resolved path
        ci = load_clip_index(raw_dir, base_dir=tmp_path / "nowhere")
        row = ci.frame.iloc[0]
        # Because the resolved path IS under raw_dir, and raw_dir is NOT under nowhere,
        # the path stays absolute
        assert os.path.isabs(row["video_path"])

    def test_loader_defaults_to_absolute_without_base_dir(self, tmp_path):
        from asl_fluency.data.chicago_fs_wild import load_clip_index

        frames = tmp_path / "ChicagoFSWild-Frames" / "aslized" / "clip_0001"
        frames.mkdir(parents=True)
        (tmp_path / "ChicagoFSWild.csv").write_text(
            ",filename,label_proc,signer\n0,aslized/clip_0001,hello,0000\n"
        )
        ci = load_clip_index(tmp_path)  # base_dir defaults to None
        row = ci.frame.iloc[0]
        assert os.path.isabs(row["video_path"])
        assert row["video_path"] == str(frames)


class TestFeaturePipelineGuards:
    def test_build_feature_table_empty_when_no_transcript_matches(self, tmp_path):
        import numpy as np

        from asl_fluency.features.pipeline import build_feature_table
        from asl_fluency.landmarks.extractor import N_LANDMARKS, LandmarkSequence

        lm_dir = tmp_path / "landmarks"
        lm_dir.mkdir()
        # save a landmark file for clip "c99" but only ask for "c00" transcript
        seq = LandmarkSequence("c99", np.zeros((10, N_LANDMARKS, 3), np.float32),
                               np.ones(10, bool), 30.0)
        seq.save(lm_dir / "c99.npz")
        table = build_feature_table(lm_dir, {"c00": "HELLO"})
        assert table.empty

    def test_build_feature_table_empty_on_empty_dir(self, tmp_path):
        from asl_fluency.features.pipeline import build_feature_table

        lm_dir = tmp_path / "landmarks"
        lm_dir.mkdir()
        table = build_feature_table(lm_dir, {})
        assert table.empty


class TestStratifiedSample:
    def _features(self, n=60, seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "clip_id": [f"c{i:03d}" for i in range(n)],
            "signer_id": [f"s{i % 10}" for i in range(n)],
            "letters_per_second": np.linspace(0.5, 8.0, n),
            "detection_rate": rng.uniform(0.5, 1.0, n).round(3),
        })

    def test_spans_strata_and_size(self):
        from asl_fluency.data.fluency_labels import stratified_rating_sample

        out = stratified_rating_sample(
            self._features(), 20, seed=1, score_col="letters_per_second"
        )
        assert len(out) == 20
        assert set(out["stratum"]) == {0, 1, 2, 3, 4}
        # stratified => wider speed coverage than the slowest/fastest 20 alone
        speeds = self._features().set_index("clip_id").loc[out["clip_id"],
                                                           "letters_per_second"]
        assert speeds.min() < 1.5 and speeds.max() > 6.5

    def test_deterministic_with_seed(self):
        from asl_fluency.data.fluency_labels import stratified_rating_sample

        a = stratified_rating_sample(self._features(), 20, seed=7,
                                     score_col="letters_per_second")
        b = stratified_rating_sample(self._features(), 20, seed=7,
                                     score_col="letters_per_second")
        assert a["clip_id"].tolist() == b["clip_id"].tolist()

    def test_batches_are_disjoint(self):
        """Adaptive top-up: batch 2 must never re-sample batch 1 clips."""
        from asl_fluency.data.fluency_labels import stratified_rating_sample

        feats = self._features()
        b1 = stratified_rating_sample(feats, 20, seed=1,
                                      score_col="letters_per_second")
        b2 = stratified_rating_sample(feats, 20, seed=2,
                                      score_col="letters_per_second",
                                      exclude_clip_ids=set(b1["clip_id"]))
        assert not set(b1["clip_id"]) & set(b2["clip_id"])
        assert len(b2) == 20

    def test_detection_rate_gate(self):
        from asl_fluency.data.fluency_labels import stratified_rating_sample

        feats = self._features()
        feats["detection_rate"] = 0.5  # everything fails the gate
        out = stratified_rating_sample(feats, 20, seed=1,
                                       score_col="letters_per_second",
                                       min_detection_rate=0.8)
        assert out.empty

    def test_max_per_signer(self):
        from asl_fluency.data.fluency_labels import stratified_rating_sample

        out = stratified_rating_sample(
            self._features(), 20, seed=1, score_col="letters_per_second",
            max_per_signer=2,
        )
        assert (out["signer_id"].value_counts() <= 2).all()


class TestConsensusGuard:
    def _ratings(self):
        return pd.DataFrame({
            "clip_id": ["c0", "c0", "c0", "c1", "c1", "c1"],
            "rater_id": ["h1", "h2", "vlm_judge"] * 2,
            "rating": [2.0, 4.0, 5.0, 1.0, 3.0, 5.0],
        })

    def test_consensus_excludes_ai_by_default(self):
        from asl_fluency.data.fluency_labels import consensus_scores

        y = consensus_scores(self._ratings())
        assert y["c0"] == pytest.approx(3.0)  # (2+4)/2, not (2+4+5)/3

    def test_consensus_include_ai_opt_in(self):
        from asl_fluency.data.fluency_labels import consensus_scores

        y = consensus_scores(self._ratings(), include_ai=True)
        assert y["c0"] == pytest.approx((2 + 4 + 5) / 3)

    def test_split_ai_raters(self):
        from asl_fluency.data.fluency_labels import split_ai_raters

        human, ai = split_ai_raters(self._ratings())
        assert set(human["rater_id"]) == {"h1", "h2"}
        assert set(ai["rater_id"]) == {"vlm_judge"}


class TestRatingsIO:
    def test_load_ratings_drops_blank_cells(self, tmp_path):
        """A partially filled intake sheet (UI mid-session) must load cleanly."""
        from asl_fluency.data.fluency_labels import load_ratings

        p = tmp_path / "r.csv"
        p.write_text("clip_id,rater_id,rating\nc0,h1,4\nc0,h2,\nc1,h1,2\n")
        df = load_ratings(p)
        assert len(df) == 2  # blank row dropped, no validation error

    def test_save_rating_creates_updates_and_clears(self, tmp_path):
        from asl_fluency.data.fluency_labels import save_rating

        p = tmp_path / "r.csv"
        save_rating(p, "c0", "h1", 4)
        save_rating(p, "c0", "h1", 5)  # update, not duplicate
        save_rating(p, "c1", "h1", 2)
        df = pd.read_csv(p, dtype={"clip_id": str})
        assert len(df) == 2
        assert df.set_index("clip_id").loc["c0", "rating"] == 5
        save_rating(p, "c1", "h1", None)  # clear back to blank
        df = pd.read_csv(p, dtype={"clip_id": str}).fillna({"rating": ""})
        assert df.set_index("clip_id").loc["c1", "rating"] == ""

    def test_save_rating_rejects_out_of_scale(self, tmp_path):
        from asl_fluency.data.fluency_labels import save_rating

        with pytest.raises(ValueError):
            save_rating(tmp_path / "r.csv", "c0", "h1", 6)


class TestRatingUI:
    def _frame_dir(self, tmp_path, name="clip_a", n=6):
        import cv2

        d = tmp_path / name
        d.mkdir()
        for i in range(n):
            cv2.imwrite(str(d / f"{i:04d}.jpg"), np.full((8, 8, 3), i * 30, np.uint8))
        return d

    def test_per_rater_order_deterministic_and_distinct(self):
        from asl_fluency.data.ratings_ui import per_rater_order

        clips = [f"c{i}" for i in range(20)]
        a1, a2 = per_rater_order(clips, "r1"), per_rater_order(clips, "r1")
        b1 = per_rater_order(clips, "r2")
        assert a1 == a2  # stable across sessions (resume-safe)
        assert a1 != b1  # different per rater (washes out order effects)
        assert sorted(a1) == sorted(clips)

    def test_catalog_serves_frames_and_blocks_traversal(self, tmp_path):
        from asl_fluency.data.ratings_ui import ClipCatalog

        d = self._frame_dir(tmp_path)
        cat = ClipCatalog({"clip_a": d})
        info = cat.info("clip_a")
        assert info["kind"] == "frames" and len(info["frames"]) == 6
        assert cat.media_file("clip_a", "0001.jpg") == d / "0001.jpg"
        assert cat.media_file("clip_a", "../secret") is None
        assert cat.media_file("clip_a", "9999.jpg") is None
        assert cat.info("nope") is None

    def test_http_end_to_end(self, tmp_path):
        """Real server: session -> clip info -> frame bytes -> POST a rating."""
        import json
        import threading
        import urllib.request

        from asl_fluency.data.fluency_labels import rating_sheet_template
        from asl_fluency.data.ratings_ui import serve

        d = self._frame_dir(tmp_path)
        sheet = tmp_path / "ratings.csv"
        rating_sheet_template(["clip_a"], ["h1", "h2"]).to_csv(sheet, index=False)

        server = serve(sheet, {"clip_a": d}, port=0)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            base = f"http://127.0.0.1:{port}"
            sess = json.loads(urllib.request.urlopen(base + "/api/session?rater=h1").read())
            assert sess["clips"] == ["clip_a"]
            info = json.loads(urllib.request.urlopen(base + "/api/clip/clip_a").read())
            frame = urllib.request.urlopen(base + info["frames"][0]).read()
            assert frame[:2] == b"\xff\xd8"  # JPEG magic bytes
            req = urllib.request.Request(
                base + "/api/rate",
                data=json.dumps({"rater_id": "h1", "clip_id": "clip_a",
                                 "rating": 4}).encode(),
                headers={"Content-Type": "application/json"},
            )
            res = json.loads(urllib.request.urlopen(req).read())
            assert res["ok"] and res["progress"]["n_rated"] == 1
            saved = pd.read_csv(sheet)
            row = saved[(saved.clip_id == "clip_a") & (saved.rater_id == "h1")]
            assert row["rating"].iloc[0] == 4
        finally:
            server.shutdown()
            server.server_close()


class TestAIJudge:
    def test_mock_backend_valid_and_deterministic(self, tmp_path):
        from asl_fluency.ai_judge import score_clip

        r1 = score_clip(tmp_path / "clip_a", backend="mock")
        r2 = score_clip(tmp_path / "clip_a", backend="mock")
        assert r1 == r2 and 1 <= r1["rating"] <= 5

    def test_live_backend_raises_with_guidance(self, tmp_path):
        from asl_fluency.ai_judge import score_clip

        with pytest.raises(NotImplementedError, match="not implemented"):
            score_clip(tmp_path / "clip_a", backend="gemini")

    def test_run_ai_rating_writes_vlm_rows_and_resumes(self, tmp_path):
        from asl_fluency.ai_judge import run_ai_rating
        from asl_fluency.data.fluency_labels import split_ai_raters

        clips = {f"c{i}": tmp_path / f"c{i}" for i in range(3)}
        out = tmp_path / "ratings_ai.csv"
        df1 = run_ai_rating(clips, out, backend="mock", limit=2)
        assert len(df1) == 2
        df2 = run_ai_rating(clips, out, backend="mock")  # resumes, adds the 3rd
        assert len(df2) == 3
        _, ai = split_ai_raters(df2)
        assert set(ai["rater_id"]) == {"vlm_judge"}

    def test_render_clip_video_from_frame_dir(self, tmp_path):
        import cv2

        from asl_fluency.ai_judge import render_clip_video

        d = tmp_path / "clip_a"
        d.mkdir()
        for i in range(5):
            cv2.imwrite(str(d / f"{i:04d}.jpg"), np.full((16, 16, 3), i * 40, np.uint8))
        out = render_clip_video(d, tmp_path / "out" / "clip_a.mp4")
        assert out.exists() and out.stat().st_size > 0

    def test_prompt_blind_to_transcript(self):
        from asl_fluency.ai_judge import build_prompt

        p = build_prompt()
        assert "1-5" in p and "fluency" in p.lower()


class TestBoundarySensitivity:
    def _landmark_dir(self, tmp_path, n=6):
        lm_dir = tmp_path / "landmarks"
        lm_dir.mkdir()
        transcripts = {}
        for i in range(n):
            seq = make_seq(f"c{i}", n_frames=90, seed=i)
            seq.save(lm_dir / f"c{i}.npz")
            transcripts[f"c{i}"] = "HELLO"
        return lm_dir, transcripts

    def test_jitter_spans_identity_at_zero(self):
        from asl_fluency.evaluation.boundary_sensitivity import jitter_spans

        spans = [(0, 10, "A"), (10, 20, "B"), (20, 30, "C")]
        assert jitter_spans(spans, 0, np.random.default_rng(0)) == spans

    def test_jitter_spans_stay_ordered(self):
        from asl_fluency.evaluation.boundary_sensitivity import jitter_spans

        spans = [(0, 11, "A"), (11, 19, "B"), (19, 30, "C")]
        for seed in range(20):
            out = jitter_spans(spans, 6, np.random.default_rng(seed))
            assert out[0][0] == 0 and out[-1][1] == 30  # endpoints fixed
            for (s1, e1, _), (s2, e2, _) in pairwise(out):
                assert s1 < e1 <= s2

    def test_report_shape_and_robust_features(self, tmp_path):
        from asl_fluency.evaluation.boundary_sensitivity import sensitivity_report

        lm_dir, transcripts = self._landmark_dir(tmp_path)
        rep = sensitivity_report(lm_dir, transcripts, jitters=(2,), reps=2)
        assert list(rep.columns) == ["feature", "rho_jit2", "min_rho"]
        assert len(rep) == len(KINEMATIC_FEATURES)
        # whole-clip features are boundary-independent: must be perfectly robust
        ldj = rep.set_index("feature").loc["log_dimless_jerk", "rho_jit2"]
        assert ldj == pytest.approx(1.0)

    def test_nan_clips_excluded_not_poisoning(self, tmp_path):
        """One sparse clip (NaN base features) must not NaN out whole columns."""
        from asl_fluency.evaluation.boundary_sensitivity import sensitivity_report

        lm_dir, transcripts = self._landmark_dir(tmp_path)
        # add a sparse clip: passes the segmentation gate but too sparse to
        # smooth -> NaN base features
        n = 60
        lm = np.zeros((n, N_LANDMARKS, 3), dtype=np.float32)
        frame_ok = np.zeros(n, bool)
        frame_ok[[10, 30, 50]] = True
        lm[~frame_ok] = np.nan
        LandmarkSequence("sparse", lm, frame_ok, 30.0).save(lm_dir / "sparse.npz")
        transcripts["sparse"] = "AB"

        rep = sensitivity_report(lm_dir, transcripts, jitters=(2,), reps=2)
        assert rep.attrs["n_dropped"] == 1
        assert rep.attrs["n_clips"] == 6
        ldj = rep.set_index("feature").loc["log_dimless_jerk", "rho_jit2"]
        assert ldj == pytest.approx(1.0)  # finite, not poisoned

    def test_constant_feature_yields_nan_without_warning(self, tmp_path):
        """Constant columns (the artifact signature) are recorded as NaN
        deliberately — no scipy ConstantInputWarning may escape."""
        import warnings

        from scipy.stats import ConstantInputWarning

        from asl_fluency.evaluation.boundary_sensitivity import sensitivity_report

        lm_dir, transcripts = self._landmark_dir(tmp_path)
        n = 60
        lm = np.zeros((n, N_LANDMARKS, 3), dtype=np.float32)
        frame_ok = np.zeros(n, bool)
        frame_ok[[10, 30, 50]] = True
        lm[~frame_ok] = np.nan
        LandmarkSequence("sparse", lm, frame_ok, 30.0).save(lm_dir / "sparse.npz")
        transcripts["sparse"] = "AB"

        with warnings.catch_warnings():
            warnings.simplefilter("error", ConstantInputWarning)  # would raise
            rep = sensitivity_report(lm_dir, transcripts, jitters=(2,), reps=2)
        # NaN is a legitimate verdict here; numeric values must be finite
        finite = rep["rho_jit2"].dropna()
        assert (finite.abs() <= 1.0).all()


class TestBoundaryLabels:
    def test_save_load_roundtrip_and_replace(self, tmp_path):
        from asl_fluency.data.boundary_labels import (
            load_boundary_labels,
            save_clip_spans,
            spans_for,
        )

        p = tmp_path / "boundary_labels.csv"
        save_clip_spans(p, "c0", [(0, 10, "H"), (10, 20, "I")])
        save_clip_spans(p, "c1", [(0, 15, "A")])
        # re-label c0 -> old rows replaced, c1 untouched
        save_clip_spans(p, "c0", [(0, 8, "H"), (8, 20, "I")])
        df = load_boundary_labels(p)
        assert len(df[df.clip_id == "c0"]) == 2
        assert spans_for(df, "c0") == [(0, 8, "H"), (8, 20, "I")]
        assert spans_for(df, "c1") == [(0, 15, "A")]
        assert spans_for(df, "nope") == []

    def test_rejects_disordered_spans(self, tmp_path):
        from asl_fluency.data.boundary_labels import save_clip_spans

        with pytest.raises(ValueError):
            save_clip_spans(tmp_path / "b.csv", "c0", [(10, 20, "A"), (0, 10, "B")])


class TestBoundaryUI:
    def test_http_boundary_mode(self, tmp_path):
        import json
        import threading
        import urllib.request

        import cv2

        from asl_fluency.data.ratings_ui import serve

        d = tmp_path / "clip_a"
        d.mkdir()
        for i in range(6):
            cv2.imwrite(str(d / f"{i:04d}.jpg"), np.full((8, 8, 3), i * 40, np.uint8))

        out = tmp_path / "boundary_labels.csv"
        server = serve(None, {"clip_a": d}, port=0, mode="boundaries",
                       boundary_path=out, transcripts={"clip_a": "HI"})
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            base = f"http://127.0.0.1:{port}"
            sess = json.loads(urllib.request.urlopen(base + "/api/session").read())
            assert sess["clips"] == ["clip_a"]
            lab = json.loads(
                urllib.request.urlopen(base + "/api/clip_label/clip_a").read())
            assert lab["transcript"] == "HI" and lab["spans"] == []
            spans = [
                {"letter": "H", "start_frame": 0, "end_frame": 3},
                {"letter": "I", "start_frame": 3, "end_frame": 6},
            ]
            req = urllib.request.Request(
                base + "/api/label",
                data=json.dumps({"clip_id": "clip_a", "spans": spans}).encode(),
                headers={"Content-Type": "application/json"},
            )
            assert json.loads(urllib.request.urlopen(req).read())["ok"]
            from asl_fluency.data.boundary_labels import load_boundary_labels
            df = load_boundary_labels(out)
            assert len(df) == 2 and set(df["letter"]) == {"H", "I"}
        finally:
            server.shutdown()
            server.server_close()


class TestPersist:
    def _features(self, n=30, seed=0):
        rng = np.random.default_rng(seed)
        cols = ["letters_per_second"] + KINEMATIC_FEATURES
        feats = pd.DataFrame(
            rng.normal(size=(n, len(cols))), columns=cols
        )
        feats["clip_id"] = [f"c{i}" for i in range(n)]
        feats["letters_per_second"] = np.linspace(1, 7, n)  # signal
        return feats

    def test_models_are_picklable(self):
        import pickle

        from asl_fluency.models.baseline import make_model

        feats = self._features()
        y = pd.Series(np.linspace(1, 5, 30), index=[f"c{i}" for i in range(30)])
        for kind in ("naive_speed", "ridge", "gradient_boosting", "mean", "ridge_cv", "ordinal"):
            fs = "naive" if kind in ("naive_speed", "mean") else "kinematic"
            m = make_model(kind, fs).fit(feats, y.to_numpy())
            blob = pickle.dumps(m)  # closure classes would raise here
            assert len(blob) > 0

    def test_fit_save_load_predict(self, tmp_path):
        from asl_fluency.models.persist import (
            fit_scorer,
            load_scorer,
            predict_scores,
            save_scorer,
        )

        feats = self._features()
        y = pd.Series(np.linspace(1, 5, 30), index=[f"c{i}" for i in range(30)])
        model, labeled = fit_scorer(feats, y, "ridge", "kinematic")
        assert len(labeled) == 30
        path = save_scorer(model, tmp_path / "scorer.joblib", kind="ridge",
                           feature_set="kinematic", n_train_clips=30)
        artifact = load_scorer(path)
        assert artifact["kind"] == "ridge" and artifact["n_train_clips"] == 30
        preds = predict_scores(artifact, feats.head(5))
        assert len(preds) == 5 and preds.between(1, 5).all()  # clipped to scale
        # perfect linear signal -> training-set fit should be strong
        full = predict_scores(artifact, feats)
        assert np.corrcoef(full, y)[0, 1] > 0.9

    def test_fit_scorer_rejects_empty_overlap(self):
        from asl_fluency.models.persist import fit_scorer

        feats = self._features()
        y = pd.Series([3.0], index=["unrelated_clip"])
        with pytest.raises(ValueError, match="no overlap"):
            fit_scorer(feats, y, "ridge", "kinematic")


class TestFeatureSets:
    def test_kinematic_robust_drops_aligner_artifacts(self):
        from asl_fluency.features.pipeline import FEATURE_SETS

        robust = FEATURE_SETS["kinematic_robust"]
        assert "letter_dur_std_s" not in robust
        assert "letter_dur_cv" not in robust
        assert "letters_per_second" in robust
        assert "trans_vel_mean" in robust  # degraded but real signal: kept
        assert len(robust) == len(FEATURE_SETS["kinematic"]) - 2


class TestSparseTrackingRows:
    def _sparse_seq(self):
        """3 detected frames (passes the segmentation gate for 2 letters)
        but too few to smooth -> NaN kinematics."""
        n = 60
        lm = np.zeros((n, N_LANDMARKS, 3), dtype=np.float32)
        frame_ok = np.zeros(n, bool)
        frame_ok[[10, 30, 50]] = True
        lm[~frame_ok] = np.nan
        return LandmarkSequence("sparse_clip", lm, frame_ok, 30.0)

    def test_no_nan_warnings_on_sparse_clip(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            row = compute_feature_row(self._sparse_seq(), "AB")
        assert np.isnan(row["trans_vel_mean"])  # honestly NaN, not noise

    def test_build_feature_table_drops_nan_rows(self, tmp_path):
        from asl_fluency.features.pipeline import build_feature_table

        lm_dir = tmp_path / "landmarks"
        lm_dir.mkdir()
        self._sparse_seq().save(lm_dir / "sparse_clip.npz")
        make_seq("healthy_clip", n_frames=90, seed=1).save(lm_dir / "healthy_clip.npz")
        table = build_feature_table(
            lm_dir, {"sparse_clip": "AB", "healthy_clip": "HELLO"}
        )
        assert table["clip_id"].tolist() == ["healthy_clip"]


class TestClipContext:
    def test_ita_bands(self):
        from asl_fluency.landmarks.context import ita_band, ita_degrees

        assert ita_band(ita_degrees(np.array([[80.0, 5.0, 15.0]]))) == "light"   # ~63 deg
        assert ita_band(ita_degrees(np.array([[60.0, 10.0, 20.0]]))) == "medium"  # ~27 deg
        assert ita_band(ita_degrees(np.array([[35.0, 12.0, 20.0]]))) == "dark"    # ~-37 deg
        assert np.isnan(ita_degrees(np.array([[60.0, 0.0, -5.0]])))  # b*<=0: undefined

    def test_clip_context_from_frame_dir(self, tmp_path):
        """Palm patch must sample the hand colour, not the background."""
        import cv2

        from asl_fluency.landmarks.context import build_clip_context

        clips, seqs = {}, tmp_path / "lm"
        seqs.mkdir()
        for cid, bgr in (("pale", (190, 210, 240)), ("deep", (40, 60, 95))):
            d = tmp_path / cid
            d.mkdir()
            for i in range(4):
                img = np.full((64, 64, 3), (30, 160, 30), np.uint8)  # green backdrop
                cv2.rectangle(img, (16, 16), (48, 48), bgr, -1)     # the "hand"
                cv2.imwrite(str(d / f"{i:04d}.png"), img)
            lm = np.full((4, N_LANDMARKS, 3), 0.5, np.float32)
            lm[:, 0, :2] = (0.5, 0.65)   # wrist
            lm[:, 9, :2] = (0.5, 0.40)   # middle MCP -> palm radius ~6 px
            LandmarkSequence(cid, lm, np.ones(4, bool), 30.0).save(seqs / f"{cid}.npz")
            clips[cid] = d
        clips["missing"] = tmp_path / "nope"
        ctx = build_clip_context(clips, seqs).set_index("clip_id")
        assert set(ctx.index) == {"pale", "deep"}
        assert ctx.loc["pale", "ita_deg"] > ctx.loc["deep", "ita_deg"]
        assert ctx.loc["pale", "skin_tone"] == "light"
        assert ctx.loc["deep", "skin_tone"] == "dark"

    def test_context_streams_and_aligns_subsampled_frames(self, tmp_path, monkeypatch):
        import cv2

        import asl_fluency.landmarks.context as context

        clip = tmp_path / "clip"
        clip.mkdir()
        for i in range(6):
            cv2.imwrite(str(clip / f"{i:04d}.png"), np.full((8, 8, 3), i * 10, np.uint8))
        landmarks = np.full((3, N_LANDMARKS, 3), 0.5, np.float32)
        seq = LandmarkSequence("clip", landmarks, np.ones(3, bool), 30.0)
        palm_frames = []
        monkeypatch.setattr(context, "_to_lab", lambda frame: frame.astype(float))
        monkeypatch.setattr(
            context,
            "_palm_patch",
            lambda lab, lm: (palm_frames.append(int(lab[0, 0, 0]))
                             or np.ones((1, 3), dtype=float)),
        )
        row = context.clip_context_row(clip, seq, max_frames=16, default_fps=60.0)
        assert palm_frames == [0, 20, 40]
        assert row["n_hand_frames"] == 3
        assert row["luma_L"] == pytest.approx(25.0)

    def test_lighting_tertiles_are_undefined_when_quantiles_tie(self):
        from asl_fluency.landmarks.context import bucket_context

        raw = pd.DataFrame({"clip_id": list("abcd"), "ita_deg": [50.0] * 4,
                            "luma_L": [10.0, 10.0, 20.0, 20.0]})
        out = bucket_context(raw)
        assert out["lighting"].isna().all()


class TestProjectSite:
    """The webpage must re-frame itself as ratings arrive (pilot -> consensus)."""

    @staticmethod
    def _data(n_raters, noise, seed=0):
        from asl_fluency.features.pipeline import FEATURE_SETS

        rng = np.random.default_rng(seed)
        n = 40
        cols = FEATURE_SETS["kinematic"]
        feats = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols)
        feats["clip_id"] = [f"c{i}" for i in range(n)]
        feats["signer_id"] = [i % 10 for i in range(n)]
        feats["detection_rate"] = 1.0
        truth = np.clip(np.rint(3 + feats["log_dimless_jerk"]), 1, 5)
        sheet = pd.concat([
            pd.DataFrame({"clip_id": feats["clip_id"], "rater_id": f"r{k}",
                          "rating": np.clip(np.rint(truth + rng.normal(0, noise, n)), 1, 5)
                          if k < n_raters else np.nan})
            for k in range(3)
        ])
        return sheet, feats

    def test_framing(self):
        from asl_fluency.site import framing

        assert framing(1, None) == "pilot"
        assert framing(3, float("nan")) == "pilot"
        assert framing(3, 0.85) == "consensus"
        assert framing(3, 0.3) == "low_agreement"

    @pytest.mark.parametrize(("n_raters", "noise", "expected"), [
        (1, 0.0, "pilot"), (3, 0.2, "consensus"), (3, 3.0, "low_agreement"),
    ])
    def test_results_and_page_follow_the_ratings(self, tmp_path, n_raters, noise, expected):
        import json

        from asl_fluency.site import compute_site_results, render_site

        sheet, feats = self._data(n_raters, noise)
        res = compute_site_results(sheet, feats, n_splits=3, bootstrap_iters=50)
        assert res["framing"] == expected
        assert res["n_active_raters"] == n_raters
        assert res["hypothesis"] is not None
        assert (res["agreement"] is None) == (n_raters == 1)

        index = render_site(res, tmp_path, {"title": "T", "tagline": "t", "course": "C",
                                            "authors": ["A"], "repo_url": "https://x"})
        page = index.read_text()
        banner = {"pilot": "single-rater pilot", "consensus": "consensus of 3 raters",
                  "low_agreement": "low agreement"}[expected]
        assert banner in page
        assert "(coming soon)" in page  # empty links render disabled, not broken
        json.loads((tmp_path / "results.json").read_text())  # written with allow_nan=False
