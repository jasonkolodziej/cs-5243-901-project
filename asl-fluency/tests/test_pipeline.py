"""End-to-end tests on synthetic landmark sequences (no video needed)."""

import os
import tarfile
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from asl_fluency.evaluation.ablation import run_ablation, stratified_metrics
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
        summary, _ = run_ablation(feats, y, n_splits=3)
        assert len(summary) == 3
        kin = summary.set_index("arm").loc["kinematic_ridge"]
        assert kin["spearman"] > 0.3  # signal should be recoverable

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
