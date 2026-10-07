"""End-to-end tests on synthetic landmark sequences (no video needed)."""

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
        (tmp_path / "ChicagoFSWild.csv").write_text(
            ",filename,label_proc,signer\n0,a/b_0001,hi,0000\n"
        )
        ci = load_clip_index(tmp_path)
        assert len(ci) == 1
        assert ci.frame.iloc[0]["transcript"] == "HI"


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
