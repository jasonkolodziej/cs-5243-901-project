"""Tests for the AI-judge backends, .env loading, and ai_judge config wiring.

No network access: the Grok API client is faked via sys.modules injection.
"""

import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from asl_fluency.config import ai_judge_config, load_config, load_dotenv, project_root


def _write_frames(dir_path, n=12, size=(16, 16)):
    """Frame dir where frame i is filled with a distinct value (order checks)."""
    import cv2

    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = np.full((*size, 3), i * 6 % 255, dtype=np.uint8)  # monotonic for n<=42
        cv2.imwrite(str(dir_path / f"{i:04d}.jpg"), img)
    return dir_path


def _fake_openai_module(rating=4, recorded=None):
    """A stand-in `openai` module whose client records the request."""
    mod = types.ModuleType("openai")

    class FakeCompletions:
        def create(self, **kwargs):
            if recorded is not None:
                recorded.append(kwargs)
            body = json.dumps(
                {"rating": rating, "confidence": "medium", "rationale": "fake"}
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=body))]
            )

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None):
            if recorded is not None:
                recorded.append({"api_key": api_key, "base_url": base_url})
            self.chat = SimpleNamespace(completions=FakeCompletions())

    mod.OpenAI = FakeOpenAI
    return mod


class TestDotenv:
    def test_loads_simple_key_value(self, tmp_path, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        env = tmp_path / ".env"
        env.write_text("# comment\n\nXAI_API_KEY=sk-test-123\n")
        load_dotenv(env)
        import os

        assert os.environ["XAI_API_KEY"] == "sk-test-123"

    def test_shell_vars_win_over_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "from-shell")
        env = tmp_path / ".env"
        env.write_text("XAI_API_KEY=from-file\n")
        load_dotenv(env)
        import os

        assert os.environ["XAI_API_KEY"] == "from-shell"

    def test_missing_file_is_silent(self, tmp_path):
        load_dotenv(tmp_path / "does-not-exist.env")  # must not raise

    def test_quoted_and_spaced_values(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MY_TOKEN", raising=False)
        env = tmp_path / ".env"
        env.write_text('MY_TOKEN = "abc def"\n')
        load_dotenv(env)
        import os

        assert os.environ["MY_TOKEN"] == "abc def"

    def test_example_env_exists_and_env_is_gitignored(self):
        root = project_root()
        assert (root / "example.env").exists(), "example.env must ship with the repo"
        gitignore = (root / ".gitignore").read_text()
        assert ".env" in gitignore.split(), ".env must be gitignored (secrets)"


class TestAIJudgeConfig:
    def test_defaults_when_section_missing(self):
        j = ai_judge_config({})
        assert j["backend"] == "mock"
        assert j["model"] == "grok-4.7"
        assert j["n_frames"] == 16

    def test_default_yaml_carries_section(self):
        j = ai_judge_config(load_config())
        assert j["backend"] == "mock"  # safe default: no API calls by accident
        assert "api_key" not in j and "key" not in j, "no secrets in YAML config"

    def test_override_yaml_deep_merges(self, tmp_path):
        override = tmp_path / "exp.yaml"
        override.write_text("ai_judge:\n  backend: grok\n  n_frames: 8\n")
        j = ai_judge_config(load_config(override))
        assert j["backend"] == "grok"
        assert j["n_frames"] == 8
        assert j["model"] == "grok-4.7"  # untouched defaults survive the merge


class TestGrokBackend:
    def test_missing_api_key_gives_guidance(self, tmp_path, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        from asl_fluency.ai_judge import score_clip

        with pytest.raises(RuntimeError, match="XAI_API_KEY is not set"):
            score_clip(tmp_path / "clip", backend="grok")

    def test_missing_openai_package_gives_guidance(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "sk-test")
        monkeypatch.setitem(sys.modules, "openai", None)  # force ImportError
        from asl_fluency.ai_judge import score_clip

        with pytest.raises(RuntimeError, match="uv add openai"):
            score_clip(tmp_path / "clip", backend="grok")

    def test_unknown_backend_lists_available(self, tmp_path):
        from asl_fluency.ai_judge import score_clip

        with pytest.raises(NotImplementedError, match="mock, grok"):
            score_clip(tmp_path / "clip", backend="gemini")

    def test_request_shape_and_response_parsing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "sk-test")
        recorded = []
        monkeypatch.setitem(sys.modules, "openai", _fake_openai_module(4, recorded))
        from asl_fluency.ai_judge import score_clip

        clip = _write_frames(tmp_path / "clip_a", n=12)
        work = tmp_path / "raw"
        out = score_clip(
            clip, backend="grok", work_dir=work, clip_id="clip_a",
            model="grok-4.7", n_frames=6, temperature=0,
        )

        assert out["rating"] == 4
        # client constructed against the xAI endpoint with the env key
        assert recorded[0] == {"api_key": "sk-test", "base_url": "https://api.x.ai/v1"}
        # request: json mode, requested model, 1 text part + n_frames image parts
        req = recorded[1]
        assert req["model"] == "grok-4.7"
        assert req["response_format"] == {"type": "json_object"}
        content = req["messages"][0]["content"]
        kinds = [part["type"] for part in content]
        assert kinds == ["text"] + ["image_url"] * 6
        assert "temporal order" in content[0]["text"]  # judge told frames are ordered
        assert all(
            p["image_url"]["url"].startswith("data:image/jpeg;base64,")
            for p in content[1:]
        )
        # raw response saved for auditability
        saved = json.loads((work / "clip_a.json").read_text())
        assert saved["rating"] == 4

    def test_frames_sampled_in_temporal_order(self, tmp_path):
        from asl_fluency.ai_judge import _sample_frames

        clip = _write_frames(tmp_path / "clip_b", n=40)
        frames = _sample_frames(clip, 8)
        assert len(frames) == 8
        vals = [int(f[0, 0, 0]) for f in frames]
        assert vals == sorted(vals), "frames must keep temporal order"
        # decoded JPEG values track the distinct per-frame fills (lossy: tolerance)
        assert vals[-1] > vals[0]

    def test_out_of_range_rating_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "sk-test")
        monkeypatch.setitem(sys.modules, "openai", _fake_openai_module(rating=9))
        from asl_fluency.ai_judge import score_clip

        clip = _write_frames(tmp_path / "clip_c", n=4)
        with pytest.raises(AssertionError, match="out of range"):
            score_clip(clip, backend="grok")


class TestScoreOptsPlumbing:
    def test_run_ai_rating_forwards_opts(self, tmp_path, monkeypatch):
        import asl_fluency.ai_judge as aj

        seen = {}

        def fake_grok(clip_path, work_dir, clip_id, **opts):
            seen.update(opts)
            return {"rating": 3, "confidence": "low", "rationale": "stub"}

        monkeypatch.setattr(aj, "_score_grok", fake_grok)
        clips = {"c0": tmp_path / "c0"}
        aj.run_ai_rating(
            clips, tmp_path / "ai.csv", backend="grok",
            model="grok-4.7", n_frames=8, temperature=0, jpeg_quality=70,
        )
        assert seen["model"] == "grok-4.7"
        assert seen["n_frames"] == 8
        assert seen["jpeg_quality"] == 70

    def test_mock_backend_tolerates_extra_opts(self, tmp_path):
        from asl_fluency.ai_judge import score_clip

        r = score_clip(tmp_path / "clip", backend="mock", model="grok-4.7", n_frames=4)
        assert 1 <= r["rating"] <= 5
