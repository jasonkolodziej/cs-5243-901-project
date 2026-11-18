"""Configuration loading utilities."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def project_root() -> Path:
    """Absolute path to the project root (the folder containing configs/)."""
    return DEFAULT_CONFIG_PATH.parents[1]


def load_dotenv(path: str | Path | None = None) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ.

    No third-party dependency; supports the simple subset of dotenv syntax
    this project needs (blank lines, `#` comments, optional quotes).
    Variables already set in the shell take precedence and are NOT
    overwritten. Called automatically by load_config(); missing file is fine.
    """
    env_path = Path(path) if path else project_root() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config, falling back to configs/default.yaml.

    A user-supplied config is deep-merged over the defaults so partial
    overrides are enough. Also loads .env from the project root so API
    credentials (e.g. XAI_API_KEY) are available to backends without being
    stored in YAML.
    """
    load_dotenv()
    with open(DEFAULT_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    if path is not None and Path(path) != DEFAULT_CONFIG_PATH:
        with open(path) as f:
            override = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, override)
    return cfg


def ai_judge_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """The ai_judge section with defaults applied.

    API credentials deliberately do NOT live here — backends read them from
    the environment (see example.env), so committed YAML never holds secrets.
    """
    section = dict(cfg.get("ai_judge") or {})
    section.setdefault("backend", "mock")
    section.setdefault("model", "grok-4.7")
    section.setdefault("n_frames", 16)
    section.setdefault("temperature", 0)
    section.setdefault("jpeg_quality", 80)
    return section


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a path from cfg['paths'] relative to the project root."""
    root = DEFAULT_CONFIG_PATH.parents[1]
    return root / cfg["paths"][key]
