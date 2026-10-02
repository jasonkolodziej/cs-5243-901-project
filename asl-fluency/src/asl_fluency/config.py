"""Configuration loading utilities."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config, falling back to configs/default.yaml.

    A user-supplied config is deep-merged over the defaults so partial
    overrides are enough.
    """
    with open(DEFAULT_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    if path is not None and Path(path) != DEFAULT_CONFIG_PATH:
        with open(path) as f:
            override = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, override)
    return cfg


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
