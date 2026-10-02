"""Optional fused model: kinematic features + raw-video temporal embeddings.

Ablation arm from proposal Section 6 (kinematic-only vs. kinematic + video).
Requires the `fused` extra:  uv sync --extra fused

Design: a small 3D-CNN (or an off-the-shelf video backbone such as a
VideoMAE/S3D checkpoint) encodes short clips around each letter span; the
resulting embedding vector is concatenated with the kinematic feature row
and fed to the same regressors as models.baseline. Kept deliberately thin —
Section 7 of the proposal warns the small rated set cannot support a large
end-to-end model, so the embedding is intended to be frozen/pretrained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_video_embeddings(
    video_path: str,
    spans: list[tuple[int, int, str]],
    backbone: str = "r3d_18",
) -> np.ndarray:
    """Per-letter-span clip embeddings, shape (n_spans, embed_dim).

    TODO: implement with torchvision's r3d_18 (Kinetics-pretrained) or swap
    in a sign-language-pretrained backbone. Returns NaNs until implemented so
    the ablation runner degrades gracefully.
    """
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Fused model needs the torch stack: uv sync --extra fused"
        ) from exc
    raise NotImplementedError(
        "Implement clip extraction + backbone forward pass here; "
        "see module docstring for the intended design."
    )


class FusedRegressor:
    """Concatenate kinematic features with video embeddings, then regress."""

    def __init__(self, base_kind: str = "ridge", seed: int = 42) -> None:

        self.base_kind = base_kind
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: np.ndarray):  # pragma: no cover
        raise NotImplementedError("Implement after extract_video_embeddings.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError("Implement after extract_video_embeddings.")
