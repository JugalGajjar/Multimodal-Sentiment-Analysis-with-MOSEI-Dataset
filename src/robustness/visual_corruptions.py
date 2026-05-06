"""Feature-level corruptions on cached visual features.

For VideoMAE patch sequences, "patch dropout" is the feature-level
analogue of frame dropout / random crop on raw video. For OpenFace2
sequences (MOSEI), patch_dropout drops random frames since OpenFace2
features are per-frame rather than per-patch.
"""

from __future__ import annotations

import torch


def patch_dropout(
    features: torch.Tensor,
    p: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Zero out a fraction ``p`` of patches/frames per sample."""
    if p <= 0:
        return features
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1]; got {p}")
    b, l, _ = features.shape
    keep = (torch.rand(b, l, device=features.device, generator=generator) > p).to(features.dtype)
    return features * keep.unsqueeze(-1)


def gaussian_noise(
    features: torch.Tensor,
    std: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add per-patch Gaussian noise (analogue of pixel-level noise)."""
    if std < 0:
        raise ValueError(f"std must be non-negative; got {std}")
    if std == 0:
        return features
    noise = torch.randn(features.shape, device=features.device, generator=generator)
    return features + std * noise.to(features.dtype)
