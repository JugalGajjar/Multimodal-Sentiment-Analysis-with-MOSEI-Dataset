"""Feature-level corruptions on cached audio features.

WavLM frame features and COVAREP frames have different scales (768-dim
neural vs 74-dim handcrafted z-scored), so the noise standard deviation
is interpreted *relative to the pre-cached signal* — for fp16-cached
features at unit-ish std, ``std=0.1`` corresponds to ~10% perturbation.
"""

from __future__ import annotations

import torch


def gaussian_noise(
    features: torch.Tensor,
    std: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add per-frame Gaussian noise to the audio feature tensor."""
    if std < 0:
        raise ValueError(f"std must be non-negative; got {std}")
    if std == 0:
        return features
    noise = torch.randn(features.shape, device=features.device, generator=generator)
    return features + std * noise.to(features.dtype)


def frame_dropout(
    features: torch.Tensor,
    p: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Zero out a fraction ``p`` of audio frames per sample (silence masking)."""
    if p <= 0:
        return features
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1]; got {p}")
    b, l, _ = features.shape
    keep = (torch.rand(b, l, device=features.device, generator=generator) > p).to(features.dtype)
    return features * keep.unsqueeze(-1)
