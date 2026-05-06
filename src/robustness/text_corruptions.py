"""Feature-level corruptions on cached text features.

We operate on encoder outputs (ModernBERT hidden states), not raw tokens,
because the trainer/evaluator consumes cached features. The closest
analogue of "ASR-like deletion" at the feature level is per-token
zero-out — equivalent to setting an embedding to the all-zero pad
representation. The paper should describe this as feature-level corruption.
"""

from __future__ import annotations

import torch


def token_dropout(
    features: torch.Tensor,
    p: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Zero out a fraction ``p`` of token feature vectors per sample.

    Padding positions are already zero, so the dropout effectively only
    removes valid tokens (no separate length-aware mask needed).
    """
    if p <= 0:
        return features
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1]; got {p}")
    b, l, _ = features.shape
    keep = (torch.rand(b, l, device=features.device, generator=generator) > p).to(features.dtype)
    return features * keep.unsqueeze(-1)


def feature_noise(
    features: torch.Tensor,
    std: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add Gaussian noise to every token feature vector."""
    if std <= 0:
        return features
    noise = torch.randn(features.shape, device=features.device, generator=generator)
    return features + std * noise.to(features.dtype)
