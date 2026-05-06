"""Shared helpers for fusion baselines.

The placeholders here let every baseline return the same ``XMoFEOutput``
shape as :class:`src.models.xmofe.XMoFE`, so the trainer and evaluator are
agnostic to which variant is in use. Callers should not interpret the
placeholder reliability or interaction values as real explanations — those
are exclusive to X-MoFE.
"""

from __future__ import annotations

import torch


def uniform_reliability(batch_size: int, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``(B, 3)`` of 1/3 — placeholder for baselines without modality reliability."""
    return torch.full((batch_size, 3), 1.0 / 3.0, device=device, dtype=dtype)


def uniform_interactions(batch_size: int, num_interactions: int = 1, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``(B, K)`` of 1/K — placeholder for baselines without interaction attribution."""
    if num_interactions < 1:
        num_interactions = 1
    return torch.full((batch_size, num_interactions), 1.0 / num_interactions, device=device, dtype=dtype)


def zeros_attention(batch_size: int, length: int, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``(B, L)`` zeros — placeholder when a modality is unused (unimodal baselines)."""
    return torch.zeros(batch_size, length, device=device, dtype=dtype)
