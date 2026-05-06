"""Entropy regularization on the reliability distribution.

Keeps reliability scores informative (not collapsed to one modality) without
forcing them to be uniform. The penalty is the absolute distance between the
mean reliability entropy across the batch and a configured target.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EntropyLoss(nn.Module):
    """``L_entropy = | H(r) - τ |`` averaged over the batch."""

    def __init__(self, target_entropy: float = 0.7, eps: float = 1e-12) -> None:
        super().__init__()
        self.target_entropy = float(target_entropy)
        self.eps = eps

    def forward(self, distribution: torch.Tensor) -> torch.Tensor:
        # distribution: (B, K) — softmax probs
        entropy_per_sample = -(distribution * (distribution + self.eps).log()).sum(dim=-1)
        return (entropy_per_sample.mean() - self.target_entropy).abs()
