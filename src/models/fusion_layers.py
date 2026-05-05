"""Explainable hybrid fusion: ``h = LayerNorm(u + i) → FFN``.

Combines the reliability-weighted unimodal sum ``u`` with the
interaction-weighted contribution ``i`` (per spec §12.6). Pre-LN around the
FFN keeps training stable; the residual preserves the linear-fusion signal
through deeper stacks if we ever increase ``fusion_layers``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HybridFusion(nn.Module):
    def __init__(
        self,
        shared_dim: int,
        dropout: float = 0.2,
        ffn_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(shared_dim)
        self.ffn = nn.Sequential(
            nn.Linear(shared_dim, shared_dim * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim * ffn_multiplier, shared_dim),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(shared_dim)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        h = u + i                                  # linear fusion of two evidence sources
        h = h + self.ffn(self.norm(h))             # pre-LN residual FFN
        return self.final_norm(h)
