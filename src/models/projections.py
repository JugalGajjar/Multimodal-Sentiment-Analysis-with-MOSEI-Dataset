"""Per-modality input projection to the shared X-MoFE dimension.

Each modality has its own input dimension (768 for ModernBERT/WavLM/VideoMAE;
74 for COVAREP, 713 for OpenFace2 on MOSEI). This module standardizes them
to a single ``shared_dim`` with LayerNorm + dropout for stable training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModalityProjection(nn.Module):
    """Linear projection from per-modality input dim to the shared dim.

    Pre-LayerNorm pattern: normalize the input first so downstream layers
    operate on a normalized signal regardless of the encoder's output scale
    (ModernBERT vs WavLM vs handcrafted COVAREP differ noticeably).
    """

    def __init__(self, input_dim: int, shared_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, shared_dim)
        self.out_norm = nn.LayerNorm(shared_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, input_dim) -> (B, L, shared_dim)
        return self.dropout(self.out_norm(self.proj(self.input_norm(x))))
