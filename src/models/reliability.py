"""Modality-reliability estimator producing ``r = [r_T, r_A, r_V]``.

Optionally consumes per-modality quality features (audio SNR, visual face
confidence, etc.) so the paper can validate that learned reliability scores
correlate with objective signal quality. For v1 we leave the API in place
but pass ``None`` — the quality vector slot is preserved with zeros so
adding real quality features later is a one-config-flip change.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReliabilityEstimator(nn.Module):
    """``r = softmax(MLP_R([z_T; z_A; z_V; q_T; q_A; q_V]))``.

    ``num_quality_features_per_modality`` is the size of each modality's
    quality feature vector. If 0 (the default), the input is just the three
    pooled modality embeddings.
    """

    def __init__(
        self,
        shared_dim: int,
        mlp_hidden: int = 256,
        dropout: float = 0.2,
        num_quality_features_per_modality: int = 0,
    ) -> None:
        super().__init__()
        self.num_quality_features_per_modality = num_quality_features_per_modality
        in_dim = shared_dim * 3 + num_quality_features_per_modality * 3

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 3),
        )

    def forward(
        self,
        z_t: torch.Tensor, z_a: torch.Tensor, z_v: torch.Tensor,
        quality: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.cat([z_t, z_a, z_v], dim=-1)
        if self.num_quality_features_per_modality > 0:
            expected = self.num_quality_features_per_modality * 3
            if quality is None:
                # Keep the API stable: zero-pad when callers haven't wired
                # quality features yet.
                quality = z_t.new_zeros(z_t.size(0), expected)
            elif quality.size(-1) != expected:
                raise ValueError(
                    f"quality features have last dim {quality.size(-1)}; "
                    f"expected {expected}"
                )
            x = torch.cat([x, quality], dim=-1)
        return F.softmax(self.mlp(x), dim=-1)
