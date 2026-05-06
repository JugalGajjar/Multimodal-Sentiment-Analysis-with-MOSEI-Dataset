"""Early-fusion baseline.

Pool each modality, concatenate the three pooled vectors, run an MLP, predict.
No cross-modal attention, no reliability, no interaction attribution. Exists
to isolate the *fusion strategy* in the controlled comparison: same encoders,
same projections, same attention pooling — only the fusion changes.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from src.models.attention_pooling import AttentionPool
from src.models.baselines._common import uniform_interactions, uniform_reliability
from src.models.explanation_heads import XMoFEOutput
from src.models.prediction_heads import PredictionHead
from src.models.projections import ModalityProjection


class EarlyFusionModel(nn.Module):
    """``[z_T; z_A; z_V] → MLP → ŷ``."""

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        visual_dim: int,
        shared_dim: int = 256,
        dropout: float = 0.2,
        ffn_multiplier: int = 2,
        task: str = "regression",
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.shared_dim = shared_dim
        self.task = task

        self.proj_text = ModalityProjection(text_dim, shared_dim, dropout)
        self.proj_audio = ModalityProjection(audio_dim, shared_dim, dropout)
        self.proj_visual = ModalityProjection(visual_dim, shared_dim, dropout)

        self.pool_text = AttentionPool(shared_dim)
        self.pool_audio = AttentionPool(shared_dim)
        self.pool_visual = AttentionPool(shared_dim)

        self.fusion = nn.Sequential(
            nn.Linear(3 * shared_dim, shared_dim * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim * ffn_multiplier, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.Dropout(dropout),
        )
        self.head = PredictionHead(shared_dim, task=task, num_classes=num_classes, dropout=dropout)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        text_dim: int,
        audio_dim: int,
        visual_dim: int,
        task: str,
        num_classes: int = 1,
    ) -> "EarlyFusionModel":
        return cls(
            text_dim=text_dim, audio_dim=audio_dim, visual_dim=visual_dim,
            shared_dim=config.get("shared_dim", 256),
            dropout=config.get("dropout", 0.2),
            ffn_multiplier=config.get("ffn_multiplier", 2),
            task=task, num_classes=num_classes,
        )

    def forward(
        self,
        text: torch.Tensor, audio: torch.Tensor, visual: torch.Tensor,
        text_length: torch.Tensor, audio_length: torch.Tensor, visual_length: torch.Tensor,
        quality: torch.Tensor | None = None,  # noqa: ARG002 — kept for signature compat
        return_intermediates: bool = True,
    ) -> XMoFEOutput:
        Z_t = self.proj_text(text)
        Z_a = self.proj_audio(audio)
        Z_v = self.proj_visual(visual)

        z_t, alpha_t = self.pool_text(Z_t, text_length)
        z_a, alpha_a = self.pool_audio(Z_a, audio_length)
        z_v, alpha_v = self.pool_visual(Z_v, visual_length)

        h = self.fusion(torch.cat([z_t, z_a, z_v], dim=-1))
        prediction = self.head(h)

        b = text.size(0)
        return XMoFEOutput(
            prediction=prediction,
            reliability=uniform_reliability(b, device=text.device),
            interactions=uniform_interactions(b, num_interactions=1, device=text.device),
            temporal_attention={"text": alpha_t, "audio": alpha_a, "visual": alpha_v},
            interaction_names=("early_concat",),
            pooled_modalities=(
                {"text": z_t, "audio": z_a, "visual": z_v} if return_intermediates else None
            ),
            fused=h if return_intermediates else None,
        )
